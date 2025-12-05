# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import torch.nn.functional as F
import numpy as np
import torchgeometry as tgm
import misc_utils, eulerangles
from tqdm import tqdm
import pdb
import open3d as o3d
import sys
sys.path.append("D:\code\python\POSA\src\gen_human")
import viz_utils


def compute_afford_loss(vertices=None, scene_data=None, gen_batch=None, pen_w=0.0, no_obj_classes=None,
                        use_semantics=False,
                        semantics_w=0.0, **kwargs):
    contact_ids = gen_batch[:, :, 0] > 0.5 #每一个顶点的触点的概率么？ 
    x = misc_utils.read_sdf(vertices, scene_data['sdf'], # x是什么呢？
                            scene_data['grid_dim'], scene_data['grid_min'], scene_data['grid_max'],
                            mode="bilinear").squeeze()

    batch_size = vertices.shape[0]
    device = vertices.device

    contact_loss = torch.sum(x[contact_ids.flatten()] ** 2) #需要接触的点的位置采样处的SDF值的平方

    pen_loss = torch.tensor(0.0)

    if pen_w > 0:
        mask = x.lt(0).flatten().int() + (~contact_ids.flatten()).int()
        x_neg = torch.abs(x[mask == 2]) #人体mesh的顶点位置处 SDF值为负（该点在场景表面内部）并且 该顶点的不是与场景的接触点 ；取该点的SDF绝对值
        if len(x_neg) == 0:
            pen_loss = torch.tensor(0.0)
        else:
            pen_loss = pen_w * x_neg.sum()

    semantics_loss = torch.tensor(0.0)
    if use_semantics:
        # Read semantics
        x_semantics = misc_utils.read_sdf(vertices, scene_data['semantics'],
                                          scene_data['grid_dim'], scene_data['grid_min'],
                                          scene_data['grid_max'], mode="bilinear").squeeze()
        x_semantics = contact_ids.flatten().float() * x_semantics.unsqueeze(0)
        x_semantics = torch.zeros(x_semantics.shape[0], x_semantics.shape[1], no_obj_classes, device=device).scatter_(
            -1, x_semantics.unsqueeze(-1).type(torch.long), 1.)
        # Compute loss
        targets = gen_batch[:, :, 1:].argmax(dim=-1).type(torch.long).reshape(batch_size, -1)
        semantics_loss = semantics_w * F.cross_entropy(x_semantics.permute(0, 2, 1), targets,
                                                       reduction='sum')

    return contact_loss, pen_loss, semantics_loss

def eval_init_points(init_pos=None, init_ang=None, vertices=None, scene_data=None, gen_batch=None, **kwargs):
    with torch.no_grad():
        losses = []
        init_pos_batches = init_pos.split(1)

        for i in tqdm(range(len(init_pos_batches))):
            curr_init_pos = init_pos_batches[i]
            rot_aa = torch.cat((torch.zeros((1, 2), device=vertices.device), init_ang[i].reshape(1, 1)), dim=1) #(0,0,angle)沿着z轴的旋转
            rot_mat = tgm.angle_axis_to_rotation_matrix(rot_aa.reshape(-1, 3))[:, :3, :3] #将旋转角度转化为旋转矩阵
            curr_vertices = torch.bmm(rot_mat, vertices.permute(0, 2, 1)).permute(0, 2, 1) #矩阵乘法，这里的permute操作没有什么特别的含义
            curr_vertices = curr_vertices + curr_init_pos
            contact_loss, pen_loss, semantics_loss = compute_afford_loss(vertices=curr_vertices, scene_data=scene_data,
                                                                         gen_batch=gen_batch, **kwargs)
            loss = contact_loss + pen_loss + semantics_loss
            losses.append(loss.item())

        # Sort initial positions and orientations from best to wrost
        losses = np.array(losses)
        ids = np.argsort(losses)
        losses = losses[ids]
        init_pos = init_pos[ids]
        init_ang = init_ang[ids]

        return losses, init_pos, init_ang

def init_points_culling(init_pos=None, vertices=None, scene_data=None, gen_batch=None, max_init_points=50, **kwargs):
    init_ang = []
    angles = torch.arange(0, 2 * np.pi, np.pi / 2, device=vertices.device) #生成向量（0，pi/2,pi,3pi/2）
    angles[0] = 1e-9 #将第一维设置为很小的量，避免为0
    for ang in angles:
        init_ang.append(ang * torch.ones(init_pos.shape[0], 1, device=vertices.device))
    # 如果得到N个初始点；
    # 得到init_ang =  [{列向量1},{列向量2},{列向量3},{列向量4}]；每个列向量的长度为N，值全部为{0，pi/2，pi，3pi/2}
    init_ang = torch.cat(init_ang).to(init_pos.device)
    # cat之后，变成一个列向量，列向量的长度为4*N 一次是N 个{0，pi/2，pi，3pi/2}
    init_pos = init_pos.repeat(angles.shape[0], 1, 1) #init_pos重复四次
    # Shuffle
    rnd_ids = np.random.choice(init_pos.shape[0], init_pos.shape[0], replace=False)
    init_pos = init_pos[rnd_ids, :]
    init_ang = init_ang[rnd_ids, :]

    losses, init_pos, init_ang = eval_init_points(init_pos=init_pos, init_ang=init_ang,
                                                  vertices=vertices.unsqueeze(0),
                                                  scene_data=scene_data, gen_batch=gen_batch, **kwargs)
    # Select only a subset from initial points for optimization
    if init_pos.shape[0] > max_init_points:
        init_pos = init_pos[:max_init_points]
        init_ang = init_ang[:max_init_points]
    return init_pos, init_ang

def init_points_culling_targetobject(body_model = None,init_pos=None, vertices=None, scene_data=None, gen_batch=None, max_init_points=50, **kwargs):
    init_ang = []
    #计算人体的骨盆和物体中心点的方向
    target_object_center = kwargs['target_object_pos']
    target_object_center = torch.tensor([target_object_center],device = kwargs['device']).unsqueeze(0)
    #每个init_pos计算一个角度：面向物体的方向
    # pdb.set_trace()
    for i in range(init_pos.shape[0]):
        #计算（0，-1，0）在场景坐标下的朝向
        # R_sml2scene = torch.tensor(eulerangles.euler2mat(np.pi / 2, 0, 0, 'sxyz'), dtype=kwargs['dtype'],
                                                # device=kwargs['device'])
        orig_orient = torch.tensor([0,-1,0],device = kwargs['device'],dtype=kwargs['dtype'])
        # orig_orient = torch.matmul(R_sml2scene,orig_orient)
        orig_orient[2] = 0 #投影到xy平面上
        orig_orient /= torch.norm(orig_orient)
        #计算面向物体的朝向 
        look_at_object = (target_object_center - init_pos[i]).float() #torch.Size([1, 1, 3])
        look_at_object[0][0][2] = 0
        look_at_object /= torch.norm(look_at_object)
        look_at_object = look_at_object.reshape(-1)
        
        ang = torch.arccos(torch.dot(orig_orient,look_at_object)).unsqueeze(0)
        init_ang.append(ang)
    # pdb.set_trace()
    init_ang = torch.tensor(init_ang)
    init_ang = init_ang.view(init_pos.shape[0],1,1).to(init_pos.device)
    
    #shuffle
    rnd_ids = np.random.choice(init_pos.shape[0], init_pos.shape[0], replace=False)
    init_pos = init_pos[rnd_ids, :]
    init_ang = init_ang[rnd_ids, :]

    losses, init_pos, init_ang = eval_init_points(init_pos=init_pos, init_ang=init_ang,
                                                  vertices=vertices.unsqueeze(0),
                                                  scene_data=scene_data, gen_batch=gen_batch, **kwargs)
    # Select only a subset from initial points for optimization
    if init_pos.shape[0] > max_init_points:
        init_pos = init_pos[:max_init_points]
        init_ang = init_ang[:max_init_points]
    return init_pos, init_ang

class opt_wrapper(object):
    def __init__(self, vertices=None, vertices_can=None, pelvis=None, scene_data=None,
                 down_sample_fn=None, down_sample_fn2=None,
                 device=None, dtype=None, pen_w=None, use_semantics=None, no_obj_classes=None, nv=None, optimizer=None,
                 gen_batch=None, body_model=None, opt_pose=False,
                 semantics_w=None, init_body_pose=None, pose_w=None, **kwargs):

        self.optimizer = optimizer
        self.vertices = vertices
        self.vertices_can = vertices_can
        self.pelvis = pelvis
        self.scene_data = scene_data
        self.down_sample_fn = down_sample_fn
        self.down_sample_fn2 = down_sample_fn2
        self.device = device
        self.dtype = dtype
        self.pen_w = pen_w
        self.pose_w = pose_w
        self.semantics_w = semantics_w
        self.use_semantics = use_semantics
        self.no_obj_classes = no_obj_classes
        self.nv = nv
        self.gen_batch = gen_batch
        self.opt_pose = opt_pose
        self.body_model = body_model
        self.init_body_pose = init_body_pose
        self.R_smpl2scene = torch.tensor(eulerangles.euler2mat(np.pi / 2, 0, 0, 'sxyz'), dtype=dtype, device=device)
        self.kwargs = kwargs

    def compute_vertices(self, t_free, y_ang, vertices=None, down_sample=True):
        curr_batch_size = self.vertices.shape[0]
        rot_aa = torch.cat((torch.zeros((curr_batch_size, 2), device=self.device), y_ang), dim=1)
        rot_mat = tgm.angle_axis_to_rotation_matrix(rot_aa.reshape(-1, 3))[:, :3, :3]

        if self.opt_pose:
            body_model_output = self.body_model(return_verts=True)
            pelvis = body_model_output.joints[:, 0, :].reshape(1, 3)
            vertices_local = body_model_output.vertices.squeeze()
            vertices_local = torch.matmul(self.R_smpl2scene, (vertices_local - pelvis).t()).t()
            vertices_local.unsqueeze_(0)
            if down_sample:
                vertices_local = self.down_sample_fn.forward(vertices_local.permute(0, 2, 1))
                vertices_local = self.down_sample_fn2.forward(vertices_local).permute(0, 2, 1)

            vertices_local = torch.bmm(rot_mat, vertices_local.permute(0, 2, 1)).permute(0, 2, 1)
            vertices_local += t_free

        else:
            # very important to make a local copy, so that you don't change the original variable
            if vertices is None:
                vertices_local = torch.bmm(rot_mat, self.vertices.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                vertices_local = torch.bmm(rot_mat, vertices.permute(0, 2, 1)).permute(0, 2, 1)
            vertices_local += t_free
        return vertices_local, rot_mat

    def compute_loss(self, t_free, y_ang):
        pose_loss = torch.tensor(0.0)
        if self.opt_pose:
            pose_loss = self.pose_w * F.mse_loss(self.body_model.body_pose, self.init_body_pose)
        #优化位姿和初始位姿差别越小越好
        vertices_local, rot_mat = self.compute_vertices(t_free, y_ang)

        contact_loss, pen_loss, semantic_loss = compute_afford_loss(vertices=vertices_local, scene_data=self.scene_data,
                                                                    gen_batch=self.gen_batch, pen_w=self.pen_w,
                                                                    no_obj_classes=self.no_obj_classes,
                                                                    use_semantics=self.use_semantics,
                                                                    semantics_w=self.semantics_w)
        return contact_loss, pen_loss, pose_loss, semantic_loss
    
    def compute_loss_object(self, t_free, y_ang):
            pose_loss = torch.tensor(0.0)
            if self.opt_pose:
                pose_loss = self.pose_w * F.mse_loss(self.body_model.body_pose, self.init_body_pose)
            #优化位姿和初始位姿差别越小越好
            vertices_local, rot_mat = self.compute_vertices(t_free, y_ang)

            contact_loss, pen_loss, semantic_loss = compute_afford_loss(vertices=vertices_local, scene_data=self.scene_data,
                                                                        gen_batch=self.gen_batch, pen_w=self.pen_w,
                                                                        no_obj_classes=self.no_obj_classes,
                                                                        use_semantics=self.use_semantics,
                                                                        semantics_w=self.semantics_w)
            
            # # 计算朝向物体的loss lookfowars_loss
            # #首先计算人体的朝向,假设初始的朝向为(1,0,0)
            # #需要将旋转角转化为方向向量 因为只绕z轴旋转，只需要将(0,0,1)转化到场景坐标系下
            # pdb.set_trace()
            init_orient = torch.tensor([0,-1,0], device=self.device).float() #torch.Size([3])
            current_orient = torch.matmul(rot_mat,init_orient) 
        
            target_object_center = self.kwargs['target_object_pos']
            target_object_center = torch.tensor([target_object_center],device = self.device).unsqueeze(0)
            body_model_output = self.body_model(return_verts=True)
            head = body_model_output.joints[:, 15, :].reshape(1,3)
            head_scene = torch.matmul(self.R_smpl2scene, (head - self.pelvis).t()) #这里都是没有问题的torch.Size([1, 3])
            head_scene = head_scene.unsqueeze(0)
            head_pos = torch.matmul(rot_mat,head_scene).squeeze(0).t() #rot_mat.shape (1,3,3) 
            head_pos = head_pos.unsqueeze(0)
            head_pos += t_free
            #头部关节点信息转化到场景坐标系下
            # pdb.set_trace()
            lookdirection = (target_object_center - head_pos).float()
            # #计算夹角作为loss
            looktowards_loss = torch.arccos(torch.dot(current_orient.view(-1), lookdirection.view(-1))/(torch.norm(current_orient) * torch.norm(lookdirection)))
            
            #加一个可视化： 物体中心点，人的头部的坐标，以及朝向
            # m_scene =  o3d.io.read_triangle_mesh("D:\data\\POSA_dir\scenes\MPH16.ply")
            # points = [m_scene]  #场景数据
            # points.append(viz_utils.create_o3d_sphere(target_object_center.detach().cpu().numpy().squeeze(), radius=0.03)) #物体中心点
            # points.append(viz_utils.create_o3d_sphere(head_pos.detach().cpu().numpy().squeeze(), radius=0.1)) #人的头部位置
            # body_v = np.array(vertices_local.detach().cpu().numpy().squeeze())

            # # 创建人体点云
            # point_cloud = o3d.geometry.PointCloud()
            # point_cloud.points = o3d.utility.Vector3dVector(body_v)     
            # points.append(point_cloud)
            
            # #朝向信息
            # m_orient = current_orient + head_pos
            # points.append(viz_utils.create_o3d_sphere(m_orient.detach().cpu().numpy().squeeze(), radius=0.03)) #物体中心点
            
            # o3d.visualization.draw_geometries(points)
            
            return contact_loss, pen_loss, pose_loss, semantic_loss, looktowards_loss
            
    def create_fitting_closure(self, t_free, y_ang):
        def fitting_func():
            self.optimizer.zero_grad()
            recon_loss, pen_loss, pose_loss, semantic_loss = self.compute_loss(t_free, y_ang)
            loss_total = recon_loss + pen_loss + pose_loss + semantic_loss
            loss_total.backward(retain_graph=True)
            return loss_total

        return fitting_func

    def create_fitting_closure_object(self, t_free, y_ang):
        def fitting_func():
            self.optimizer.zero_grad()
            recon_loss, pen_loss, pose_loss, semantic_loss , looktowards_loss = self.compute_loss_object(t_free, y_ang)
            loss_total = recon_loss + pen_loss + pose_loss + semantic_loss + 0.25 * looktowards_loss
            loss_total.backward(retain_graph=True)
            return loss_total
        
        return fitting_func
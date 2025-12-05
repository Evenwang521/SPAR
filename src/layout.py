import os
import os.path as osp
import shutil
import numpy as np
import open3d as o3d
import trimesh
import argparse
import pickle
import json
import math
import torch
import torch.nn.functional as F
import torch.optim as Optim
from tqdm import tqdm
import torchgeometry as tgm
import pdb
import datetime
from torch.utils.tensorboard import SummaryWriter
from gen_layout import obb
from gen_layout import xrgonomic_metrics 

from gen_human import eulerangles
from gen_human import misc_utils

from gen_layout.viz_ultis import viz_body_info,viz_obb,viz_results,viz_results_baseline
from gen_layout.data_loader import load_scene_aabbs,create_scene_obbs,create_ele_obbs

from population import self_define_scenes

import time
def my_cmdargs():  
    
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='创建命令行解析器')
    
    # 添加命令行参数
    parser.add_argument('--scenemesh_folder', type=str, default= 'D:\data\\POSA_dir\scenes',help='POSA场景mesh的路径')
    parser.add_argument('--affordance_folder', type=str, default='D:\data\\POSA_dir\\affordance',help='POSA生成结果的路径')
    parser.add_argument('--sdf_folder', type=str, default='D:\data\\POSA_dir\sdf', help='场景SDF的路径' )
    parser.add_argument('--scene_aabbs_folder',type=str,default='D:\code\python\POSA\output\scenes_ins_aabb',help="存储场景分割的包围盒")
    parser.add_argument('--scene_name', type=str, default= 'MPH16',help='场景名称')
    parser.add_argument('--elements_folder', type = str, default = 'D:\data\\POSA_dir\\virtual_elements',help = '存储需要排布的UI元素')
    parser.add_argument('--num_pts', type=int, default=1024, help='对每个mesh的采样点')
    parser.add_argument('--num_samples', type=int, default=8192, help='iou估计的采样点数')
    parser.add_argument('--body_name', type=str, default="rp_aaron_posed_001_0_0_00_05", help='放置的人体模型的名称')
    parser.add_argument('--scenario',type=str,default='1',help="排布场景，1 点餐 2 游戏设计 3 烹饪")
    
    parser.add_argument('--save_dir',type=str,default='D:\data\\POSA_dir\\results',help='存放排布结果的路径')
    
    parser.add_argument('--w_vis',type=float,default=1.0,help='visibility_loss的权重')
    parser.add_argument('--w_int',type=float,default=4.8,help='interaction_loss的权重')
    parser.add_argument('--w_pen',type=float,default=1.5,help='pen_iou_loss的权重')
    parser.add_argument('--w_con',type=float,default=0.0,help='connection_loss的权重')
    parser.add_argument('--w_occ',type=float,default=1.5,help='occlusion_loss的权重')
    parser.add_argument('--w_line',type=float,default=1.0,help='lin_loss的权重')
    parser.add_argument('--w_geoalign',type=float,default=1.0,help='geoalign_loss的权重')
    
    parser.add_argument('--alpha_1',type=float,default=0.97,help='ifov超参')
    parser.add_argument('--alpha_2',type=float,default=1.1,help='occ超参')
    
    parser.add_argument('--arm_len', type=float, default = 0.84,help='臂长')
    parser.add_argument('--proarm_len',type=float, default = 0.2817, help='大臂长度')
    parser.add_argument('--forearm_len',type=float, default = 0.2689, help='小臂长度')
    parser.add_argument('--palm_len',type=float, default = 0.078, help='大臂长度')
    parser.add_argument('--hand_len',type=float, default = 0.1899, help='大臂长度')
    parser.add_argument('--weight',type=float, default=73, help='体重')
    parser.add_argument('--gender',type=str, default='M',help='性别')
    
    parser.add_argument('--use_cuda', default=True, type=lambda x: x.lower() in ['true', '1'],help = '是否使用cuda')
    
    # 解析命令行参数
    args = parser.parse_args()  
    args_dict = vars(args)
    return args,args_dict

def update_vertices_obb(vertices,obbs,ind_vertices,ele_pos,ele_orient):
    '''
    update vertices and obbs for each element according to its position and orientation
    '''
    curr_vertices = torch.empty(0)
    curr_obbs = torch.empty(0)
    for i in range(len(ele_pos)):
        orig_vertices = (vertices[ind_vertices[i]:ind_vertices[i+1],:]) #(N,3)
        orig_obb = obbs[i] # (2,3)
        
        init_orient = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32) #元素初始朝向为(1,0,0)
        axis = torch.cross(init_orient, ele_orient[i]/torch.norm(ele_orient[i])) # 旋转轴
        axis_norm = axis/torch.norm(axis)
        angle = torch.acos(torch.dot(init_orient, ele_orient[i])/torch.norm(ele_orient[i])) #旋转角
        rot_mat = tgm.angle_axis_to_rotation_matrix(angle*axis_norm.reshape(-1,3))[:,:3,:3].squeeze(0) #(3,3)
        
        curr_v = torch.matmul(rot_mat,orig_vertices.t()).t() + ele_pos[i]
        curr_vertices = torch.cat([curr_vertices,curr_v])
        curr_obb = torch.matmul(rot_mat,orig_obb.t()).t() + ele_pos[i]
        curr_obbs = torch.cat([curr_obbs,curr_obb])
        
    return curr_vertices, curr_obbs.reshape(-1,2,3)

def compute_visibility_loss(ele_pos,obbs_ele,edge_t,view_point_rc,thresh =math.pi/4.0,alpha = 0.95):
    """计算可视性

    Args:
        ele_pos (_tensor(n,3)_): n个AR元素的位置
        obbs_ele (_list(Box)_) : AR元素的包围盒
        esge_t (_tensor(n)_) :AR元素包围盒的水平长度
        view_point_rc (_tensor(3,)_): 肩关节局部坐标系下的视点位置
        thresh (_float_): near peripheral的范围
        alpha (_float_): para
        
    Returns:
        _tensor(1,)_: 可视性
    """
    #visibility_loss
    # dirs = (ele_pos - view_point_rc) / torch.norm(ele_pos-view_point_rc,dim=1).unsqueeze(1)
    
    dirs = torch.mul(ele_pos-view_point_rc,torch.tensor([1.0,1.0,.0],dtype=torch.float32))
    dirs_norm = dirs / torch.norm(dirs,dim=1).unsqueeze(1)
    
    
    view_orient = torch.tensor([.0,-1.0,.0],dtype=torch.float32)
    cos_theta1 = F.cosine_similarity(dirs_norm, view_orient.view(1, -1).expand_as(dirs_norm), dim=-1)
    theta1 = torch.acos(torch.clamp(cos_theta1,-0.99,0.99))
 
    tan_theta2 = torch.div(edge_t,torch.norm(dirs,dim=1)) 
    theta2 = torch.atan(tan_theta2)    
    theta = theta1 + theta2 
    inVof_loss = torch.exp(-alpha * (thresh - theta))
    
    return inVof_loss.sum()/len(obbs_ele)

def compute_iou_loss(obbs_ele,obbs_scene,num_samples):
    '''计算AR元素的OBB与场景所有OBB的碰撞估计

    Args:
        obbs_ele (_list(OBB)_): UI Box
        boxes_scene (_list(OBB)_): 场景中所有的物体的Box构成的列表
        num_samples (_int_): 估计IoU的采样点数
        
    Returns:
        iou_loss (_tensor(1,)_) : 碰撞损失
    '''
    
    # 所有的box求和
    iou_sdf_es = torch.tensor([.0],dtype=torch.float32)
    iou_sdf_ee = torch.tensor([.0],dtype=torch.float32)
    
    for i in range(len(obbs_ele)):
        for j in range(len(obbs_scene)):
            iou_sdf_es += obb.iou_sdf_estimate(obbs_ele[i],obbs_scene[j],num_samples)
    
    for i in range(len(obbs_ele)):
        for j in range(i+1,len(obbs_ele)):
            iou_sdf_ee += obb.iou_sdf_estimate(obbs_ele[i],obbs_ele[j],num_samples)
                
    return (iou_sdf_es + 2 * iou_sdf_ee) / len(obbs_ele)

def compute_interaction_loss(ele_pos,proarm_len=0.2817,forearm_len=0.2689,palm_len=0.0862,hand_len=0.1899,weight=73.0,gender='Male',ang_step = math.pi/16):
    """计算疲劳loss

    Args:
        ele_pos (_tensor(n,3)_): n个AR元素的位置
        proarm_len (float, optional): 大臂长度. Defaults to 0.2817.
        forearm_len (float, optional): 小臂长度. Defaults to 0.2689.
        palm_len (float, optional): 手掌长度. Defaults to 0.0862.
        hand_len (float, optional):手长. Defaults to 0.1899.
        weight (float, optional): 体重. Defaults to 73.0.
        gender (str, optional): 性别. Defaults to 'Male'.
        ang_step (_type_, optional): 采样. Defaults to math.pi/16.

    Returns:
        _tensor(1,)_: _description_
    """
    interaction_loss = torch.tensor([.0],dtype=torch.float32)
    
    for i in range(ele_pos.shape[0]):
        hand_pos = ele_pos[i]
        interaction_loss += xrgonomic_metrics.compute_CE(hand_pos,proarm_len,forearm_len,palm_len,hand_len,weight,gender,ang_step)
    
    return interaction_loss / ele_pos.shape[0]

def compute_line_loss(ele_pos,alpha=1.0):
    all_z = ele_pos[:,2]
    # line_loss = torch.abs(all_z.max() - all_z.min())
    line_loss = torch.exp(alpha *torch.abs(all_z.max() - all_z.min()))-1
    return line_loss

def compute_geoalign_loss(eles_obb,target_obj_obb,alpha=10.0):
    """计算最匹配的特征的距离关系

    Args:
        eles_obb (_list(OBB)_): _description_
        target_obj_obb (_OBB_): _description_
        alpha (float, optional): _description_. Defaults to 10.0.

    Returns:
        _tensor(1,)_: _description_
    """
    geoalign_loss = torch.tensor([.0],dtype=torch.float32)
    target_obb_p = get_geo_pts(target_obj_obb)
    for i in range(len(eles_obb)):
        ele_obb_p = get_geo_pts(eles_obb[i])
        dis_mat = torch.cdist(ele_obb_p,target_obb_p)
        min_dist = torch.min(dis_mat)
        geoalign_loss += torch.exp(alpha*(min_dist-0.2)) 
    return geoalign_loss/len(eles_obb)

def compute_occlusion_loss(ele_pos,view_point_rc,threshhold = math.pi/4,alpha = 1.0):
    """计算AR元素之间的遮挡

    Args:
        ele_pos (_tensor(n,3)_): AR元素的位置
        view_point_rc (_tensor(3,)_): 肩关节坐坐标系的位置
        threshhold (_float_, optional): 发生遮挡的阈值. Defaults to math.pi/6.

    Returns:
        _tensor(1,)_: occlusion_loss
    """
    occlusion_loss = torch.tensor([.0],dtype=torch.float32)
    for i in range(ele_pos.shape[0]):
        for j in range(i+1,ele_pos.shape[0]):
            view_1 = ele_pos[i] - view_point_rc
            view_2 = ele_pos[j] - view_point_rc
            cos_theta = torch.dot(view_1,view_2) / (torch.norm(view_1) * torch.norm(view_2))
            theta = torch.acos(cos_theta)
            occlusion_loss += torch.exp(-alpha*(theta - threshhold))
    return occlusion_loss / ele_pos.shape[0]

def init_eles_cylinder(num_eles,mesh_sizes,arm_len,viewpoint_rc,offset=20.0):
    """在肩关节前方手臂距离处环形排布

    Args:
        num_eles (_int_): AR元素的数目
        arm_len (_float_): 手臂长度. Defaults to 0.8.
        viewpoint_rc (_tensor(1,3)_) :视点坐标
        offset (_float_): 相邻元素的角度. Defaults to 30.0.

    Returns:
        _tensor(num_ele,3)_: AR元素的初始位置
    """
    
    all_pos = []

    sequence = torch.cat((torch.arange(0, (num_eles + 1) // 2), -torch.arange(1, num_eles // 2 + 1))) #generate sequences like [0,1,-1,2,-2]
    sorted_indices1 = torch.argsort(torch.abs(sequence))
    sorted_seq = -sequence[sorted_indices1]
    
    sorted_indices2 = sorted(range(len(mesh_sizes)), key=lambda i: mesh_sizes[i], reverse=True)

    final_positions = [None] * len(mesh_sizes)
    for i, index in enumerate(sorted_indices2):
        final_positions[index] = sorted_seq[i]
    
    sorted_tensor = torch.tensor(final_positions,dtype=torch.float32)
    
    angles = sorted_tensor * offset

    # if num_eles % 2 == 0:
    #     angles = angles - offset/2.0
    
    viewpoint_pro = torch.mul(viewpoint_rc,torch.tensor([1.0,1.0,.0],dtype=torch.float32))
    for i in range(num_eles):
        pos_i = torch.tensor([arm_len * torch.sin(angles[i]/180.0 * math.pi),-arm_len * torch.cos(angles[i]/180.0 * math.pi),.0],dtype=torch.float32)
        pos_i = pos_i + viewpoint_pro
        all_pos.append(pos_i) 
               
    all_pos = torch.stack(all_pos).squeeze(1) 
    
    return all_pos

def init_random(num_eles,target_obj_obb):
    """在目标物体包围盒内初始化UI的位置

    Args:
        num_eles (_int_): AR元素的数目
        target_obj_aabb (_array(2,3)_): 目标物体的包围盒

    Returns:
        _tensor(n,3)_: AR元素的初始位置
    """
    scale = torch.abs(target_obj_obb._vertices[8] - target_obj_obb._vertices[1])
    point = (torch.rand(num_eles,3)-0.5) * scale + target_obj_obb._vertices[0]
    point[:,2] = 0.1
    return torch.tensor(point,dtype=torch.float32)

def get_obb_norm(target_obj_obb):
    """获取目标obb的与视线方向最接近的朝向

    Args:
        target_obj_obb (_OBB_): 目标物体的obb

    Returns:
        _tensor(3)_: 目标物体的朝向
    """
    c = target_obj_obb._vertices[0]
    p_1 = (target_obj_obb._vertices[1] + target_obj_obb._vertices[2] + target_obj_obb._vertices[3] + target_obj_obb._vertices[4])/4.0
    p_2 = (target_obj_obb._vertices[1] + target_obj_obb._vertices[2] + target_obj_obb._vertices[5] + target_obj_obb._vertices[6])/4.0
    p_3 = (target_obj_obb._vertices[5] + target_obj_obb._vertices[6] + target_obj_obb._vertices[7] + target_obj_obb._vertices[8])/4.0
    p_4 = (target_obj_obb._vertices[7] + target_obj_obb._vertices[8] + target_obj_obb._vertices[3] + target_obj_obb._vertices[4])/4.0
    
    points = torch.stack([p_1, p_2, p_3, p_4])
    dirs = points - c
    dirs_norm = dirs / torch.norm(dirs,dim=1).unsqueeze(1)
    cos_sim = F.cosine_similarity(dirs_norm, torch.tensor([.0,1.0,.0],dtype=torch.float32).view(1, -1).expand_as(dirs_norm), dim=-1)
    cos_max_index = torch.argmax(cos_sim)
    norm_dir = dirs_norm[cos_max_index]
    
    return norm_dir

def get_geo_pts(obb):
    """计算一个obb的所有边和面的中心点

    Args:
        obb (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    #计算12条边和6个面的中心点、方向向量
    edge1_mid = (obb._vertices[1] + obb._vertices[2])/2.0
    edge2_mid = (obb._vertices[2] + obb._vertices[4])/2.0
    edge3_mid = (obb._vertices[4] + obb._vertices[3])/2.0
    edge4_mid = (obb._vertices[3] + obb._vertices[1])/2.0
    edge5_mid = (obb._vertices[1] + obb._vertices[5])/2.0
    edge6_mid = (obb._vertices[5] + obb._vertices[6])/2.0
    edge7_mid = (obb._vertices[6] + obb._vertices[2])/2.0    
    edge8_mid = (obb._vertices[5] + obb._vertices[7])/2.0
    edge9_mid = (obb._vertices[7] + obb._vertices[8])/2.0
    edge10_mid = (obb._vertices[8] + obb._vertices[6])/2.0
    edge11_mid = (obb._vertices[8] + obb._vertices[4])/2.0
    edge12_mid = (obb._vertices[7] + obb._vertices[3])/2.0
    
    
    face1_mid = (obb._vertices[1] + obb._vertices[2] + obb._vertices[3] + obb._vertices[4])/4.0 
    face2_mid = (obb._vertices[1] + obb._vertices[2] + obb._vertices[5] + obb._vertices[6])/4.0 
    face3_mid = (obb._vertices[5] + obb._vertices[6] + obb._vertices[7] + obb._vertices[8])/4.0 
    face4_mid = (obb._vertices[7] + obb._vertices[8] + obb._vertices[3] + obb._vertices[4])/4.0     
    face5_mid = (obb._vertices[2] + obb._vertices[4] + obb._vertices[6] + obb._vertices[8])/4.0     
    face6_mid = (obb._vertices[1] + obb._vertices[5] + obb._vertices[3] + obb._vertices[7])/4.0 

    all_pts = torch.stack([edge1_mid,edge2_mid,edge3_mid,edge4_mid,edge5_mid,edge6_mid,
                          edge7_mid,edge8_mid,edge9_mid,edge10_mid,edge11_mid,edge12_mid,
                          face1_mid,face2_mid,face3_mid,face4_mid,face5_mid,face6_mid])
    
    # dir_e1 = (obb._vertices[1] - obb._vertices[2])/torch.norm(obb._vertices[1] - obb._vertices[2])
    # dir_e2 = (obb._vertices[2] - obb._vertices[4])/torch.norm(obb._vertices[2] - obb._vertices[4])
    # dir_e3 = (obb._vertices[4] - obb._vertices[3])/torch.norm(obb._vertices[4] - obb._vertices[3])
    # dir_e4 = (obb._vertices[3] - obb._vertices[1])/torch.norm(obb._vertices[3] - obb._vertices[1])
    # dir_e5 = (obb._vertices[1] - obb._vertices[5])/torch.norm(obb._vertices[1] - obb._vertices[5])
    # dir_e6 = (obb._vertices[5] - obb._vertices[6])/torch.norm(obb._vertices[5] - obb._vertices[6])
    # dir_e7 = (obb._vertices[6] - obb._vertices[2])/torch.norm(obb._vertices[6] - obb._vertices[2])
    # dir_e8 = (obb._vertices[5] - obb._vertices[7])/torch.norm(obb._vertices[5] - obb._vertices[7])
    # dir_e9 = (obb._vertices[7] - obb._vertices[8])/torch.norm(obb._vertices[7] - obb._vertices[8])
    # dir_e10 = (obb._vertices[8] - obb._vertices[6])/torch.norm(obb._vertices[8] - obb._vertices[6])
    # dir_e11 = (obb._vertices[8] - obb._vertices[4])/torch.norm(obb._vertices[8] - obb._vertices[4])
    # dir_e12 = (obb._vertices[7] - obb._vertices[3])/torch.norm(obb._vertices[7] - obb._vertices[3])

    
    # dir_f1 = torch.cross(-dir_e1,dir_e4)
    # dir_f2 = torch.cross(-dir_e7,-dir_e6)
    # dir_f3 = torch.cross(-dir_e10,dir_e9)
    # dir_f4 = torch.cross(dir_e11,-dir_e3)
    # dir_f5 = torch.cross(dir_e7,-dir_e2)
    # dir_f6 = torch.cross(dir_e8,-dir_e12)
    
    # all_dir = torch.stack([dir_e1,dir_e2,dir_e3,dir_e4,dir_e5,dir_e6,dir_e7,dir_e8,dir_e9,dir_e10,dir_e11,dir_e12,
    #                        dir_f1,dir_f2,dir_f3,dir_f4,dir_f5,dir_f6])
     
    # return all_pts,all_dir
    return all_pts

def culling_obbs_scene(obbs_ele,obbs_scene,num_samples,eps):
    '''挑选场景中与obb_ui相穿插的obb

    Args:
        obbs_ele (_list(OBB)_): AR元素OBBs
        obbs_scene (_list(OBB)_): 环境OBBs
        num_samples (_int_): 估计iou的采样点数
        eps (_float_) :表示筛选的精度

    Returns:
        _list(OBB)_: 与ui_obb穿插的环境OBB列表
        _list(int)_: 碰撞的OBB的标号

    '''
    obbs_intersection = []
    ins = []
    
    for i in range(len(obbs_scene)):
        for j in range(len(obbs_ele)):
            if obb.iou_sdf_estimate(obbs_scene[i],obbs_ele[j],num_samples) > eps :
                obbs_intersection.append(obbs_scene[i])
                ins.append(i)
                break
    return obbs_intersection,ins
  
def save_results_geoalign(save_dir,scenario,scene_path,body_path,ele_paths,ele_pos,target_obj_orient):
    
    scene_name = osp.splitext(osp.basename(scene_path))[0]
    body_name =  osp.splitext(osp.basename(body_path))[0]
    save_path = osp.join(save_dir,scenario,scene_name + '_'+body_name)
    os.makedirs(save_path,exist_ok=True)
    
    ele_paras = []
    for i in range(len(ele_paths)):   
        x_prime = target_obj_orient
        z_prime = torch.tensor([.0,.0,1.0],dtype=torch.float32)
        y_prime = torch.cross(z_prime,x_prime)
        rot_mat = torch.stack([x_prime, y_prime, z_prime], dim=0).t()
        
        ele_mesh = o3d.io.read_triangle_mesh(ele_paths[i])
        ele_mesh.rotate(rot_mat.detach().numpy(),center=(0, 0, 0))
        ele_mesh_center = ele_mesh.get_center()
        ele_mesh_t = ele_pos[i].squeeze(0).detach().numpy().reshape(3,) - ele_mesh_center
        ele_mesh.translate(ele_mesh_t)
        
        ele_name = osp.splitext(osp.basename(ele_paths[i]))[0]
        ele_mat = np.eye(4)
        ele_mat[:3,:3] = rot_mat.detach().numpy()
        ele_mat[:3,3] = ele_mesh_t
        ele_para = {"name":ele_name,
                    "mat":ele_mat.tolist()}
        ele_paras.append(ele_para) 
        
    virtual_ele_json = osp.join(save_path,'virtual_eles_baseline.json') 
    with open(virtual_ele_json,'w')as vf:
        json.dump(ele_paras,vf)  

def save_results_base(save_dir,scenario,scene_path,body_path,ele_paths,ele_pos,target_obj_orient):
    
    scene_name = osp.splitext(osp.basename(scene_path))[0]
    body_name =  osp.splitext(osp.basename(body_path))[0]
    save_path = osp.join(save_dir,scenario,scene_name + '_'+body_name)
    os.makedirs(save_path,exist_ok=True)
    
    ele_paras = []
    for i in range(len(ele_paths)):   
        x_prime = target_obj_orient
        z_prime = torch.tensor([.0,.0,1.0],dtype=torch.float32)
        y_prime = torch.cross(z_prime,x_prime)
        rot_mat = torch.stack([x_prime, y_prime, z_prime], dim=0).t()
        
        ele_mesh = o3d.io.read_triangle_mesh(ele_paths[i])
        ele_mesh.rotate(rot_mat.detach().numpy(),center=(0, 0, 0))
        ele_mesh_center = ele_mesh.get_center()
        ele_mesh_t = ele_pos[i].squeeze(0).detach().numpy().reshape(3,) - ele_mesh_center
        ele_mesh.translate(ele_mesh_t)
        
        ele_name = osp.splitext(osp.basename(ele_paths[i]))[0]
        ele_mat = np.eye(4)
        ele_mat[:3,:3] = rot_mat.detach().numpy()
        ele_mat[:3,3] = ele_mesh_t
        ele_para = {"name":ele_name,
                    "mat":ele_mat.tolist()}
        ele_paras.append(ele_para) 
        
    virtual_ele_json = osp.join(save_path,'virtual_eles_base.json') 
    with open(virtual_ele_json,'w')as vf:
        json.dump(ele_paras,vf)  
        
def opt_orient_rsc():
    '''
    在肩关节坐标系下进行全局的优化
    '''
    # 解析命令行参数
    args,_ = my_cmdargs()  
    t_start = time.time()
    # load scene_mesh
    scene_name = args.scene_name
    if scene_name in self_define_scenes:
        scene_path = osp.join("D:/data/ICAR_SE/",scene_name,"env.obj")
    else:
        scene_path = osp.join(args.scenemesh_folder, scene_name +'.ply')
    scene_mesh = o3d.io.read_triangle_mesh(scene_path)
    
    # load body_mesh
    body_path = osp.join(args.affordance_folder, 'Layout', args.scene_name ,'meshes', args.body_name + ".obj")
    body_mesh = o3d.io.read_triangle_mesh(body_path)
    
    # load UI mesh
    scenario = 'scenario_' + args.scenario
    ele_base = osp.join(args.elements_folder,scenario)
    ele_names = []
    ele_paths = []
    ele_meshes = []
    ele_sizes = []
    # pdb.set_trace()
    for ele in os.listdir(ele_base):
        ele_dir = osp.join(ele_base,ele,ele+'.obj')
        ele_mesh = o3d.io.read_triangle_mesh(ele_dir)
        ele_names.append(ele)
        ele_paths.append(ele_dir)
        ele_meshes.append(ele_mesh)
        aabb_i = ele_mesh.get_axis_aligned_bounding_box()
        ele_aabb = np.array([aabb_i.get_min_bound(),aabb_i.get_max_bound()])
        ele_scale = ele_aabb[1]-ele_aabb[0]
        ele_sizes.append(np.linalg.norm(np.dot(np.array([1.0,1.0,.0]),ele_scale)))
    
    num_eles = len(ele_names)
    print("{} elements are optimizing!".format(num_eles))
    
    # load body_pkl  
    pkl_file_path = osp.join(args.affordance_folder,'Layout',args.scene_name,'pkls',args.body_name + ".pkl")
    with open(pkl_file_path,'rb') as f:
        body_info = pickle.load(f)
        
    curr_orient_norm,view_point,_,r_shoulder,pelvis = viz_body_info(body_info,scene_mesh,viz = False)
    
    view_point = torch.tensor(view_point,dtype=torch.float32)
    r_shoulder = torch.tensor(r_shoulder,dtype=torch.float32)
    pelvis = torch.tensor(pelvis,dtype=torch.float32)
    curr_orient_norm = torch.tensor(curr_orient_norm,dtype=torch.float32)
    arm_len = args.proarm_len + args.forearm_len + args.hand_len
      
    #计算坐标系变换
    axis = np.cross(np.array([.0, -1.0, .0]), curr_orient_norm) # 旋转轴
    axis_norm = axis/np.linalg.norm(axis)
    angle = np.arccos(np.dot(np.array([.0, -1.0, 0.0]), curr_orient_norm)) #旋转角
    rot_w2rc = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_norm * angle)
    rot_w2rc = torch.tensor(rot_w2rc,dtype=torch.float32)
    
    view_point_t = view_point - r_shoulder
    view_point_rc = torch.matmul(rot_w2rc.t(),view_point_t.reshape(3,1)).reshape(1,3) 
    
    # 构建场景中所有包围盒的OBB以及所有UI的OBB(肩关节局部坐标系下)
    # pdb.set_trace()
    aabbs_scene = load_scene_aabbs(args.scene_name,args.scene_aabbs_folder)
    obbs_scene = create_scene_obbs(aabbs_scene,rot_w2rc.detach(),(r_shoulder.reshape(3,)).detach())
    init_pos = init_eles_cylinder(num_eles,ele_sizes,arm_len,view_point_rc.reshape(3,)) #肩关节坐标系下的初始位置
    obbs_ele = create_ele_obbs(ele_meshes,view_point_rc.detach().reshape(3,),init_pos.detach())


    #包围盒在水平位置的长度
    edge = []
    for i in range(len(obbs_ele)):
        vertices_i = obbs_ele[i]._vertices
        edge_i = torch.norm(vertices_i[1] - vertices_i[3])
        edge.append(edge_i/2.0)
    edge_t = torch.tensor(edge)
    
    # 得到目标交互物体的中心点位置
    # pdb.set_trace()
    body_name = args.body_name
    parts = body_name.split("_")
    target_object_id = parts[-1]
    target_object = obbs_scene[int(target_object_id)]
    target_object_c = torch.tensor(target_object._vertices,dtype=torch.float32).mean(dim=0)
    
    
    # pdb.set_trace()
    #挑选发生穿插的OBBs
    obbs_intersection,labels_intersection = culling_obbs_scene(obbs_ele,obbs_scene,args.num_samples,1e-4)
    #label_remove
    scenes_label_remove = [{"scene_name":"BasementSittingBooth","label":7},
                           {"scene_name":"N3office","label":0},
                           {"scene_name":"N3Library","label":2},
                           {"scene_name":"MPH8","label":0},
                           {"scene_name":"MPH16","label":0}]
    for scene_remo in scenes_label_remove:
        if scene_remo['scene_name'] == args.scene_name :
            label_to_remove = scene_remo["label"]
            obbs_intersection = [obb for obb, label in zip(obbs_intersection, labels_intersection) if label != label_to_remove]             
    
    # viz_obbs(scene_mesh,obbs_intersection,obbs_ele,r_shoulder.detach().reshape(3,),rot_w2rc.detach())
    # pdb.set_trace()
    
    # 初始化
    init_pos_rc = init_eles_cylinder(num_eles,ele_sizes,0.7,view_point_rc)
    ele_pos = init_pos_rc.clone().detach() 
    ele_pos.requires_grad=True
    opt_param = [ele_pos]
    
    # pdb.set_trace()
    # scene_mesh = o3d.io.read_triangle_mesh(osp.join(args.scenemesh_folder, args.scene_name +'.ply')) 
    # viz_results(scene_mesh,body_mesh,ele_meshes,r_shoulder.detach().reshape(3,),rot_w2rc.detach(),ele_pos,view_point_rc,True)  
    # 构建优化器
    optimizer = Optim.Adam(opt_param, lr=0.05)
    lr_schedule = Optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    
    # 开始优化
    num_iter = 60  
    logdir = "./gen_layout/mylayout/without_orient_rsc/loss_log" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(logdir)
    
    w_v = args.w_vis
    w_i = args.w_int
    w_p = args.w_pen
    w_o = args.w_occ    
    w_lin = args.w_line
    
    for epoch in tqdm(range(num_iter)):
        optimizer.zero_grad()
        
        # pdb.set_trace()
        # 更新box_ui的translation 和 rotation 和 vertices
        for i in range(num_eles):
    
            obbs_ele[i]._translation = ele_pos[i]
            x_prime = view_point_rc.reshape(3,) - ele_pos[i].reshape(3,)
            x_prime = torch.mul(x_prime,torch.tensor([1.0,1.0,.0],dtype=torch.float32))
            x_prime = torch.div(x_prime,torch.norm(x_prime))
            z_prime = torch.tensor([.0,.0,1.0],dtype=torch.float32)
            y_prime = torch.cross(z_prime,x_prime)
            obbs_ele[i]._rotation = torch.stack((x_prime,y_prime,z_prime),dim=0).t()
            obbs_ele[i]._vertices = torch.matmul(obbs_ele[i]._rotation,(torch.tensor(obb.UNIT_BOX,dtype=torch.float32)*obbs_ele[i]._scale).t()).t() + obbs_ele[i]._translation
        
        # pdb.set_trace()
        pen_iou_loss = compute_iou_loss(obbs_ele,obbs_intersection,args.num_samples)
        
        visibility_loss = compute_visibility_loss(ele_pos,obbs_ele,edge_t,view_point_rc.reshape(3,))
        
        interaction_loss = compute_interaction_loss(ele_pos,args.proarm_len,args.forearm_len,args.palm_len,args.hand_len,args.weight,args.gender)
        
        # connection_loss = compute_connection_loss(ele_pos,target_object_c)
        line_loss = compute_line_loss(ele_pos)
        
        occlusion_loss = compute_occlusion_loss(ele_pos,view_point_rc.reshape(3,))
        
        # loss_total = w_v * visibility_loss + w_i * interaction_loss + w_p * pen_iou_loss + w_c * connection_loss + w_o * occlusion_loss
        # loss_total = w_v * visibility_loss + w_i * interaction_loss + w_p * pen_iou_loss + w_o * occlusion_loss
        loss_total = w_v * visibility_loss + w_i * interaction_loss + w_p * pen_iou_loss + w_o * occlusion_loss + w_lin * line_loss
        loss_total.backward()
        lr_schedule.step()
        optimizer.step()
        
        writer.add_scalar("visibility_loss",w_v * visibility_loss,epoch)
        writer.add_scalar("interaction_loss", w_i * interaction_loss,epoch)
        writer.add_scalar("pen_loss", w_p * pen_iou_loss ,epoch)
        writer.add_scalar("occlusion_loss",w_o * occlusion_loss,epoch)
        writer.add_scalar("total_loss",loss_total,epoch)
        
        # print(ele_pos.grad)
        
    writer.flush() 
    writer.close()
    t_end = time.time()
    cost_time = t_end-t_start
    print(f"compute time: {cost_time:.4f} seconds")  
    print(f"best total loss : {loss_total.item():.4f}")   
    
    # viz_obbs(scene_mesh,obbs_intersection,obbs_ele,r_shoulder.detach().reshape(3,),rot_w2rc.detach())
    ####### 可视化(肩关节坐标系)并保存结果########
    # if scene_name in self_define_scenes:
    #     scene_mesh = o3d.io.read_triangle_mesh(osp.join("D:/data/ICAR_SE/",scene_name,"env.obj"))
    # else:
    #     scene_mesh = o3d.io.read_triangle_mesh(osp.join(args.scenemesh_folder, args.scene_name +'.ply')) 
    # viz_results(scene_mesh,body_mesh,ele_meshes,r_shoulder.detach().reshape(3,),rot_w2rc.detach(),ele_pos,view_point_rc,True)
    # pdb.set_trace()
    # save_dir =  args.save_dir
    # save_results(save_dir,scenario,scene_path,body_path,ele_paths,r_shoulder,rot_w2rc.detach(),ele_pos,view_point_rc)
    ############结束可视化以及保存##################

def opt_baseline_orient_rsc(): 
    '''
    在肩关节坐标系下进行全局优化
    '''
    # 解析命令行参数
    args,_ = my_cmdargs()  
     
    # load scene_mesh
    scene_name = args.scene_name
    if scene_name in self_define_scenes:
        scene_path = osp.join("D:/data/ICAR_SE/",scene_name,"env.obj")
    else:
        scene_path = osp.join(args.scenemesh_folder, scene_name +'.ply')
    scene_mesh = o3d.io.read_triangle_mesh(scene_path)
    
    # load body_mesh
    body_path = osp.join(args.affordance_folder, 'Layout', args.scene_name ,'meshes', args.body_name + ".obj")
    body_mesh = o3d.io.read_triangle_mesh(body_path)
    
    # load UI mesh
    scenario = 'scenario_' + args.scenario
    ele_base = osp.join(args.elements_folder,scenario)
    ele_names = []
    ele_paths = []
    ele_meshes = []
    # pdb.set_trace()
    for ele in os.listdir(ele_base):
        ele_dir = osp.join(ele_base,ele,ele+'.obj')
        ele_mesh = o3d.io.read_triangle_mesh(ele_dir)
        ele_names.append(ele)
        ele_paths.append(ele_dir)
        ele_meshes.append(ele_mesh)
    
    num_eles = len(ele_names)
    print("{} elements are optimizing!".format(num_eles))
    
    # load body_pkl  
    pkl_file_path = osp.join(args.affordance_folder,'Layout',args.scene_name,'pkls',args.body_name + ".pkl")
    with open(pkl_file_path,'rb') as f:
        body_info = pickle.load(f)
        
    curr_orient_norm,view_point,_,r_shoulder,pelvis = viz_body_info(body_info,scene_mesh,viz = False)
    
    view_point = torch.tensor(view_point,dtype=torch.float32)
    r_shoulder = torch.tensor(r_shoulder,dtype=torch.float32)
    pelvis = torch.tensor(pelvis,dtype=torch.float32)
    curr_orient_norm = torch.tensor(curr_orient_norm,dtype=torch.float32)
      
    #计算坐标系变换
    axis = np.cross(np.array([.0, -1.0, .0]), curr_orient_norm) # 旋转轴
    axis_norm = axis/np.linalg.norm(axis)
    angle = np.arccos(np.dot(np.array([.0, -1.0, 0.0]), curr_orient_norm)) #旋转角
    rot_w2rc = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_norm * angle)
    rot_w2rc = torch.tensor(rot_w2rc,dtype=torch.float32)
    
    view_point_t = view_point - r_shoulder
    view_point_rc = torch.matmul(rot_w2rc.t(),view_point_t.reshape(3,1)).reshape(1,3) 
    
    # 构建场景中所有包围盒的OBB
    aabbs_scene = load_scene_aabbs(args.scene_name,args.scene_aabbs_folder)
    obbs_scene = create_scene_obbs(aabbs_scene,rot_w2rc.detach(),(r_shoulder.reshape(3,)).detach())
    
    # 得到目标交互物体的中心点位置
    body_name = args.body_name
    parts = body_name.split("_")
    target_object_id = parts[-1]
    target_obj_obb = obbs_scene[int(target_object_id)]   
    
    # pdb.set_trace()
    # viz_obb(aabbs_scene[int(target_object_id)],scene_mesh)
    # # 初始化ele及其obb
    # pdb.set_trace()
    init_pos = init_random(num_eles,target_obj_obb)
    obbs_ele = create_ele_obbs(ele_meshes,view_point_rc.detach().reshape(3,),init_pos.detach())
    
    #计算场景OBB的朝向
    # pdb.set_trace()
    target_obj_norm = get_obb_norm(target_obj_obb)
    # viz_results_baseline(scene_mesh,body_mesh,ele_meshes,r_shoulder.detach().reshape(3,),rot_w2rc.detach(),init_pos,view_point_rc,target_obj_norm,True)  
    # pdb.set_trace()
    #挑选发生穿插的OBBs
    obbs_intersection,labels_intersection = culling_obbs_scene(obbs_ele,obbs_scene,args.num_samples,1e-4)
    #label_remove
    scenes_label_remove = [{"scene_name":"BasementSittingBooth","label":7},
                           {"scene_name":"N3office","label":0},
                           {"scene_name":"N3Library","label":2},
                           {"scene_name":"MPH8","label":0},
                           {"scene_name":"MPH16","label":0}]
    for scene_remo in scenes_label_remove:
        if scene_remo['scene_name'] == args.scene_name :
            label_to_remove = scene_remo["label"]
            obbs_intersection = [obb for obb, label in zip(obbs_intersection, labels_intersection) if label != label_to_remove]             
    
    # 初始化
    ele_pos = init_pos.clone().detach() 
    ele_pos.requires_grad=True
    opt_param = [ele_pos]
    
    # scene_mesh = o3d.io.read_triangle_mesh(osp.join(args.scenemesh_folder, args.scene_name +'.ply')) 
    # viz_results_baseline(scene_mesh,body_mesh,ele_meshes,r_shoulder.detach().reshape(3,),rot_w2rc.detach(),ele_pos,view_point_rc,target_obj_norm,True)  
    # 构建优化器
    optimizer = Optim.Adam(opt_param, lr=0.05)
    lr_schedule = Optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    
    # 开始优化
    num_iter = 60  
    logdir = "./gen_layout/mylayout/without_orient_rsc/loss_log" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(logdir)
    
    w_p = args.w_pen  
    w_lin = args.w_line
    w_geoalign = args.w_geoalign
    
    for epoch in tqdm(range(num_iter)):
        optimizer.zero_grad()
        
        # pdb.set_trace()
        # 更新box_ui的translation 和 rotation 和 vertices
        for i in range(num_eles):
    
            obbs_ele[i]._translation = ele_pos[i]
            x_prime = target_obj_norm
            z_prime = torch.tensor([.0,.0,1.0],dtype=torch.float32)
            y_prime = torch.cross(z_prime,x_prime)
            obbs_ele[i]._rotation = torch.stack((x_prime,y_prime,z_prime),dim=0).t()
            obbs_ele[i]._vertices = torch.matmul(obbs_ele[i]._rotation,(torch.tensor(obb.UNIT_BOX,dtype=torch.float32)*obbs_ele[i]._scale).t()).t() + obbs_ele[i]._translation
        
        # pdb.set_trace()
        pen_iou_loss = compute_iou_loss(obbs_ele,obbs_intersection,args.num_samples)
        
        line_loss = compute_line_loss(ele_pos)
        # pdb.set_trace()
        geoalign_loss = compute_geoalign_loss(obbs_ele,target_obj_obb)
        
        # loss_total = w_v * visibility_loss + w_i * interaction_loss + w_p * pen_iou_loss + w_c * connection_loss + w_o * occlusion_loss
        # loss_total = w_v * visibility_loss + w_i * interaction_loss + w_p * pen_iou_loss + w_o * occlusion_loss
        loss_total = w_p * pen_iou_loss + w_geoalign * geoalign_loss + w_lin * line_loss
        # loss_total = w_p * pen_iou_loss + w_lin * line_loss
        loss_total.backward()
        lr_schedule.step()
        optimizer.step()
        
        # writer.add_scalar("visibility_loss",w_v * visibility_loss,epoch)
        # writer.add_scalar("interaction_loss", w_i * interaction_loss,epoch)
        writer.add_scalar("pen_loss", w_p * pen_iou_loss ,epoch)
        writer.add_scalar("line_loss",w_lin * line_loss,epoch)
        writer.add_scalar("geoalign_loss",w_geoalign * geoalign_loss,epoch)
        writer.add_scalar("total_loss",loss_total,epoch)
        
        # print(ele_pos.grad)
        
    writer.flush() 
    writer.close()
    
    # viz_obbs(scene_mesh,obbs_intersection,obbs_ele,r_shoulder.detach().reshape(3,),rot_w2rc.detach())
    # 可视化(肩关节坐标系)
    if scene_name in self_define_scenes:
        scene_mesh = o3d.io.read_triangle_mesh(osp.join("D:/data/ICAR_SE/",scene_name,"env.obj"))
    else:
        scene_mesh = o3d.io.read_triangle_mesh(osp.join(args.scenemesh_folder, args.scene_name +'.ply'))
    
    viz_results_baseline(scene_mesh,body_mesh,ele_meshes,r_shoulder.detach().reshape(3,),rot_w2rc.detach(),ele_pos,view_point_rc,target_obj_norm,True)
    pdb.set_trace()
    save_dir =  args.save_dir
    save_results_geoalign(save_dir,scenario,scene_path,body_path,ele_paths,ele_pos,target_obj_norm)
    
if __name__ == '__main__':

    # opt_orient_rsc()
    # opt_baseline_orient_rsc()
   
    
    
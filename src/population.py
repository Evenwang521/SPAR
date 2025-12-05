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

import os
import os.path as osp
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm
import random
import trimesh
import glob
import yaml
import pickle
import torchgeometry as tgm
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
import json
import pdb
from torch.utils.tensorboard import SummaryWriter
import optimizers.optim_factory as optim_factory
from gen_human import cmd_parser 
from gen_human import posa_utils,eulerangles,viz_utils,misc_utils,data_utils,opt_utils
from scipy import stats

import time
self_define_scenes =  {'S1_E1', 'S1_E2', 'S2_E1', 'S2_E2'}
if __name__ == '__main__':
    args, args_dict = cmd_parser.parse_config()
    args_dict['batch_size'] = 1
    args_dict['semantics_w'] = 0.01
    batch_size = args_dict['batch_size']
    if args.use_clothed_mesh:
        args_dict['opt_pose'] = False
    args_dict['base_dir'] = osp.expandvars(args_dict.get('base_dir'))
    args_dict['ds_us_dir'] = osp.expandvars(args_dict.get('ds_us_dir'))
    args_dict['affordance_dir'] = osp.expandvars(args_dict.get('affordance_dir'))
    args_dict['pkl_file_path'] = osp.expandvars(args_dict.get('pkl_file_path'))
    args_dict['model_folder'] = osp.expandvars(args_dict.get('model_folder'))
    args_dict['rp_base_dir'] = osp.expandvars(args_dict.get('rp_base_dir'))
    base_dir = args_dict.get('base_dir')
    ds_us_dir = args_dict.get('ds_us_dir')
    scene_name = args_dict.get('scene_name')
    targetobject = args.target_object
    registration_file = args.scene_registration_file
    args_dict['targer_object_pos'] = None

    # Create results folders
    affordance_dir = osp.join(args_dict.get('affordance_dir'))
    os.makedirs(affordance_dir, exist_ok=True)
    pkl_folder = osp.join(affordance_dir, 'pkl', args_dict.get('scene_name'))
    os.makedirs(pkl_folder, exist_ok=True)
    physical_metric_folder = osp.join(affordance_dir, 'physical_metric', args_dict.get('scene_name'))
    os.makedirs(physical_metric_folder, exist_ok=True)
    rendering_folder = osp.join(affordance_dir, 'renderings', args_dict.get('scene_name'))
    os.makedirs(rendering_folder, exist_ok=True)
    os.makedirs(osp.join(affordance_dir, 'meshes', args_dict.get('scene_name')), exist_ok=True)
    os.makedirs(osp.join(affordance_dir, 'meshes_clothed', args_dict.get('scene_name')), exist_ok=True)
    
    os.makedirs(osp.join(affordance_dir,"Layout",args_dict.get('scene_name'),"meshes"), exist_ok=True)
    os.makedirs(osp.join(affordance_dir,"Layout",args_dict.get('scene_name'),"pkls"),exist_ok=True)
    

    device = torch.device("cuda" if args_dict.get('use_cuda') else "cpu")
    dtype = torch.float32
    args_dict['device'] = device
    args_dict['dtype'] = dtype
    
    t_start = time.time()

    A_1, U_1, D_1 = posa_utils.get_graph_params(args_dict.get('ds_us_dir'), 1, args_dict['use_cuda'])
    down_sample_fn = posa_utils.ds_us(D_1).to(device)
    up_sample_fn = posa_utils.ds_us(U_1).to(device)

    A_2, U_2, D_2 = posa_utils.get_graph_params(args_dict.get('ds_us_dir'), 2, args_dict['use_cuda'])
    down_sample_fn2 = posa_utils.ds_us(D_2).to(device)
    up_sample_fn2 = posa_utils.ds_us(U_2).to(device)

    faces_arr = trimesh.load(osp.join(ds_us_dir, 'mesh_{}.obj'.format(0)), process=False).faces
    model = misc_utils.load_model_checkpoint(**args_dict).to(device)

    # load 3D scene
    # pdb.set_trace()
    scene = vis_o3d = None
    if scene_name in self_define_scenes:
        if args.viz or args.show_init_pos:
            scene = o3d.io.read_triangle_mesh(osp.join("D:\data\ICAR_SE", scene_name,  'env.obj'))
        scene_data = data_utils.load_scene_data(name=scene_name, sdf_dir=osp.join("D:\data\ICAR_SE", 'sdf'),
                                            **args_dict)
    else:
        if args.viz or args.show_init_pos:
            scene = o3d.io.read_triangle_mesh(osp.join(base_dir, 'scenes', args_dict.get('scene_name') + '.ply'))
        scene_data = data_utils.load_scene_data(name=scene_name, sdf_dir=osp.join(base_dir, 'sdf'),
                                            **args_dict)
    pkl_file_path = args_dict.pop('pkl_file_path')  
    if osp.isdir(pkl_file_path):
        pkl_file_dir = pkl_file_path
        pkl_file_paths = glob.glob(osp.join(pkl_file_dir, '*.pkl'))
        random.shuffle(pkl_file_paths) 
    else:
        pkl_file_paths = [pkl_file_path]
    #为pkl_file_path下的所有pkl文件(人体模型)生成可能的放置结果，本质上也是一个个放置并生成，但这里并没有处理人体之间的碰撞
    for pkl_file_path in pkl_file_paths: 
        pkl_file_basename = osp.splitext(osp.basename(pkl_file_path))[0]
        vertices_clothed = None
        if args.use_clothed_mesh:
            clothed_mesh = trimesh.load(osp.join(args.rp_base_dir, pkl_file_basename + '.obj'), process=False)
            vertices_clothed = clothed_mesh.vertices

        print('file_name: {}'.format(pkl_file_path))
        # load pkl file
        # vertices_org是已经经过旋转的，对齐到场景坐标系下的mesh顶点,面向y轴负方向
        vertices_org, vertices_can, faces_arr, body_model, R_can, pelvis, torch_param, vertices_clothed = data_utils.pkl_to_canonical(
            pkl_file_path, vertices_clothed=vertices_clothed, **args_dict)
        
        #限制骨盆高度不低于0.5m
        #原理是关节点坐标以骨盆为原点，距离骨盆最远的的负值点就是脚底板，对该负值取反得到了骨盆距离地面的高度
        pelvis_z_offset = - vertices_org.detach().cpu().numpy().squeeze()[:, 2].min() 
        pelvis_z_offset = pelvis_z_offset.clip(min=0.5)
        init_body_pose = body_model.body_pose.detach().clone()
        # DownSample
        vertices_org_ds = down_sample_fn.forward(vertices_org.unsqueeze(0).permute(0, 2, 1))
        vertices_org_ds = down_sample_fn2.forward(vertices_org_ds).permute(0, 2, 1).squeeze()
        vertices_can_ds = down_sample_fn.forward(vertices_can.unsqueeze(0).permute(0, 2, 1))
        vertices_can_ds = down_sample_fn2.forward(vertices_can_ds).permute(0, 2, 1).squeeze()
        
        
        if args.use_semantics:
            scene_semantics = scene_data['scene_semantics']
            scene_obj_ids = np.unique(scene_semantics.nonzero().detach().cpu().numpy().squeeze()).tolist() #获取场景中所有包含物体类别的id
            n = 50
            print('Generating feature map')
            selected_batch = None
            z = torch.tensor(np.random.normal(0, 1, (n, args.z_dim)).astype(np.float32)).to(device)
            gen_batches = model.decoder(z, vertices_can_ds.unsqueeze(0).expand(n, -1, -1)).detach()
            #对于每一个等待放入场景的人体模型，生成n次触点信息
            #统计每一次生成的触点信息中中出现的物体类型，得到最常出现的物体品类
            #如果目标场景中存在该物体，则将该次生成的的触点模型作为要放入场景的模型
            #否则，警告，场景中不存在最优的匹配位置，结果可能是次优的，直接将第一个生成的触点模型作为放入场景的模型
            for i in range(gen_batches.shape[0]):
                x, x_semantics = data_utils.batch2features(gen_batches[i], **args_dict)
                x_semantics = np.argmax(x_semantics, axis=-1)
                modes = stats.mode(x_semantics[x_semantics != 0])
                # pdb.set_trace()
                most_common_obj_id = modes.mode[0]
                if most_common_obj_id not in scene_obj_ids:
                    continue
                selected_batch = i
                break
            if selected_batch is not None:
                gen_batches = gen_batches[i].unsqueeze(0)
            else:
                print('No good semantic feat found - Results might be suboptimal')
                gen_batches = gen_batches[0].unsqueeze(0)
        else:
            z = torch.tensor(np.random.normal(0, 1, (args.num_rendered_samples, args.z_dim)).astype(np.float32)).to(
                device)
            gen_batches = model.decoder(z,
                                        vertices_can_ds.unsqueeze(0).expand(args.num_rendered_samples, -1, -1)).detach()
        #对于每一个生成的合理的触点模型 (gen_batches存放生成的所有的人体模型)
        for sample_id in range(gen_batches.shape[0]):
            
            result_filename = pkl_file_basename + '_{:02d}'.format(sample_id)
            gen_batch = gen_batches[sample_id, :, :].unsqueeze(0)

            if args.show_gen_sample:
                gen = gen_batch.clone()
                gen_batch_us = up_sample_fn2.forward(gen.transpose(1, 2))
                gen_batch_us = up_sample_fn.forward(gen_batch_us).transpose(1, 2)
                if args.viz:
                    gen = viz_utils.show_sample(vertices_org, gen_batch_us, faces_arr, **args_dict)
                    o3d.visualization.draw_geometries(gen)
                if args.render:
                    gen_sample_img = viz_utils.render_sample(gen_batch_us, vertices_org, faces_arr, **args_dict)[0]
                    gen_sample_img.save(osp.join(affordance_dir, 'renderings', args_dict.get('scene_name'),
                                                 pkl_file_basename + '_gen.png'))

            # Create init points grid
            #增加一个修改初始散点位置的函数即可，只需要修改bbox参数 知道目标物体的包围盒参数即可
            if targetobject == None: #如果没有添加目标物体
                init_pos = torch.tensor(
                    misc_utils.create_init_points(scene_data['bbox'].detach().cpu().numpy(), args.affordance_step,
                                              pelvis_z_offset), dtype=dtype, device=device).reshape(-1, 1, 3)
            else:#如果添加目标物体
                #instance_aabb folder
                aabb_folder = os.path.join("D:\code\python\\POSA\output\scenes_ins_aabb",scene_name+".json")
                with open(aabb_folder,'r') as abf:
                    instance_aabb = json.load(abf)
                if scene_name not in self_define_scenes:
                    instance_aabb = json.loads(instance_aabb)
                
                target_instance = instance_aabb[targetobject]
                min_bound = target_instance['min_bound']
                max_bound = target_instance['max_bound']
                init_pos = torch.tensor(
                    misc_utils.create_init_points_targetobject(min_bound,max_bound, args.affordance_step,
                                              pelvis_z_offset), dtype=dtype, device=device).reshape(-1, 1, 3)
                #计算目标物体的中心位置
                target_object_pos = np.array([(min_bound[0]+max_bound[0])/2,(min_bound[1]+max_bound[1])/2,(min_bound[2] + max_bound[2])/2])
                args_dict['target_object_pos'] = target_object_pos
                
            if args.show_init_pos:
                points = [scene]
                for i in range(len(init_pos)):
                    points.append(
                        viz_utils.create_o3d_sphere(init_pos[i].detach().cpu().numpy().squeeze(), radius=0.03))
                o3d.visualization.draw_geometries(points)
                
            # Eval init points
            # 不进行裁剪 在opt_utils.init_points_culling函数中注释掉opt_utils.eval_init_points
            if targetobject == None:
                init_pos, init_ang = opt_utils.init_points_culling(body_model=body_model,init_pos=init_pos, vertices=vertices_org_ds,
                                scene_data=scene_data, gen_batch=gen_batch, **args_dict)
            else:                                                  
                init_pos, init_ang = opt_utils.init_points_culling_targetobject(init_pos=init_pos, vertices=vertices_org_ds,
                                                               scene_data=scene_data, gen_batch=gen_batch, **args_dict)
                if scene_name in self_define_scenes:
                    init_ang = -init_ang
                    
            if args.show_init_pos:
                points = []
                vertices_np = vertices_org.detach().cpu().numpy()
                bodies = []
                for i in range(len(init_pos)):
                    points.append(
                        viz_utils.create_o3d_sphere(init_pos[i].detach().cpu().numpy().squeeze(), radius=0.03))
                    rot_aa = torch.cat((torch.zeros((1, 2), device=device), init_ang[i].reshape(1, 1)), dim=1)
                    rot_mat = tgm.angle_axis_to_rotation_matrix(rot_aa.reshape(-1, 3))[:, :3,
                              :3].detach().cpu().numpy().squeeze()
                
                    v = np.matmul(rot_mat, vertices_np.transpose()).transpose() + init_pos[i].detach().cpu().numpy()
                    body = viz_utils.create_o3d_mesh_from_np(vertices=v, faces=faces_arr)
                    bodies.append(body)
                    # tag_point = viz_utils.create_o3d_sphere(target_object_pos.squeeze(), radius=0.5)
                    # o3d.visualization.draw_geometries([scene,body,tag_point])
                    # pdb.set_trace()

                o3d.visualization.draw_geometries(points + [scene])

            # ###########################################################################################################
            # #####################            Start of Optimization Loop                  ##############################
            # ###########################################################################################################
            results = []
            results_clothed = []
            ts = []
            angs = []
            rots = []
            joints = []
            losses = []
            # pdb.set_trace()
            #在每个可能的放置位置处调整位姿，得到优化结果最好的
            # 只优化一个位置，并可视化这个初始位姿
            # vertices_np = vertices_org.detach().cpu().numpy()
            # rot_aa = torch.cat((torch.zeros((1, 2), device=device), init_ang[0].reshape(1, 1)), dim=1)
            # rot_mat = tgm.angle_axis_to_rotation_matrix(rot_aa.reshape(-1, 3))[:, :3,:3].detach().cpu().numpy().squeeze()
            # v = np.matmul(rot_mat, vertices_np.transpose()).transpose() + init_pos[i].detach().cpu().numpy()
            # body = viz_utils.create_o3d_mesh_from_np(vertices=v, faces=faces_arr)
            # bodies.append(body)
            # o3d.visualization.draw_geometries([body,scene])
            
            for i in tqdm(range(init_pos.shape[0])):
                body_model.reset_params(**torch_param) #每次需要将人体模型复原
                t_free = init_pos[i].reshape(1, 1, 3).clone().detach().requires_grad_(True)
                ang_free = init_ang[i].reshape(1, 1).clone().detach().requires_grad_(True)    
                # t_free = init_pos[0].reshape(1, 1, 3).clone().detach().requires_grad_(True)
                # ang_free = init_ang[0].reshape(1, 1).clone().detach().requires_grad_(True) 
                free_param = [t_free, ang_free]
                if args.opt_pose:
                    free_param += [body_model.body_pose] # 这里为啥可以直接加上去呢？这里的加号不是加法的意思，是添加body_pose到参数列表的意思
                optimizer, _ = optim_factory.create_optimizer(free_param, optim_type='lbfgsls',
                                                              lr=args_dict.get('affordance_lr'), ftol=1e-9,
                                                              gtol=1e-9,
                                                              max_iter=args.max_iter)

                opt_wrapper_obj = opt_utils.opt_wrapper(vertices=vertices_org_ds.unsqueeze(0),
                                                        vertices_can=vertices_can_ds, pelvis=pelvis,
                                                        scene_data=scene_data,
                                                        down_sample_fn=down_sample_fn, down_sample_fn2=down_sample_fn2,
                                                        optimizer=optimizer, gen_batch=gen_batch, body_model=body_model,
                                                        init_body_pose=init_body_pose, **args_dict)
                
                #如果是针对某个target_object的放置，需要修改closure中的compute_loss函数
                if args_dict['target_object_pos'] is None :
                    closure = opt_wrapper_obj.create_fitting_closure(t_free, ang_free)
                else:
                     closure = opt_wrapper_obj.create_fitting_closure_object(t_free, ang_free)
                #总共迭代10次
                # logdir = "tensorboard_logs_pen_loss" 
                # writer = SummaryWriter(logdir)
                # writer = SummaryWriter()
                for epoch in range(10):
                    # recon_loss, pen_loss, pose_loss, semantic_loss , looktowards_loss = opt_wrapper_obj.compute_loss_object(t_free, ang_free)
                    loss = optimizer.step(closure)
                #     writer.add_scalar('recon_loss',recon_loss,epoch)
                #     writer.add_scalar('pen_loss',pen_loss,epoch)
                #     writer.add_scalar('pose_loss',pose_loss,epoch)
                #     writer.add_scalar('semantic_loss',semantic_loss,epoch)
                #     writer.add_scalar('looktowards_loss',looktowards_loss,epoch)
                #     writer.add_scalar('total_loss',loss,epoch)          
                # writer.flush()
                # writer.close()
                # Get body vertices after optimization
                curr_results, rot_mat = opt_wrapper_obj.compute_vertices(t_free, ang_free,
                                                                         vertices=vertices_org.unsqueeze(0),
                                                                         down_sample=False)
                # pdb.set_trace()
                if torch.is_tensor(loss):
                    loss = float(loss.detach().cpu().squeeze().numpy())
                losses.append(loss)
                results.append(curr_results.squeeze().detach().cpu().numpy())
                ts.append(t_free)
                angs.append(ang_free)
                rots.append(rot_mat)
                # pdb.set_trace()
                body_model_output = body_model(return_verts=True)

                joints.append(torch.tensor(body_model_output.joints))
                

                # Get clothed body vertices after optimization
                if vertices_clothed is not None:
                    curr_results_clothed, rot_mat = opt_wrapper_obj.compute_vertices(t_free, ang_free,
                                                                                     vertices=vertices_clothed.unsqueeze(
                                                                                         0),
                                                                                     down_sample=False)
                    results_clothed.append(curr_results_clothed.squeeze().detach().cpu().numpy())

            ###########################################################################################################
            #####################            End of Optimization Loop                  ################################
            ###########################################################################################################
    
            #存储了每一个init_pos的loss和结果
            losses = np.array(losses)
            if len(losses > 0):
                idx = losses.argmin() #取loss最小的那个下标
                print('minimum final loss = {}'.format(losses[idx]))
                sorted_ind = np.argsort(losses) #按照loss对索引排序
                for i in range(min(args.num_rendered_samples, len(losses))): #loss数组的长度却决于初始点的数目,按照loss由低到高输出最好的几个结果，一般输出一个， 因为num_rendered_samples=1
                    ind = sorted_ind[i]
                    cm = mpl_cm.get_cmap('Reds')
                    norm = mpl_colors.Normalize(vmin=0.0, vmax=1.0)
                    colors = cm(norm(losses))

                    ## Save pickle for layout
                    # 需要注意的是，源代码这里的rot_mat和t_free 没有保存loss最小的rot_mat
                    #这里修改一下，选取loss最小的结果对应的rot_mat和t_free
                    rot_mat = rots[ind]
                    t_free = ts[ind]
                    ang_free = angs[ind]
                    joint = joints[ind]
                    R_smpl2scene = torch.tensor(eulerangles.euler2mat(np.pi / 2, 0, 0, 'sxyz'), dtype=dtype,
                                                device=device)
                    Rcw = torch.matmul(rot_mat.reshape(3, 3), R_smpl2scene) #这里的rot_mat应该存储了旋转的信息 Rcw可以使得人体模型直接从smpl坐标系到POSA坐标系，并旋转到正确位置
                    torch_param = misc_utils.smpl_in_new_coords(torch_param, Rcw, t_free.reshape(1, 3),
                                                                rotation_center=pelvis, **args_dict)#这里的torch_param是准备放置到场景的人体的初始参数
                    # 建立参数表，只保存t_free,ang_free,rot_mat,vertices
                    param = {}
                    param = {'t_free':   t_free.detach().cpu().numpy(),
                             'ang_free': ang_free.detach().cpu().numpy(),
                             'rot_mat':  rot_mat.detach().cpu().numpy(),
                             'vertices': results[ind],
                             'joints' : joint.detach().cpu().numpy()
                            }
                    # print(param)
                    
                    # 保存的地址：/affordance/Layout/MPH16/result_filename.pkl
                    # 新建layout目录
                    #####不保存结果
                    # layout_folder = osp.join(affordance_dir,'Layout', scene_name)
                    # if targetobject is not  None:
                    #     result_filename = result_filename + '_{:02d}'.format(targetobject)
                    # with open(osp.join(layout_folder, 'pkls', '{}.pkl'.format(result_filename)), 'wb') as f:
                    #     pickle.dump(param, f)
                    # #result_name ： rp_xxx_posed_xxx_0_0_00 +"_targetid.pkl" :_00表示第0个触点模型

                    # # Evaluate Physical Metric
                    # gen_batch_us = up_sample_fn2.forward(gen_batch.transpose(1, 2))
                    # gen_batch_us = up_sample_fn.forward(gen_batch_us).transpose(1, 2)

                    # non_collision_score, contact_score = misc_utils.eval_physical_metric(
                    #     torch.tensor(results[ind], dtype=dtype, device=device).unsqueeze(0),
                    #     scene_data)
                    # with open(osp.join(physical_metric_folder, '{}.yaml'.format(result_filename)), 'w') as f:
                    #     yaml.dump({'non_collision_score': non_collision_score,
                    #                'contact_score': contact_score},
                    #               f)
                    ########
                    if args.viz:
                        bodies = [scene]
                        if args.use_clothed_mesh:
                            body = viz_utils.create_o3d_mesh_from_np(vertices=results_clothed[ind],
                                                                     faces=clothed_mesh.faces)
                        else:
                            body = viz_utils.create_o3d_mesh_from_np(vertices=results[ind], faces=faces_arr)
                        bodies.append(body)
                        o3d.visualization.draw_geometries(bodies)

                    if args.render or args.save_meshes:
                        default_vertex_colors = np.ones((results[ind].shape[0], 3)) * np.array(viz_utils.default_color)
                        body = trimesh.Trimesh(results[ind], faces_arr, vertex_colors=default_vertex_colors,
                                               process=False)
                        clothed_body = None
                        if args.use_clothed_mesh:
                            clothed_body = trimesh.Trimesh(results_clothed[ind], clothed_mesh.faces, process=False)

                        if args.save_meshes:
                            body.export(osp.join(affordance_dir, 'meshes', args_dict.get('scene_name'),
                                                 '{}.obj'.format(result_filename)))
                            body.export(osp.join(affordance_dir, 'Layout', args_dict.get('scene_name'), 'meshes',
                                                 '{}.obj'.format(result_filename)))
                            if args.use_clothed_mesh:
                                clothed_body.export(
                                    osp.join(affordance_dir, 'meshes_clothed', args_dict.get('scene_name'),
                                             '{}.obj'.format(result_filename)))
                                body.export(osp.join(affordance_dir, 'Layout',args_dict.get('scene_name'),'meshes_clothed',
                                                 '{}.obj'.format(result_filename)))

                        if args.render:
                            if scene_name in self_define_scenes:
                                scene_mesh = trimesh.load(osp.join("D:\data\ICAR_SE",scene_name,"env.obj"),force='mesh')
                            else:
                                scene_mesh = trimesh.load(
                                osp.join(base_dir, 'scenes', args_dict.get('scene_name') + '.ply'))
                            img_collage = viz_utils.render_interaction_snapshot(body, scene_mesh, clothed_body,
                                                                                body_center=True,
                                                                                collage_mode='horizantal', **args_dict)
                            img_collage.save(osp.join(rendering_folder, '{}.png'.format(result_filename)))
    t_end = time.time()
    cost_time = t_end-t_start
    print(f"compute time: {cost_time:.4f} seconds")
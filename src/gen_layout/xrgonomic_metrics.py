import torch
import torch.optim as Optim
import numpy as np
import open3d as o3d
import os.path as osp
import math
import torch.linalg
import pdb
import pickle
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import datetime
import layout
from gen_human import eulerangles
from gen_human import data_utils 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy.ma as ma

def compute_nicer(hand_pos,proarm_len=0.2817,forearm_len=0.2689,palm_len=0.0862,hand_len=0.1899,weight=73.0,gender='M',ang_step = math.pi/16):
   
    hand_pos = hand_pos.reshape(1,3)
    if torch.norm(hand_pos,dim=1) >= (proarm_len + forearm_len + hand_len):
        elbow_pos = hand_pos / torch.norm(hand_pos,dim=1) * proarm_len 
    else :
        hand_dir = (hand_pos)/torch.norm(hand_pos,dim=1) #[1,3]
        z_dir = torch.tensor([.0,.0,1.0],dtype=torch.float32).unsqueeze(0) 
        u = -z_dir + F.cosine_similarity(z_dir,hand_dir) * hand_dir  #[1,3]
        u = u/torch.norm(u,dim=1)
        v = torch.cross(u,hand_dir)
    
        cos_beta = (proarm_len**2 + torch.norm(hand_pos)**2 - (forearm_len + hand_len) **2)/(2 * proarm_len * torch.norm(hand_pos)) #(1,)
        sin_beta = torch.sqrt(1 - cos_beta**2) #(1,)
    
        center = cos_beta * proarm_len * hand_dir #(1,3)
        radius = sin_beta * proarm_len  #(1,)
    
        # 计算肘关节位置
        angles = torch.arange(math.pi * 1/8, math.pi * 2/4, ang_step, dtype=torch.float32)
        cos_values = torch.cos(angles).unsqueeze(1) #(36,)
        sin_values = torch.sin(angles).unsqueeze(1)
        elbow_pos = radius * (cos_values * u + sin_values * v) + center
    
    #计算质心位置
    elbow_dir = (elbow_pos)/(torch.norm(elbow_pos,dim=1).unsqueeze(1))
    elbow2hand_dir = (hand_pos - elbow_pos) / torch.norm(hand_pos - elbow_pos,dim=1).unsqueeze(1)
    
    if gender == 'M':
        proarm_mass = weight * 0.0271
        forearm_mass = weight * 0.0162
        hand_mass = weight * 0.0061
        
        proarm_com = elbow_dir * proarm_len * 0.5772
        forearm_com = elbow_pos + elbow2hand_dir * forearm_len * 0.4574
        hand_com = elbow_pos + elbow2hand_dir * (forearm_len + palm_len * 0.79)
        
    if gender == "F":
        proarm_mass = weight * 0.0255
        forearm_mass = weight * 0.0138
        hand_mass = weight * 0.0056
        
        proarm_com = elbow_dir * proarm_len * 0.5754
        forearm_com = elbow_pos + elbow2hand_dir * forearm_len * 0.4559
        hand_com = elbow_pos + elbow2hand_dir * (forearm_len + palm_len * 0.7474)    
        
    arm_mass = proarm_mass + forearm_mass + hand_mass
    
    arm_com = proarm_com * (proarm_mass / arm_mass) + forearm_com * (forearm_mass / arm_mass) + hand_com * (hand_mass / arm_mass)
    
    r = arm_com
    mg = arm_mass * torch.tensor([.0,.0,-9.8],dtype=torch.float32).unsqueeze(0)
    torque_shoulder = torch.cross(r,mg.expand_as(r)) 
     
    min_idx = torch.argmin(torque_shoulder.norm(dim=1))
    final_elbow_pos = elbow_pos[min_idx] if elbow_pos.dim() > 1 else elbow_pos
    
    upper_arm_vec = final_elbow_pos  # 从肩到肘的向量
    cos_alpha_s = F.cosine_similarity(upper_arm_vec.unsqueeze(0), torch.tensor([.0,.0,-9.8],dtype=torch.float32).unsqueeze(0), dim=1)
    alpha_s = torch.acos(torch.clamp(cos_alpha_s, -1.0, 1.0))
    alpha_s_deg = torch.rad2deg(alpha_s)
    
    # alpha_e: 肘部伸展角 - 前臂与上臂的夹角
    forearm_vec = hand_com[min_idx] - final_elbow_pos  # 从肘到手的向量
    cos_alpha_e = F.cosine_similarity(upper_arm_vec.unsqueeze(0), forearm_vec.unsqueeze(0), dim=1)
    alpha_e = torch.acos(torch.clamp(cos_alpha_e, -1.0, 1.0))
    alpha_e_deg = torch.rad2deg(alpha_e)
    
    #计算Torque_max - 使用Chaffin模型
    if gender == 'M':
        G = 0.2845
    else:
        G = 0.1495
    
    # Chaffin模型: Max_Torque = (227.338 + 0.525*α_e - 0.296*α_s) * G
    Max_Torque = (227.338 + 0.525 * alpha_e_deg - 0.296 * alpha_s_deg) * G
    
    # 计算当前力矩 Torque
    Torque = torch.norm(torque_shoulder[min_idx])
    
    #计算修正项 C(theta) - 基于论文中的sigmoid函数
    def correction_term(theta_deg):
        """
        修正项C(θ)，当肩角>90°时增加额外负荷
        基于论文公式(4)和(7)、(8)
        """
        
        # 基础sigmoid曲线 - 论文公式(4)
        f_theta = 0.0095 / (1 + torch.exp((66.40 - theta_deg) / 7.83))
        
        # 根据性别进行缩放 - 论文公式(7)、(8)
        if gender == 'M':
            # 男性修正项
            C_theta = 1230 * f_theta - torch.sin(theta_deg / 360 * math.pi * 2)/0.09
        else:
            # 女性修正项  
            C_theta = 1005 * f_theta - torch.sin(theta_deg / 360 * math.pi * 2) /0.11
        
        # 只在肩角>90°时应用修正
        C_theta = torch.where(theta_deg > 90, C_theta, torch.tensor(0.0))
        
        return C_theta
    
    C_theta = correction_term(alpha_s_deg)
    
    #计算NICER - 论文公式(9)
    exertion_percent = (Torque + C_theta) / Max_Torque
    
    return exertion_percent

def compute_elbow_pos(hand_pos,r_shoulder,body_orient,proarm_len=0.38,forearm_len=0.46,ang_step = math.pi/16):
    
    #将手的坐标旋转至肩关节坐标系下
    #在世界坐标系下，默认人与坐标轴对齐，面向y轴负方向：
    axis = np.cross(np.array([.0, -1.0, .0]), body_orient) # 旋转轴
    axis_norm = axis/np.linalg.norm(axis)
    angle = np.arccos(np.dot(np.array([.0, -1.0, 0.0]), body_orient)) #旋转角
    rot_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_norm * angle)
    
    hand_pos_trans = hand_pos - r_shoulder
    hand_pos_rot = torch.matmul(torch.tensor(rot_mat,dtype=torch.float32).t(),hand_pos_trans.reshape(3,1)).reshape(1,3)
    
    # 计算肘关节位置 
    if torch.norm(hand_pos_rot,dim=1) > (proarm_len + forearm_len):
        elbow_pos = hand_pos_rot / torch.norm(hand_pos_rot) * proarm_len
    else :
        hand_dir = (hand_pos_rot)/torch.norm(hand_pos_rot,dim=1) #[1,3]
        y_dir = torch.tensor([.0,1.0,.0],dtype=torch.float32).unsqueeze(0) #需要修改 POSA坐标系下，重力坐标方向为 (0,0,-1)
        u = -y_dir + F.cosine_similarity(y_dir,hand_dir) * hand_dir  #[1,3]
        u = u/torch.norm(u,dim=1)
        v = torch.cross(hand_dir,u) #[1,3]
    
        cos_beta = (proarm_len**2 + torch.norm(hand_pos_rot)**2 - forearm_len **2)/(2 * proarm_len * torch.norm(hand_pos_rot)) #(1,)
        sin_beta = torch.sqrt(1 - cos_beta**2) #(1,)
    
        center = cos_beta * proarm_len * hand_dir #(1,3)
        radius = sin_beta * proarm_len  #(1,)
    
        # 计算肘关节位置
        angles = -torch.arange(0, math.pi * 3/4, ang_step, dtype=torch.float32)
        cos_values = torch.cos(angles).unsqueeze(1) #(36,)
        sin_values = torch.sin(angles).unsqueeze(1)
        elbow_pos = radius * (cos_values * u + sin_values * v) + center
        
    return hand_pos_rot, elbow_pos

def compute_invof_loss(view_point,view_orient,pos):
    
    dirs = (pos - torch.tensor(view_point,dtype=torch.float32))
    cos_theta1 = F.cosine_similarity(dirs, torch.tensor(view_orient,dtype=torch.float32).view(1, -1).expand_as(dirs), dim=-1)
    inVof_loss = 1 - cos_theta1
    return inVof_loss

def compute_CE(hand_pos,proarm_len=0.2817,forearm_len=0.2689,palm_len=0.0862,hand_len=0.1899,weight=73.0,gender='M',ang_step = math.pi/16):
    
    # 计算肘关节位置
    hand_pos = hand_pos.reshape(1,3)
    if torch.norm(hand_pos,dim=1) >= (proarm_len + forearm_len + hand_len):
        elbow_pos = hand_pos / torch.norm(hand_pos,dim=1) * proarm_len 
    else :
        hand_dir = (hand_pos)/torch.norm(hand_pos,dim=1) #[1,3]
        # y_dir = torch.tensor([.0,1.0,.0],dtype=torch.float32).unsqueeze(0) #需要修改 POSA坐标系下，重力坐标方向为 (0,0,-1)
        # u = -y_dir + F.cosine_similarity(y_dir,hand_dir) * hand_dir  #[1,3]
        z_dir = torch.tensor([.0,.0,1.0],dtype=torch.float32).unsqueeze(0) #需要修改 POSA坐标系下，重力坐标方向为 (0,0,-1)
        u = -z_dir + F.cosine_similarity(z_dir,hand_dir) * hand_dir  #[1,3]
        u = u/torch.norm(u,dim=1)
        # v = torch.cross(hand_dir,u) #[1,3]
        v = torch.cross(u,hand_dir)
    
        cos_beta = (proarm_len**2 + torch.norm(hand_pos)**2 - (forearm_len + hand_len) **2)/(2 * proarm_len * torch.norm(hand_pos)) #(1,)
        sin_beta = torch.sqrt(1 - cos_beta**2) #(1,)
    
        center = cos_beta * proarm_len * hand_dir #(1,3)
        radius = sin_beta * proarm_len  #(1,)
    
        # 计算肘关节位置
        # angles = -torch.arange(0, math.pi * 3/4, ang_step, dtype=torch.float32)
        angles = torch.arange(math.pi * 1/8, math.pi * 2/4, ang_step, dtype=torch.float32)
        cos_values = torch.cos(angles).unsqueeze(1) #(36,)
        sin_values = torch.sin(angles).unsqueeze(1)
        elbow_pos = radius * (cos_values * u + sin_values * v) + center
    
    #计算质心位置
    elbow_dir = (elbow_pos)/(torch.norm(elbow_pos,dim=1).unsqueeze(1))
    elbow2hand_dir = (hand_pos - elbow_pos) / torch.norm(hand_pos - elbow_pos,dim=1).unsqueeze(1)
    
    if gender == 'M':
        proarm_mass = weight * 0.0271
        forearm_mass = weight * 0.0162
        hand_mass = weight * 0.0061
        
        proarm_com = elbow_dir * proarm_len * 0.5772
        forearm_com = elbow_pos + elbow2hand_dir * forearm_len * 0.4574
        hand_com = elbow_pos + elbow2hand_dir * (forearm_len + palm_len * 0.79)
        
    if gender == "F":
        proarm_mass = weight * 0.0255
        forearm_mass = weight * 0.0138
        hand_mass = weight * 0.0056
        
        proarm_com = elbow_dir * proarm_len * 0.5754
        forearm_com = elbow_pos + elbow2hand_dir * forearm_len * 0.4559
        hand_com = elbow_pos + elbow2hand_dir * (forearm_len + palm_len * 0.7474)    
        
    arm_mass = proarm_mass + forearm_mass + hand_mass
    
    if torch.norm(hand_pos,dim=1) <= (proarm_len + forearm_len + hand_len):
        arm_com = proarm_com * (proarm_mass / arm_mass) + forearm_com * (forearm_mass / arm_mass) + hand_com * (hand_mass / arm_mass)
    else:
        #如果交互位置超出手臂长度，修改计算重心的方式
        #计算临界条件
        if gender == 'M':
            bound_com = (proarm_len * 0.5772) * (proarm_mass / arm_mass) + (proarm_len + forearm_len * 0.4574) * (forearm_mass / arm_mass) + (proarm_len + forearm_len + palm_len * 0.79) * (hand_mass / arm_mass)
        if gender == 'F':
            bound_com = (proarm_len * 0.5754) * (proarm_mass / arm_mass) + (proarm_len + forearm_len * 0.4559) * (forearm_mass / arm_mass) + (proarm_len + forearm_len + palm_len * 0.7474) * (hand_mass / arm_mass)
        
        
        arm_len = proarm_len + forearm_len + hand_len
        arm_com = hand_pos * (bound_com / arm_len)
    
    r = arm_com
    mg = arm_mass * torch.tensor([.0,.0,-9.8],dtype=torch.float32).unsqueeze(0)
    # torque_shoulder = torch.cross(r,mg.expand_as(r)) 
    # torque_shoulder = torch.norm(torque_shoulder,dim=1)/101.6  
    torque_shoulder_cos = 1 - F.cosine_similarity(r,mg.expand_as(r),dim=-1)
    # if gender == 'M': 
    #     torque_shoulder = torch.norm(mg) * torque_shoulder_cos * torch.norm(r,dim=1)/101.6
    # if gender == 'F':
    #     torque_shoulder = torch.norm(mg) * torque_shoulder_cos * torch.norm(r,dim=1)/87.2
    if gender == 'M': 
        torque_shoulder = torch.norm(mg) * torque_shoulder_cos * torch.norm(r,dim=1)/(101.6 * (proarm_len + forearm_len + hand_len))
    if gender == 'F':
        torque_shoulder = torch.norm(mg) * torque_shoulder_cos * torch.norm(r,dim=1)/(87.2 * (proarm_len + forearm_len + hand_len))
    torque_min = torque_shoulder.min() 
    

    return torque_min

def visualize(r_shoulder,pose_list,body_mesh,scene_mesh):
    cur_pose = np.array(pose_list)
    num_pts= cur_pose.shape[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cur_pose.reshape(-1,3))
    #渐变色数组
    start_rgb = np.array([1.0, .0, 0.0]) 
    end_rgb = np.array([.0, .0, 1.0])
    r = np.linspace(start_rgb[0],end_rgb[0],num_pts)
    g = np.linspace(start_rgb[1],end_rgb[1],num_pts)
    b = np.linspace(start_rgb[2],end_rgb[2],num_pts)
    gradient_array = [(r[i],g[i],b[i]) for i in range(num_pts)]

    pcd.colors = o3d.utility.Vector3dVector(gradient_array)
    r_shoulder_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=50)
    color_shoulder = np.array([255, 0, 0])  # 设置为红色
    r_shoulder_mesh.paint_uniform_color(color_shoulder/255.0)
    r_shoulder_mesh.translate(r_shoulder.reshape(3,1))
    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

    o3d.visualization.draw_geometries([r_shoulder_mesh,pcd,body_mesh,scene_mesh,mesh_frame])

def visualize_onlybody(pos_list,body_path):
    pkl_file_path = body_path +'.pkl'
    body_mesh_path = body_path +'.obj'
    with open(pkl_file_path, 'rb') as f:
        param = pickle.load(f) 
    model_path = 'D:\data\\POSA_dir\smplx_models'
    _, _, _, body_model, _, _, _, _  = data_utils.pkl_to_canonical(pkl_file_path=pkl_file_path,device="cpu",dtype=torch.float32,batch_size=1,gender='male', model_folder=model_path, vertices_clothed=None)
    body_model_output = body_model(return_verts=True)
    joints = torch.tensor(body_model_output.joints).squeeze()
    r_shoulder = joints[17,:].reshape(1,3)
    #将mesh旋转到肩关节坐标系下
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    body_mesh = o3d.io.read_triangle_mesh(body_mesh_path)
    r_shoulder_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=50)
    color_shoulder = np.array([255, 0, 0])  # 设置为红色
    r_shoulder_mesh.paint_uniform_color(color_shoulder/255.0)
    
    # 坐标系变换到肩关节为坐标原点
    R_smpl2scene = torch.tensor(eulerangles.euler2mat(np.pi / 2, 0, 0, 'sxyz'),dtype=torch.float32)
    body_mesh.rotate(R_smpl2scene.numpy(),center=(0, 0, 0))
    r_shoulder = torch.matmul(R_smpl2scene,r_shoulder.reshape(3,1))
    r_shoulder_mesh.translate(r_shoulder.numpy().reshape(3,1))
    
    body_mesh.translate(-r_shoulder.numpy().reshape(3,1))
    r_shoulder_mesh.translate(-r_shoulder.numpy().reshape(3,1))
        
    cur_pose = np.array(pos_list)
    num_pts= cur_pose.shape[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cur_pose.reshape(-1,3))
    #渐变色数组
    start_rgb = np.array([1.0, .0, 0.0]) 
    end_rgb = np.array([.0, .0, 1.0])
    r = np.linspace(start_rgb[0],end_rgb[0],num_pts)
    g = np.linspace(start_rgb[1],end_rgb[1],num_pts)
    b = np.linspace(start_rgb[2],end_rgb[2],num_pts)
    gradient_array = [(r[i],g[i],b[i]) for i in range(num_pts)]

    pcd.colors = o3d.utility.Vector3dVector(gradient_array)
    o3d.visualization.draw_geometries([body_mesh,r_shoulder_mesh,mesh_frame,pcd])

def visualiaze_voxel(is_face=True):
    '''
    可视化人体面前的一面墙，用颜色表示ce_loss的高低
    body_path : 人体模型的存放位置
    '''
    # 生成网格坐标
    grid_size = 0.1
    if is_face:
        x = np.linspace(-0.4, 0.4, 9)
        fixed_y = -0.65
        z = np.linspace(0, 0.4, 5) 
        X, Z = np.meshgrid(x, z)
        Y = np.full_like(X, fixed_y)
        grid_points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    else:
        fixed_x = 0.1
        y = np.linspace(-0.1,-1.5,16)
        z = np.linspace(-0.5,0.8,14) 
        Y,Z = np.meshgrid(y,z)
        X = np.full_like(Y,fixed_x)
        grid_points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    # pdb.set_trace()
    #计算loss
    num_pts = grid_points.shape[0]
    
    ce_loss_list = []
    ece_loss_list = []
    
    nicer_loss_list = []

    for i in range(num_pts):

        ce_i = compute_CE_old(torch.tensor(grid_points[i],dtype=torch.float32))
        ce_loss_list.append(ce_i)
        ece_i = compute_CE(torch.tensor(grid_points[i],dtype=torch.float32))
        ece_loss_list.append(ece_i)
        nicer_i = compute_nicer(torch.tensor(grid_points[i],dtype=torch.float32))
        nicer_loss_list.append(nicer_i.item())
        
    
    ce_loss = np.array(ce_loss_list)  
    ece_loss = np.array(ece_loss_list)
    nicer_loss = np.array(nicer_loss_list)
    
    
    ce = ce_loss.reshape(5,9)
    ece = ece_loss.reshape(5,9)
    nicer = nicer_loss.reshape(5,9)
    
    
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))

    x_values = np.linspace(-0.4, 0.4, 9)
    y_values = np.linspace(0, 0.4, 5)
    
    #第一个留空
    ax1.axis('off')
    font_s = 16
    
    # 绘制第一个网格图
    im2 = ax2.imshow(ce, cmap='YlGnBu', origin='lower',extent=[x_values.min(), x_values.max(), y_values.min(), y_values.max()])
    ax2.grid(False)
    ax2.set_title('CE',fontsize=font_s)
    ax2.set_xlabel('X',fontsize=font_s)
    ax2.set_ylabel('Z',fontsize=font_s)
    ax2.tick_params(axis='both',labelsize=font_s)
    fig.colorbar(im2, ax=ax2, shrink=0.4)


    # 绘制第二个网格图
    im3 = ax3.imshow(ece, cmap='YlGnBu', origin='lower',extent=[x_values.min(), x_values.max(), y_values.min(), y_values.max()])
    ax3.grid(False)
    ax3.set_title('ECE',fontsize=font_s)
    ax3.set_xlabel('X',fontsize=font_s)
    ax3.set_ylabel('Z',fontsize=font_s) 
    ax3.tick_params(axis='both',labelsize=font_s)

    fig.colorbar(im3, ax=ax3, shrink=0.4)

    
    #绘制网格的第三个图
    # pdb.set_trace()
    im4 = ax4.imshow(nicer, cmap='YlGnBu', origin='lower',extent=[x_values.min(), x_values.max(), y_values.min(), y_values.max()])
    ax4.grid(False)
    ax4.set_title('NICER',fontsize=font_s)
    ax4.set_xlabel('X',fontsize=font_s)
    ax4.set_ylabel('Z',fontsize=font_s) 
    ax4.tick_params(axis='both',labelsize=font_s)

    fig.colorbar(im4, ax=ax4, shrink=0.4)

    

    plt.tight_layout()
    plt.show()

def visualization_elbow_pos(body_path,elbow_pos,hand_pos):
    """可视化所有可能的肘关节点

    Args:
        body_path (_string_): 人体模型路径
        elbow_pos (_tensor(n,3)_): 所有肘关节位置
        hand_pos (_tensor(3)_): 手的位置
        
    """
    all_meshes = []
    
    # 人体mesh
    pkl_file_path = body_path +'.pkl'
    body_mesh_path = body_path +'.obj'
    with open(pkl_file_path, 'rb') as f:
        param = pickle.load(f) 
    model_path = 'D:\data\\POSA_dir\smplx_models'
    _, _, _, body_model, _, _, _, _  = data_utils.pkl_to_canonical(pkl_file_path=pkl_file_path,device="cpu",dtype=torch.float32,batch_size=1,gender='male', model_folder=model_path, vertices_clothed=None)
    body_model_output = body_model(return_verts=True)
    joints = torch.tensor(body_model_output.joints).squeeze()
    r_shoulder = joints[17,:].reshape(1,3)
    body_mesh = o3d.io.read_triangle_mesh(body_mesh_path)
    R_smpl2scene = torch.tensor(eulerangles.euler2mat(np.pi / 2, 0, 0, 'sxyz'),dtype=torch.float32)
    body_mesh.rotate(R_smpl2scene.numpy(),center=(0, 0, 0))
    r_shoulder = torch.matmul(R_smpl2scene,r_shoulder.reshape(3,1))
    body_mesh.translate(-r_shoulder.numpy().reshape(3,1))
    
    all_meshes.append(body_mesh)
    
    #肩关节mesh
    shoulder_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=50)
    color_shoulder = np.array([255, 0, 0])  # 设置为红色
    shoulder_mesh.paint_uniform_color(color_shoulder/255.0)
    all_meshes.append(shoulder_mesh)
    
    #肘关节mesh
    for i in range(elbow_pos.shape[0]):
        elbow_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=50)
        color_elbow = np.array([0, 0, 255])  # 设置为蓝色
        elbow_mesh.paint_uniform_color(color_elbow/255.0)
        elbow_mesh.translate(elbow_pos[i].detach().numpy().reshape(3))
        all_meshes.append(elbow_mesh)
    
    #手指尖mesh
    hand_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=50)
    color_hand = np.array([255, 0, 0])  # 设置为红色
    hand_mesh.paint_uniform_color(color_hand/255.0)
    hand_mesh.translate(hand_pos.detach().numpy().reshape(3))
    all_meshes.append(hand_mesh)
    
    o3d.visualization.draw_geometries(all_meshes)
    
def visualization_elbow_com(body_path,elbow,com,hand_pos):
    """可视化肘关节位置以及重心位置

    Args:
        body_path (_string_): 人体模型路径
        elbow (_tensor(3)_): 肘关节位置
        com (_tensor(3)_): 手臂重心
        hand_pos (_tensor(3)_): 手的位置
    """
    all_meshes = []
    
    # 人体mesh
    pkl_file_path = body_path +'.pkl'
    body_mesh_path = body_path +'.obj'
    with open(pkl_file_path, 'rb') as f:
        param = pickle.load(f) 
    model_path = 'D:\data\\POSA_dir\smplx_models'
    _, _, _, body_model, _, _, _, _  = data_utils.pkl_to_canonical(pkl_file_path=pkl_file_path,device="cpu",dtype=torch.float32,batch_size=1,gender='male', model_folder=model_path, vertices_clothed=None)
    body_model_output = body_model(return_verts=True)
    joints = torch.tensor(body_model_output.joints).squeeze()
    r_shoulder = joints[17,:].reshape(1,3)
    body_mesh = o3d.io.read_triangle_mesh(body_mesh_path)
    R_smpl2scene = torch.tensor(eulerangles.euler2mat(np.pi / 2, 0, 0, 'sxyz'),dtype=torch.float32)
    body_mesh.rotate(R_smpl2scene.numpy(),center=(0, 0, 0))
    r_shoulder = torch.matmul(R_smpl2scene,r_shoulder.reshape(3,1))
    body_mesh.translate(-r_shoulder.numpy().reshape(3,1))
    
    all_meshes.append(body_mesh)
    
    #肩关节mesh
    shoulder_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=50)
    color_shoulder = np.array([255, 0, 0])  # 设置为红色
    shoulder_mesh.paint_uniform_color(color_shoulder/255.0)
    all_meshes.append(shoulder_mesh)
    
    #肘关节mesh
    elbow_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=50)
    color_elbow = np.array([0, 0, 255])  # 设置为蓝色
    elbow_mesh.paint_uniform_color(color_elbow/255.0)
    elbow_mesh.translate(elbow.detach().numpy().reshape(3))
    all_meshes.append(elbow_mesh)
    
    #手指尖mesh
    hand_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=50)
    color_hand = np.array([255, 0, 0])  # 设置为红色
    hand_mesh.paint_uniform_color(color_hand/255.0)
    hand_mesh.translate(hand_pos.detach().numpy().reshape(3))
    all_meshes.append(hand_mesh)
    
    #重心mesh
    com_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=50)
    color_com = np.array([255, 0, 255])  # 设置为紫色
    com_mesh.paint_uniform_color(color_com/255.0)
    com_mesh.translate(com.detach().numpy().reshape(3))
    all_meshes.append(com_mesh)
    
    o3d.visualization.draw_geometries(all_meshes) 

if __name__ == '__main__':
    visualiaze_voxel()
    
    

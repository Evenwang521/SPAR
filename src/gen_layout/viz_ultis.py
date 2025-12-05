import open3d as o3d
import numpy as np
import torch
import os
import json
import pdb
import os.path as osp
from obb import EDGES
import eulerangles
import misc_utils


def viz_body_info(body_info,scene_mesh=None,viz = False):
    '''
    curr_orient_norm       人体的朝向信息
    view_point             视点信息
    l_shoulder,,r_shoulder 肩膀关节点位置  
    pelvis                 骨盆位置   
    '''
    # 获取关节点信息
    joints = np.squeeze(body_info['joints'])
    pelvis = joints[0,:].reshape(1,3)
    head = joints[15,:].reshape(1,3)
    l_eye = joints[23,:].reshape(1,3)
    r_eye = joints[24,:].reshape(1,3)
    l_shoulder = joints[16,:].reshape(1,3)
    r_shoulder = joints[17,:].reshape(1,3)
    
    # 计算变换矩阵
    R_smpl2scene = eulerangles.euler2mat(np.pi / 2, 0, 0, 'sxyz')
    Rcw = body_info['rot_mat'].reshape(3, 3) @ R_smpl2scene
    
    # 变换关节点坐标
    head = (Rcw @ (head-pelvis).T).T + body_info['t_free'].reshape(1,3)# ((3,3)@(3,1)).T = (1,3)
    l_eye = (Rcw @ (l_eye-pelvis).T).T + body_info['t_free'].reshape(1,3)
    r_eye = (Rcw @ (r_eye-pelvis).T).T + body_info['t_free'].reshape(1,3)
    l_shoulder = (Rcw @ (l_shoulder-pelvis).T).T + body_info['t_free'].reshape(1,3)
    r_shoulder = (Rcw @ (r_shoulder-pelvis).T).T + body_info['t_free'].reshape(1,3)
    pelvis = body_info['t_free'].reshape(1,3)
    
    # 计算人体朝向信息，以方向向量的形式可视化
    view_point = (l_eye + r_eye)/2.0    # 以两只眼睛的中间为视点 (1,3)
    orig_orient = np.array([0.0,-1.0, 0.0]) # 初始朝向为y轴负方向
    curr_orient = (body_info['rot_mat'].reshape(3, 3) @ orig_orient.T).T # 旋转之后的朝向 (1,3)
    curr_orient_norm = curr_orient / np.linalg.norm(curr_orient)
    
    if viz:
        # 绘制各个关节点的信息
        s_head = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=50)
        color_head = np.array([255, 0, 0])  # 设置为红色
        s_head.paint_uniform_color(color_head/255.0)
        s_head.translate(head.reshape(3,))
        
        s_pelvis = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=50)
        color_pelvis = np.array([0, 255, 0])  # 设置为绿色
        s_pelvis.paint_uniform_color(color_pelvis / 255.0)
        s_pelvis.translate(pelvis.reshape(3,))
        
        s_ls = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=50)
        s_rs = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=50)
        color_shoulder = np.array([0, 0, 255])  # 设置为蓝色
        s_ls.paint_uniform_color(color_shoulder / 255.0)
        s_rs.paint_uniform_color(color_shoulder / 255.0)
        s_ls.translate(l_shoulder.reshape(3,))
        s_rs.translate(r_shoulder.reshape(3,))

        s_le = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=50)
        s_re = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=50)
        color_eye = np.array([128, 0, 128])  # 设置为紫色
        s_le.paint_uniform_color(color_eye / 255.0)
        s_re.paint_uniform_color(color_eye / 255.0)
        s_le.translate(l_eye.reshape(3,))
        s_re.translate(r_eye.reshape(3,))
        
        # 可视化朝向信息为箭头
        view_arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cylinder_height = 0.3,cone_radius=0.02, cone_height=0.03)
        arrow_color = np.array([255,182,193])  # RGB颜色，这里设置为粉色
        view_arrow.paint_uniform_color(arrow_color/255.0)
        # 旋转该箭头的方向，首先从(0,0,1)转向(0,-1,0)，然后乘上旋转矩阵
        axis = np.cross(np.array([0.0, 0.0, 1.0]), orig_orient) # 旋转轴
        angle = np.arccos(np.dot(np.array([0.0, 0.0, 1.0]), orig_orient)) #旋转角
        z2ny_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        # view_arrow.rotate(z2ny_mat)
        view_arrow.rotate(body_info['rot_mat'].reshape(3, 3) @ z2ny_mat,center=np.array([0,0,0]))
        translation_vector = view_point.reshape(3,)
        view_arrow.translate(translation_vector)
        #  绘制点云
        o3d.visualization.draw_geometries([s_head,s_pelvis,s_ls,s_rs,s_le,s_re,scene_mesh,view_arrow])
        pdb.set_trace()
        o3d.visualization.draw_geometries([s_head,s_pelvis,s_ls,s_rs,s_le,s_re,view_arrow])
    
    # 返回当前人体的朝向信息，视点位置，左肩膀位置，右肩膀位置，骨盆位置
    return curr_orient_norm,view_point,l_shoulder,r_shoulder,pelvis

def viz_trun_range(trun_range,scene_mesh,body_mesh):
    
    # 计算边界框的顶点 8个
    vertices = np.array([[trun_range[0][0],trun_range[0][1],trun_range[0][2]],
                         [trun_range[1][0],trun_range[0][1],trun_range[0][2]],
                         [trun_range[0][0],trun_range[1][1],trun_range[0][2]],
                         [trun_range[1][0],trun_range[1][1],trun_range[0][2]],
                         [trun_range[0][0],trun_range[0][1],trun_range[1][2]],
                         [trun_range[1][0],trun_range[0][1],trun_range[1][2]],
                         [trun_range[0][0],trun_range[1][1],trun_range[1][2]],
                         [trun_range[1][0],trun_range[1][1],trun_range[1][2]]])
    # 线 12条
    lines = [[0, 1], [0, 2], [0, 4], [1, 5], [1, 3], [2, 3],
             [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(vertices)
    lineset.lines = o3d.utility.Vector2iVector(lines) 
    line_color = np.array([255,182,193])/255.0
    lineset.colors = o3d.utility.Vector3dVector(np.array([line_color for _ in range(len(lines))]))
    o3d.visualization.draw_geometries([lineset,scene_mesh,body_mesh])

def viz_sdf(sdf_data, scene_mesh,resolution=256):
    # pdb.set_trace()
    # 得到的SDF数据是D*D*D*1的
    grid_min = sdf_data['grid_min']
    grid_max = sdf_data['grid_max']
    grid_dim = sdf_data['grid_dim']
    scene_sdf = sdf_data['sdf']
    trun_range = np.array([grid_min.detach().numpy(),grid_max.detach().numpy()])
    voxel_size = (trun_range[1]-trun_range[0]) / resolution
    x_range = np.arange(trun_range[0][0], trun_range[1][0], voxel_size[0])
    y_range = np.arange(trun_range[0][1], trun_range[1][1], voxel_size[1])
    z_range = np.arange(trun_range[0][2], trun_range[1][2], voxel_size[2])
    grid_points = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1, 3) 
    grid_points = torch.tensor(grid_points,dtype=torch.float32).unsqueeze(0) #(1,N,3)
    
    # 采样点
    x = misc_utils.read_sdf(grid_points, scene_sdf, grid_dim, grid_min, grid_max, mode='bilinear').squeeze()
    num_pts = len(x)
    # 创建颜色
    colors = np.zeros((num_pts, 3))
    # 对于正的 SDF 值，颜色从绿色到红色渐变
    positive_mask = x >= 0
    colors[positive_mask, 0] = 1  # 设置红色通道为1
    colors[positive_mask, 1] = 1 - np.abs(x[positive_mask])  # 设置绿色通道渐变

    # 对于负的 SDF 值，颜色从绿色到蓝色渐变
    negative_mask = x < 0
    colors[negative_mask, 1] = 1  # 设置绿色通道为1
    colors[negative_mask, 2] = 1 - np.abs(x[negative_mask])  # 设置蓝色通道渐变
    
    # 可视化
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(grid_points.detach().numpy().reshape(-1,3))
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([point_cloud,scene_mesh])

def viz_ele(ele_path):
    '''
    暂时可视化Obj,不添加材质和纹理
    '''
    mesh = o3d.io.read_triangle_mesh(ele_path)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    o3d.visualization.draw_geometries([mesh,mesh_frame])
    # o3d.visualization.draw_geometries([mesh])

def viz_opti_eles(ele_list,ele_pos,ele_orient,scene_mesh,body_mesh):
    
    eles = []
    for i in range(len(ele_list)):
        mesh = o3d.io.read_triangle_mesh(ele_list[i])
        y_prime = torch.cross(torch.tensor([0.0, 0.0, 1.0],dtype=torch.float32),ele_orient[i])
        y_prime = y_prime / torch.norm(y_prime)
        z_prime = torch.cross(ele_orient[i],y_prime)
        z_prime = z_prime / torch.norm(z_prime)
        rot_mat = torch.stack([ele_orient[i], y_prime, z_prime], dim=0).t()
        mesh.rotate(rot_mat.detach().numpy(),center=(0, 0, 0))
        mesh.translate(ele_pos[i].detach().numpy().reshape(3,))
        eles.append(mesh)
        
    eles.append(scene_mesh)
    eles.append(body_mesh)  
    o3d.visualization.draw_geometries(eles, mesh_show_wireframe=False,mesh_show_back_face=True)

def viz_opti_eles_obbs(curr_obbs,ele_list,ele_pos,ele_orient,scene_mesh,body_mesh):
    eles = []
    for i in range(len(ele_list)):    
        mesh = o3d.io.read_triangle_mesh(ele_list[i])
        y_prime = torch.cross(torch.tensor([0.0, .0, 1.0],dtype=torch.float32),ele_orient[i])
        z_prime = torch.cross(ele_orient[i],y_prime)
        rot_mat = torch.stack([ele_orient[i], y_prime, z_prime], dim=0).t()
        mesh.rotate(rot_mat.detach().numpy(),center=(0, 0, 0))
        mesh.translate(ele_pos[i].detach().numpy().reshape(3,))
        eles.append(mesh)
        
        vertices,lines = get_lines_obb(curr_obbs[i].detach().numpy())
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(vertices)
        lineset.lines = o3d.utility.Vector2iVector(lines) 
        line_color = np.array([255,182,193])/255.0
        lineset.colors = o3d.utility.Vector3dVector(np.array([line_color for _ in range(len(lines))]))
        eles.append(lineset)
        
    eles.append(scene_mesh)
    eles.append(body_mesh) 
     
    o3d.visualization.draw_geometries(eles, mesh_show_wireframe=False,mesh_show_back_face=True)

def viz_init_pos_orient(num_eles,init_pos,init_orient,scene_mesh,body_mesh):
    '''
    可视化初始化位置的目标位置和朝向
    '''
    #点
    start_pts = init_pos.detach().numpy()
    init_dir = init_orient.detach().numpy()
    end_pts = start_pts + init_dir * 0.3
    pcd_start = o3d.geometry.PointCloud()
    pcd_start.points = o3d.utility.Vector3dVector(start_pts)
    pcd_start.paint_uniform_color([.0,1.0,.0]) #初始点是绿色
    pcd_end = o3d.geometry.PointCloud()
    pcd_end.points = o3d.utility.Vector3dVector(end_pts)
    pcd_end.paint_uniform_color([.0,.0, .0]) #初始点是蓝色
    
    #线
    vertices = np.concatenate((start_pts,end_pts),axis=0)
    lines_ls = []
    for i in range(num_eles):
        line = np.array([i,i + num_eles])
        lines_ls.append(line)
    lines = np.array(lines_ls)
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(vertices)
    lineset.lines = o3d.utility.Vector2iVector(lines) 
    line_color = np.array([255,0,0])/255.0
    lineset.colors = o3d.utility.Vector3dVector(np.array([line_color for _ in range(len(lines))]))
    # pdb.set_trace()
    #验证矩阵计算是否正确
    #验证(1,0,0)按照计算旋转矩阵的方式能否旋转到目标位置
    x_dir = torch.tensor([1.0,.0,.0],dtype=torch.float32)
    target_dir = init_orient[3]/torch.norm(init_orient[3])
    axis = torch.cross(x_dir, target_dir)
    axis_norm = axis/torch.norm(axis)
    angle = torch.acos(torch.dot(x_dir, target_dir)) #旋转角
    rot_mat = tgm.angle_axis_to_rotation_matrix(angle * axis_norm.reshape(-1,3))[:,:3,:3].squeeze(0)
    now_dir = torch.matmul(rot_mat,x_dir).detach().numpy()
    new_end_pt = start_pts[3] + now_dir * 0.5 #长于0.3
    new_vertices = np.array([start_pts[3],new_end_pt])
    new_lines = np.array([0,1])
    new_lineset = o3d.geometry.LineSet()
    new_lineset.points = o3d.utility.Vector3dVector(new_vertices)
    new_lineset.lines = o3d.utility.Vector2iVector(np.expand_dims(new_lines,axis=0))
    new_line_color = np.array([128,0,128])/255.0
    new_lineset.colors = o3d.utility.Vector3dVector(np.array([new_line_color for _ in range(len(new_lines))]))
    
    o3d.visualization.draw_geometries([lineset,new_lineset,pcd_start,pcd_end,scene_mesh,body_mesh])
    
def viz_obb(obb,scene_mesh):
    '''
    可视化元素ele及其包围盒
    '''
    # 计算边界框的顶点 8个
    vertices,lines = get_lines_obb(obb)
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(vertices)
    lineset.lines = o3d.utility.Vector2iVector(lines) 
    line_color = np.array([255,182,193])/255.0
    lineset.colors = o3d.utility.Vector3dVector(np.array([line_color for _ in range(len(lines))]))
    
    o3d.visualization.draw_geometries([lineset,scene_mesh])

def get_lines_obb(obb):
    '''
    计算一个包围盒所有的点和线
    '''
    vertices = np.array([[obb[0][0],obb[0][1],obb[0][2]],
                         [obb[1][0],obb[0][1],obb[0][2]],
                         [obb[0][0],obb[1][1],obb[0][2]],
                         [obb[1][0],obb[1][1],obb[0][2]],
                         [obb[0][0],obb[0][1],obb[1][2]],
                         [obb[1][0],obb[0][1],obb[1][2]],
                         [obb[0][0],obb[1][1],obb[1][2]],
                         [obb[1][0],obb[1][1],obb[1][2]]])
    # 线 12条
    lines = [[0, 1], [0, 2], [0, 4], [1, 5], [1, 3], [2, 3],
             [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    return vertices, lines 

def get_trun_range(view_point,l_shoulder,r_shoulder,pelvis,arm_len):
    '''
    计算截断区域
    z 轴：骨盆高度以上，视点高度以下；xy 轴：右侧肩膀xy轴+-臂长 
    由于人体朝向迥异，无法只截取人体面前一截
    需要参数:手臂长度 ，默认手臂长度为80cm
    '''
    min_x = min([l_shoulder.flatten()[0],r_shoulder.flatten()[0]])
    min_y = min([l_shoulder.flatten()[1],r_shoulder.flatten()[1]])
    min_z = pelvis.reshape(3,)[2]
    max_x = max([l_shoulder.flatten()[0],r_shoulder.flatten()[0]])
    max_y = max([l_shoulder.flatten()[1],r_shoulder.flatten()[1]])
    max_z = view_point.reshape(3,)[2]
    trun_range = np.array([[min_x - arm_len, min_y - arm_len, min_z], 
                           [max_x + arm_len, max_y + arm_len, max_z + 0.3]])
    
    return trun_range

def viz_obbs(scene_mesh,obbs_scene,obbs_ele,r_shoulder,rot_w2rs):
    '''可视化计算得到的OBB(需要将场景mesh变换到肩关节局部坐标系)

    Args:
        scene_mesh (_o3d.geometry.trianglemesh_): 场景mesh
        obbs_scene (_list(OBB)_): 场景物体肩关节坐标系下的OBB
        obbs_ele (_list(OBB)_): AR元素在肩关节坐标系下的OBB
        r_shoulder (_tensor(3,)_): 世界坐标系下的肩关节坐标
        rot_w2rs (_tensor(3,3)_): 世界坐标系到肩关节坐标系的旋转矩阵
    '''
    geometries = []
    scene_mesh.translate(-r_shoulder.detach().numpy().reshape(3,))
    scene_mesh.rotate((rot_w2rs.t()).detach().numpy(),center = (0,0,0))
    o3d.io.write_triangle_mesh("copy_of_knot.ply", scene_mesh)
    geometries.append(scene_mesh)
    for i in range(len(obbs_scene)):
        vertices_i = obbs_scene[i]._vertices.detach().numpy() 
        lineset_i = o3d.geometry.LineSet()
        lineset_i.points = o3d.utility.Vector3dVector(vertices_i)
        lineset_i.lines = o3d.utility.Vector2iVector(EDGES) 
        line_color_i = np.array([255,0,0])/255.0
        lineset_i.colors = o3d.utility.Vector3dVector(np.array([line_color_i for _ in range(len(EDGES))]))
        geometries.append(lineset_i)
    
    for i in range(len(obbs_ele)):    
        vertices_ele = obbs_ele[i]._vertices.detach().numpy() 
        lineset_ele = o3d.geometry.LineSet()
        lineset_ele.points = o3d.utility.Vector3dVector(vertices_ele)
        lineset_ele.lines = o3d.utility.Vector2iVector(EDGES) 
        line_color_ele = np.array([0,0,255])/255.0
        lineset_ele.colors = o3d.utility.Vector3dVector(np.array([line_color_ele for _ in range(len(EDGES))]))
        geometries.append(lineset_ele)
    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5,origin=(0,0,0))
    geometries.append(mesh_frame)
        
    o3d.visualization.draw_geometries(geometries)

def viz_results(scene_mesh,body_mesh,ele_meshes,r_shoulder,rot_w2rs,ele_pos,view_point_rc,viz_body=False):
    '''可视化最终的优化结果(肩关节坐标下)

    Args:
        scene_mesh (_o3d.geometry.trianglemesh_): 场景mesh
        body_mesh (_o3d.geometry.trianglemesh_): 人体mesh
        ele_meshes (_list(o3d.geometry.trianglemesh)_): AR元素mesh
        r_shoulder (_tensor(3,)_): 世界坐标系下的肩关节坐标
        rot_w2rs (_tensor(3,3)_): 世界坐标系到肩关节坐标系的旋转矩阵
        ele_pos (_tensor(3)_): AR元素在局部坐标系下的位置
        view_point_rc (_tensor(1,3)_): 视点在肩关节局部坐标下下的坐标
        viz_body (_bool_) : 是否可视化人体
    '''
    geometries = []
    
    scene_mesh.translate(-r_shoulder.detach().numpy().reshape(3,))
    scene_mesh.rotate((rot_w2rs.t()).detach().numpy(),center = (0,0,0))
    geometries.append(scene_mesh)
    
    if viz_body:
        body_mesh.translate(-r_shoulder.detach().numpy().reshape(3,))
        body_mesh.rotate((rot_w2rs.t()).detach().numpy(),center = (0,0,0))
        geometries.append(body_mesh)
    
    for i in range(len(ele_meshes)):   
        ele_orient =torch.tensor([1.0,1.0,.0],dtype=torch.float32)*(view_point_rc - ele_pos[i]) #投影到xoy平面上
        x_prime = ele_orient / torch.norm(ele_orient,dim=1)
        x_prime = x_prime.squeeze(0)
        z_prime = torch.tensor([.0,.0,1.0],dtype=torch.float32)
        y_prime = torch.cross(z_prime,x_prime)
        rot_mat = torch.stack([x_prime, y_prime, z_prime], dim=0).t()
        
        ele_mesh = ele_meshes[i]
        ele_mesh.rotate(rot_mat.detach().numpy(),center=(0, 0, 0))
        ele_mesh_center = ele_mesh.get_center()
        ele_mesh_t = ele_pos[i].squeeze(0).detach().numpy().reshape(3,) - ele_mesh_center
        ele_mesh.translate(ele_mesh_t)
        
        geometries.append(ele_mesh)
        
    o3d.visualization.draw_geometries(geometries)

def viz_results_baseline(scene_mesh,body_mesh,ele_meshes,r_shoulder,rot_w2rs,ele_pos,view_point_rc,target_obj_orient,viz_body=False):
    '''可视化最终的优化结果(肩关节坐标下)

    Args:
        scene_mesh (_o3d.geometry.trianglemesh_): 场景mesh
        body_mesh (_o3d.geometry.trianglemesh_): 人体mesh
        ele_meshes (_list(o3d.geometry.trianglemesh)_): AR元素mesh
        r_shoulder (_tensor(3,)_): 世界坐标系下的肩关节坐标
        rot_w2rs (_tensor(3,3)_): 世界坐标系到肩关节坐标系的旋转矩阵
        ele_pos (_tensor(3)_): AR元素在局部坐标系下的位置
        view_point_rc (_tensor(1,3)_): 视点在肩关节局部坐标下下的坐标
        target_obj_orient (_tensor(3,)_) : 目标物体的朝向
        viz_body (_bool_) : 是否可视化人体
    '''
    geometries = []
    
    scene_mesh.translate(-r_shoulder.detach().numpy().reshape(3,))
    scene_mesh.rotate((rot_w2rs.t()).detach().numpy(),center = (0,0,0))
    geometries.append(scene_mesh)
    
    if viz_body:
        body_mesh.translate(-r_shoulder.detach().numpy().reshape(3,))
        body_mesh.rotate((rot_w2rs.t()).detach().numpy(),center = (0,0,0))
        geometries.append(body_mesh)
    
    for i in range(len(ele_meshes)):    
        x_prime = target_obj_orient

        z_prime = torch.tensor([.0,.0,1.0],dtype=torch.float32)
        y_prime = torch.cross(z_prime,x_prime)
        rot_mat = torch.stack([x_prime, y_prime, z_prime], dim=0).t()
        
        ele_mesh = ele_meshes[i]
        ele_mesh.rotate(rot_mat.detach().numpy(),center=(0, 0, 0))
        ele_mesh_center = ele_mesh.get_center()
        ele_mesh_t = ele_pos[i].squeeze(0).detach().numpy().reshape(3,) - ele_mesh_center
        ele_mesh.translate(ele_mesh_t)
        
        geometries.append(ele_mesh)
        
    o3d.visualization.draw_geometries(geometries)
     
def viz_culling(scene_mesh,aabbs,labels_intersection):
    '''可视化发生穿插的OBB

    Args:
        scene_mesh (_o3d.geometry.trianglemesh_): 场景mesh
        aabbs (_array(n,2,3)_): 场景实例的AABB包围盒
        labels_intersection (_type_): 发生碰撞的OBB标号
        
    '''
    geometries = []
    geometries.append(scene_mesh)
    
    for i in range(len(labels_intersection)):
        label_i = labels_intersection[i]
        aabb_i = aabbs[i]
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(aabb_i[0], aabb_i[1])
        geometries.append(bounding_box)
    
    o3d.visualization.draw_geometries(geometries)
    
def viz_scene_ins(scene_name,obj_id): 
    scenes_aabb_folder = r"D:\code\python\POSA\output\scenes_ins_aabb"
    scenes_folder = r"D:\data\POSA_dir\scenes"
    save_base = r"D:\code\python\POSA\output\scenes_ins"
    scene_mesh = o3d.io.read_triangle_mesh(osp.join(scenes_folder,scene_name+'.ply'))
    scene_aabb = osp.join(scenes_aabb_folder,scene_name+'.json')
    geometries = []
    geometries.append(scene_mesh)
    with open(scene_aabb,'r') as f:
        data = json.load(f)
    ins_aabb = json.loads(data)
    for ins in ins_aabb:
        if ins['instance_id']== obj_id:
            id = ins['instance_id']
            min_bound = ins["min_bound"]
            max_bound = ins["max_bound"]
            vertices,lines = get_lines_obb(np.array([min_bound,max_bound]))
            lineset = o3d.geometry.LineSet()
            lineset.points = o3d.utility.Vector3dVector(vertices)
            lineset.lines = o3d.utility.Vector2iVector(lines) 
            line_color = np.array([255,0,0])/255.0
            lineset.colors = o3d.utility.Vector3dVector(np.array([line_color for _ in range(len(lines))]))
            geometries.append(lineset)
            break
    o3d.visualization.draw_geometries(geometries)    

if __name__ == "__main__":
    viz_scene_ins("MPH11",5)
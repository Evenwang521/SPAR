import torch
import json
import numpy as np
import open3d as o3d
import os.path as osp
from obb import OBB,UNIT_BOX

self_define_scenes =  {'S1_E1', 'S1_E2', 'S2_E1', 'S2_E2'}

def load_sdf_data(sdf_dir):
    
    with open(sdf_dir + '.json','r') as f:
        sdf_data = json.load(f)
        grid_dim = sdf_data['dim']
        badding_val = sdf_data['badding_val']
        grid_min = torch.tensor(np.array(sdf_data['min']), dtype=torch.float32)
        grid_max = torch.tensor(np.array(sdf_data['max']), dtype=torch.float32)
        voxel_size = (grid_max - grid_min) / grid_dim
        bbox = torch.tensor(np.array(sdf_data['bbox']), dtype=torch.float32)  
         
    sdf = np.load(sdf_dir + '_sdf.npy').astype(np.float32)
    sdf = sdf.reshape(grid_dim, grid_dim, grid_dim, 1)
    sdf = torch.tensor(sdf, dtype=torch.float32)
    
    return {'grid_dim': grid_dim, 'grid_min': grid_min,
            'grid_max': grid_max, 'voxel_size': voxel_size,
            'bbox': bbox, 'badding_val': badding_val,
            'sdf': sdf}
    
def load_vertices_obb(ele_list,num_pts):
    '''
    load vertices and obbs for each element in ele_list
    
    structure of element folder : 
    |-- UI_1
    |   |-- UI.mtl
    |   `-- UI.obj
    |-- UI_2
    |   |-- UI.mtl
    |   `-- UI.obj
    `-- UI_3
        |-- UI.mtl
        `-- UI.obj
    暂时只放没有纹理和材质的空白版UI元素
    
    create interactive panel with blender
    POSA: XYZ 
    blender: XZY
    '''
    vertices = []
    ind_vertices = []
    obbs = []
    cnt_vertices = 0
    ind_vertices.append(cnt_vertices)
    for i in range(len(ele_list)):
        mesh_i = o3d.io.read_triangle_mesh(ele_list[i])
        points_i = mesh_i.sample_points_uniformly(number_of_points=num_pts)
        # 对所有的物体的mesh均匀采样
        aabb_i = mesh_i.get_axis_aligned_bounding_box()
        vertices.append(torch.tensor(np.array(points_i.points),dtype=torch.float32))
        cnt_vertices += np.array(points_i.points).shape[0]
        ind_vertices.append(cnt_vertices)
        obbs.append(torch.tensor(np.array([aabb_i.get_min_bound(),aabb_i.get_max_bound()]),dtype=torch.float32))
        
    return torch.cat(vertices),torch.stack(obbs),torch.tensor(ind_vertices) # the number of vertices of each mesh is different!

def load_scene_aabbs(scene_name,scene_aabbs_folder):
    '''计算场景中所有分割的AABB包围盒

    Args:
        scene_name (_string_): 场景名称
        scene_aabbs_folder (_string_): 存放aabb包围盒的

    Returns:
        _array(n,2,3)_: 所有分割实例的包围盒
        _array(n,)_: 所有分割实例的包围盒的宽度
    '''
    obbs_path = osp.join(scene_aabbs_folder,scene_name+'.json')
    with open(obbs_path,'r') as obbf:
        ins_aabb = json.load(obbf)
    if scene_name not in self_define_scenes:
        ins_aabb = json.loads(ins_aabb)
    num_ins = len(ins_aabb)
    aabbs_ls = []
    for i in range(num_ins):
        ins_i = ins_aabb[i]
        min_i = ins_i['min_bound']
        max_i = ins_i['max_bound']
        aabbs_ls.append(np.array([min_i,max_i]))
    #convert list to array
    aabbs = np.array(aabbs_ls)  
      
    return aabbs

def create_scene_obbs(scene_aabbs,rotation,translation):
    '''计算场景中所有包围盒的OBB(肩关节局部坐标系下的)

    Args:
        scene_aabbs (_array(n,2,3)_): 包围盒的min_bound,max_bound
        rotation (_tensor(3,3)_): 世界坐标系到局部坐标系的旋转
        translation (_tensor(3)_): 世界坐标系到局部坐标系的平移

    Returns:
        _list(OBB)_: 场景中所有包围盒的OBB列表
    '''
    obbs_list = []
    num_instances = scene_aabbs.shape[0]
    scale_aabbs = torch.tensor((scene_aabbs[:,1,:] - scene_aabbs[:,0,:]),dtype=torch.float32)
    pos_aabbs = torch.tensor(((scene_aabbs[:,0,:] + scene_aabbs[:,1,:])/2.0),dtype=torch.float32)
    for i in range(num_instances):
        scale_i = scale_aabbs[i].reshape(1,3)
        rotation_i = rotation.t()
        orig_vertices_i = torch.tensor(UNIT_BOX,dtype=torch.float32) * scale_i + pos_aabbs[i]
        vertices_i = torch.matmul(rotation.t(),(orig_vertices_i - translation).t()).t()  
        translation_i = torch.matmul(rotation.t(),(pos_aabbs[i] - translation).unsqueeze(0).t()).t().squeeze() 
        obb_i = OBB(scale_i,rotation_i,translation_i,vertices_i)
        obbs_list.append(obb_i)
    return obbs_list
    
def create_ele_obbs(ele_meshes,view_point_rc,init_pos_rc):
    '''计算AR元素mesh在局部坐标系下的OBB包围盒

    Args:
        ele_meshes (_o3d.geometry.trianglemesh_): 排布元素的mesh
        view_point_rc (_tensor(3,)_): 局部坐标系下视点的
        init_pos_rc (_tensor(n,3)_): 世界坐标系到局部坐标系的平移

    Returns:
        _list(Box)_: UImesh在局部坐标系下的OBB列表
    '''
    
    obbs_ele = []
    for i in range(len(ele_meshes)):
        ele_mesh = ele_meshes[i]
        aabb_ele = ele_mesh.get_axis_aligned_bounding_box()
        aabb = torch.tensor(np.array([aabb_ele.get_min_bound(),aabb_ele.get_max_bound()]),dtype=torch.float32)
        scale_ele = (aabb[1,:] - aabb[0,:]).unsqueeze(0)
        
        x_prime = (torch.tensor((view_point_rc - init_pos_rc[i]),dtype=torch.float32))
        x_prime *= torch.tensor([1.0,1.0,.0],dtype=torch.float32)
        x_prime /= torch.norm(x_prime)
        z_prime = torch.tensor([.0,.0,1.0],dtype=torch.float32)
        y_prime = torch.cross(z_prime,x_prime)
        
        rot_ui_rs = torch.stack((x_prime,y_prime,z_prime),dim=0).t()
        vertices_ui = torch.matmul(rot_ui_rs,(torch.tensor(UNIT_BOX,dtype=torch.float32) * scale_ele).t()).t() + init_pos_rc[i]
        obb_ele = OBB(scale_ele,rot_ui_rs,init_pos_rc[i],vertices_ui)
        obbs_ele.append(obb_ele)
        
    return obbs_ele

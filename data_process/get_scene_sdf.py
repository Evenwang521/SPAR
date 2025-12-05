from mesh_to_sdf import get_surface_point_cloud, scale_to_unit_sphere
import trimesh
import skimage, skimage.measure
import os
import os.path as osp
import pdb
import numpy as np
import re
import json
import time
import open3d as o3d
import mesh2sdf
from scipy.spatial import KDTree
from get_aabb import category_mapping
from PIL import Image

color_mapping = {
    2: (220, 220, 220),   # light gray for "floor"
    3: (255, 160, 122),   # light salmon for "chair"
    5: (240, 230, 140),   # khaki for "table"
    7: (144, 238, 144),   # light green for "cabinet"
    10: (173, 216, 230),  # light blue for "sofa"
    34: (255, 182, 193),  # light pink for "seating"
    36: (221, 160, 221),  # plum for "clothrack"
    39: (255, 218, 185)   # peach puff for "computer" and "printer"
}

def get_query_points_mesh2sdf(mesh,grid_size=256):
    
    # pdb.set_trace()
    # 获取AABB的最小和最大边界点
    aabb = mesh.bounding_box
    aabb_min, aabb_max = aabb.bounds
    bbox = aabb_max - aabb_min
    scale = 2.0 * 0.8 / bbox.max()
    center = (aabb_min + aabb_max) / 2
    # pdb.set_trace()
    # query_points = np.stack(np.meshgrid(np.arange(grid_size), np.arange(grid_size), np.arange(grid_size)), -1).reshape(-1, 3)
    query_points = np.transpose(np.stack(np.meshgrid(np.arange(grid_size), np.arange(grid_size), np.arange(grid_size)), -1),(1,0,2,3)).reshape(-1, 3)
    query_points = (query_points/grid_size - 0.5) * 2 #[0,1]
    query_points = query_points / scale + center
    
    return query_points

def compute_sdf_mesh2sdf(mesh,mesh_name,grid_size=512,save=True):
        
    # normalize mesh
    mesh_scale = 0.8
    size = grid_size
    level = 2 / size
    
    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale

    # fix mesh
    t0 = time.time()
    sdf, mesh = mesh2sdf.compute(
        vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)
    t1 = time.time()
    
    # 计算并输出时间差
    elapsed_time = t1 - t0
    print(f'It took {elapsed_time:.4f} seconds to compute the SDF of {mesh_name}.')
    
    sdf = sdf / scale
    
    if save:
        sdf_path = osp.join("D:\data\ICAR_SE\sdf",f"{mesh_name}_sdf.npy")
        np.save(sdf_path, sdf.reshape(-1))
    
    return sdf.reshape(-1)

def load_scene_meshes(scene_name,scene_folder,mesh_grid_size=64):
    
    # pdb.set_trace()
    mesh_base = osp.join(scene_folder, scene_name)
    objs_trans = np.load(osp.join(mesh_base, 'objs_transform.npy'),allow_pickle=True).item()
    meshes = []
    labels = []
    # sdfs = []
    for instance_id,(obj_name, transform_matrix) in enumerate(objs_trans.items()):
        # obj_name_clean = re.sub(r'[^A-Za-z]', '', obj_name)
        obj_name_clean = re.sub(r'\d', '', obj_name)
        labels.append(category_mapping[obj_name_clean])
        
        mesh_path = osp.join(mesh_base,obj_name_clean, obj_name_clean + '.obj')
        mesh = trimesh.load(mesh_path, force='mesh')
        mesh.apply_transform(transform_matrix)
        meshes.append(mesh)
    
        # sdf = compute_sdf_mesh2sdf("S1_E1",mesh,obj_name,mesh_grid_size)
        # sdf = np.load(osp.join("D:\data\ICAR_SE\data_process\output\sdf\mesh2sdf",scene_name, f"{obj_name}.npy"))
        # sdfs.append(sdf)
        
    # return meshes,labels,sdfs
    return meshes,labels

def viz_pcd(scene_mesh,labels,sdf):
    
    query_points = get_query_points_mesh2sdf(scene_mesh, 256).astype(np.float32)
    negative_mask = sdf < 0
    negative_points = query_points[negative_mask]
    negative_labels = labels[negative_mask]
    negative_colors = np.array([color_mapping[label] for label in negative_labels])
    point_cloud = trimesh.points.PointCloud(vertices=negative_points, colors=negative_colors)
    # scene = trimesh.Scene([scene_mesh, point_cloud])
    # scene.show()
    return point_cloud
    
def viz_marchingcube(sdf,grid_size,orig_mesh):
    
    voxels = sdf.reshape(grid_size,grid_size,grid_size)
    # Choose a level within this range, e.g., the midpoint
    _level =  2 / grid_size
    vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=_level)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    # Combine the marching cube mesh with the other mesh in a scene
    # scene = trimesh.Scene([mesh, orig_mesh])
    # # Visualize the scene
    # scene.show()
    return mesh

def get_scene_json(scene_name,scene_folder,dim = 512,padding_value = 0):
    
    scene_mesh = trimesh.load(osp.join(scene_folder, scene_name, "env.obj"), force='mesh')
    aabb = scene_mesh.bounding_box
    bbox_min, bbox_max = aabb.bounds
    scale = 2.0 * 0.8 / (bbox_max-bbox_min).max()
    center = (bbox_min + bbox_max) / 2
    grid_min = (np.array([0, 0, 0],dtype=np.float32)/dim - 0.5) * 2 / scale + center
    grid_max = (np.array([dim, dim, dim],dtype=np.float32)/dim -0.5) * 2 /scale + center
    
    padding_value = ((grid_max - grid_min) - (bbox_max - bbox_min)).max() / 2
    
    bbox = [bbox_max.tolist(), bbox_min.tolist()]
    grid_size = grid_max - grid_min
    voxel_size = grid_size / dim
    aabb_dict = {
    "min": grid_min.tolist(),
    "max": grid_max.tolist(),
    "dim": dim,
    "bbox": bbox,
    "badding_val": padding_value,  # 假设的填充值，可以根据需要进行调整
    "voxel_size": voxel_size.tolist()  # 将 voxel_size 转换为列表
    }
    
    save_path = osp.join(scene_folder, "sdf", f"{scene_name}.json")
    with open(save_path, 'w') as f:
        json.dump(aabb_dict, f, indent=4)
            
    return aabb_dict

def sdf_merge(scene_name, query_points, mesh_list, labels, batch_size=10000):
    # Prepare KDTree for each SDF
    kdtree_list = []
    for i in range(len(mesh_list)):
        # s_pts = get_query_points_mesh2sdf(mesh_list[i], grid_size).astype(np.float32)
        # kdtree = KDTree(s_pts)
        # kdtree_list.append(kdtree)
        kdtree = KDTree(mesh_list[i].vertices.astype(np.float32))
        kdtree_list.append(kdtree)
    
    # Initialize result array
    N = query_points.shape[0]
    # nearest_sdf_with_label = np.full((N, 2), np.inf)
    nearest_distance = np.full(N, np.inf)
    nearest_sdf_with_label = np.zeros(N,dtype=np.int)
    
    # Process in batches
    print("======================merging============================")
    for start in range(0, N, batch_size):
        print(f"==========================={start}=============================")
        end = min(start + batch_size, N)
        batch_query_points = query_points[start:end]
        
        for i in range(len(mesh_list)):
            # print(f"SDF {i} is being merged...")
            # s_sdf = sdf_list[i]
            kdtree = kdtree_list[i]
            label = labels[i]
            
            # Query KDTree for nearest neighbors
            # distances, indices = kdtree.query(batch_query_points, k=1)
            # nearest_local_sdf = s_sdf[indices.flatten()]
            # # Update nearest SDF and label
            # update_mask = (distances.flatten() < nearest_sdf_with_label[start:end, 0])
            # nearest_sdf_with_label[start:end, 0][update_mask] = nearest_local_sdf[update_mask]
            # nearest_sdf_with_label[start:end, 1][update_mask] = label
            
            distances, _ = kdtree.query(batch_query_points, k=1)
            update_mask = (distances.flatten() < nearest_distance[start:end])
            nearest_sdf_with_label[start:end][update_mask] = label
            nearest_distance[start:end][update_mask] = distances.flatten()[update_mask]
            
    
    # Save results
    print("====================saving=========================")
    
    semantic_save_path = f"D:/data/ICAR_SE/sdf/{scene_name}_semantics.npy" 
    np.save(semantic_save_path, nearest_sdf_with_label)

    return nearest_sdf_with_label
   
def process_scene(scene_name,scene_folder,dim = 256, mesh_grid_size=64,  padding_value = 0):
    #get scene_json
    # pdb.set_trace()
    print(f"======================{scene_name} is processing==========================")
    
    get_scene_json(scene_name,scene_folder,dim,padding_value)
    
    print(f"======================json of {scene_name} is done==========================")
    
    #get scene sdf
    scene_mesh = trimesh.load(osp.join(scene_folder, scene_name, "env.obj"), force='mesh')
    compute_sdf_mesh2sdf(scene_mesh,scene_name,dim,save=True)
    
    print(f"======================sdf of {scene_name} is done==========================")
    
    
    #get scene_sdf.json and scene_semantics.json
    mesh_list,labels = load_scene_meshes(scene_name,scene_folder,mesh_grid_size)
    scene_mesh = trimesh.load(osp.join(scene_folder, scene_name, "env.obj"), force='mesh')
    query_points = get_query_points_mesh2sdf(scene_mesh,dim)
    sdf_semantic = sdf_merge(scene_name,query_points,mesh_list,labels)
    
    print(f"======================semantic of {scene_name} is done==========================")
    return sdf_semantic

def exp_sdf_imgs(scene_name):
    mesh_path = f"D:/data/ICAR_SE/images/{scene_name}_mesh.png"
    sdf_path = f"D:/data/ICAR_SE/images/{scene_name}_sdf.png"
    semantic_path = f"D:/data/ICAR_SE/images/{scene_name}_semantic.png" 
    
    scene_path = f"D:/data/ICAR_SE/{scene_name}/env.obj"
    mesh = trimesh.load(scene_path,force="mesh")
    sdf = np.load(osp.join("D:\data\ICAR_SE\sdf", f"{scene_name}_sdf.npy"))
    semantic_labels = np.load(osp.join("D:\data\ICAR_SE\sdf", f"{scene_name}_semantics.npy"))
    
    scene = trimesh.Scene()
    scene_path = f"D:/data/ICAR_SE/{scene_name}/env.obj"
    mesh = trimesh.load(scene_path,force="mesh")
    scene.add_geometry(mesh)
    scene.show()
    camera_transform = scene.camera_transform
    scene_img = scene.save_image(visible=True,resolution=(1920,1080))
    with open(mesh_path,"wb") as f:
        f.write(scene_img)
        f.close()
    del scene
    
    sdf_mesh = viz_marchingcube(sdf,256,mesh)
    scene = trimesh.Scene()
    scene.add_geometry(sdf_mesh)
    scene.camera_transform = camera_transform
    scene.show()
    scene_img = scene.save_image(visible=True,resolution=(1920,1080))
    with open(sdf_path,"wb") as f:
        f.write(scene_img) 
        f.close() 
    del scene
    
    semantic_pcd = viz_pcd(mesh,semantic_labels,sdf)
    scene = trimesh.Scene()
    scene.add_geometry(semantic_pcd)
    scene.camera_transform = camera_transform
    scene.show()    
    scene_img = scene.save_image(visible=True,resolution=(1920,1080))
    with open(semantic_path,"wb") as f:
        f.write(scene_img) 
        f.close
    del scene
   
if __name__ == '__main__':

    scene_name = "S2_E1"
    exp_sdf_imgs(scene_name)
    # labels = process_scene(scene_name,"D:\data\ICAR_SE",256,128)
    # labels = np.load(osp.join("D:\data\ICAR_SE\sdf", f"{scene_name}_semantics.npy"))
    # scene_mesh = trimesh.load(osp.join("D:\data\ICAR_SE", scene_name,"env.obj"), force='mesh')
    # sdf = np.load(osp.join("D:\data\ICAR_SE\sdf", f"{scene_name}_sdf.npy"))
    # scene_sdf = np.load(f"D:/data/ICAR_SE/sdf/{scene_name}_sdf.npy")
    # viz_marchingcube(scene_sdf,256,scene_mesh)
    # viz_pcd(scene_mesh,labels,sdf)
    
    
    

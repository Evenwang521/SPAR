import copy
import json
import pickle
import numpy as np
import open3d as o3d
import os
import os.path as osp
import pandas as pd
from PIL import ImageColor
import pdb
import argparse

def get_instance_aabb(scene_name,use_semantic):
    
    #需要确保得到的点云数据是从POSA_dir下的场景中得到的
    scene_path = os.path.join("D:\data\\POSA_dir\scenes", scene_name + '.ply')
    scene_seg_folder = 'D:\data\\PROXE_ExpansionPack\scenes_semantics'
    scene_semantic_path = os.path.join(scene_seg_folder, scene_name + '_withlabels.ply') if scene_name != 'N0Sofa' else os.path.join(scene_seg_folder, scene_name + '_withlabels_old.ply')
    scene_original = o3d.io.read_triangle_mesh(scene_path)
    scene = o3d.io.read_triangle_mesh(scene_semantic_path)
    scene_mesh = scene_original
    
    # 语义标签是 255/5 = 51 最多有52类物体 物体的语义信息包含在顶点颜色当中
    scene_semantic_label = (np.asarray(scene.vertex_colors)[:, 0] * 255 / 5).astype(int) 
    # 按照类别将所有的点进行划分，保存的是点的索引
    objects_idx = {}
    for idx, vertex_color in enumerate(scene.vertex_colors):
        category = int(vertex_color[0] * 255 / 5)
        if not category in objects_idx:
            objects_idx[category] = []
        objects_idx[category].append(idx)
   
    # 加载保存类被颜色属性的文件 mpcat40.tsv
    mptsv_path = "D:\code\python\POSA\mpcat40.tsv"
    category_dict = pd.read_csv(mptsv_path, sep='\t')
    category_dict['color'] = category_dict.apply(lambda row: np.array(ImageColor.getrgb(row['hex'])), axis=1)
    
    # POSA2PROX
    with open("D:\code\python\POSA\output\\registration.pkl", 'rb') as file:
        POSA_to_PROX_transform = pickle.load(file)  
    transform = np.linalg.inv(POSA_to_PROX_transform[scene_name])
    
    # 总共有 42 categories in total, exclude wall-1 and floor-2？
    # 对于每一类物体分别处理
    instance_id = 0
    instance_list = []
    for category in range(1, 42):
        if category in objects_idx:
            if (category == 17):  # also exclude ceiling
                # print(objects_idx[category])
                continue
            #把所有类别是当前category的点挑出来
            object = scene.select_by_index(objects_idx[category])
            vis_color = np.array(category_dict.loc[category]['color']) / 255

            # 把挑出来的点连接成三角面片并划分成簇
            (cluster_idxs, num_faces, areas) = object.cluster_connected_triangles()
            #实例的数量就是簇的个数
            num_instances = np.asarray(num_faces).shape[0]
            instances_idx = {}
            for idx in range(num_instances):
                instances_idx[idx] = set()
            for idx, cluster_idx in enumerate(cluster_idxs):
                vertices = object.triangles[idx]
                for vertex in vertices:
                    instances_idx[cluster_idx].add(vertex)
            for idx in range(num_instances):
                instance = object.select_by_index(list(instances_idx[idx]))
                instance = instance.transform(transform)
                if np.asarray(instance.vertices).shape[0] < 100 or \
                        (np.asarray(instance.vertices).shape[0] < 2000 and category <= 2) or \
                        (np.asarray(instance.vertices).shape[0] < 500 and category != 39):  # TODO maybe we need a category specific threshold, for objects, this should be smaller
                    # if DEBUG:
                    #     print("discard an isntance of ", category_dict.loc[category]['mpcat40'], " with ", np.asarray(instance.vertices).shape[0], "points")
                    continue
                instance.paint_uniform_color(vis_color)
                aabb = instance.get_axis_aligned_bounding_box()
                aabb.color = vis_color
                instance_list.append({'instance_id':instance_id,  'category':category, 'aabb': aabb, 'min_bound': aabb.min_bound, 'max_bound': aabb.max_bound, 'pointcloud':o3d.geometry.PointCloud(instance.vertices)}
                                     )
                instance_id += 1
    
    save_folder = 'D:\code\python\\POSA\output\scenes_ins_aabb'
    print(save_folder)
    os.makedirs(save_folder, exist_ok=True)
        
    # 可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080, top=0, left=0, visible=True)
    vis.get_render_option().mesh_show_back_face = True
    vis.get_render_option().line_width = 20.0
    
    geometries = []
    # visualize scene
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0])
    geometries.append(mesh_frame)
    if scene_name in ['Werkraum', 'MPH1Library'] and not use_semantic:
        geometries.append(scene_mesh)
    else:
        for ins in instance_list:
            orig_aabb = ins['aabb']
            # new_aabb = transform_aabb(orig_aabb,transform)
            # geometries.append(new_aabb)
            geometries.append(orig_aabb)
            # geometries.append(ins['pointcloud'])
        print("1")
        geometries.append(scene_mesh)
    for geometry in geometries:
        vis.add_geometry(geometry)

    vis.poll_events()
    vis.update_renderer()
    vis.run()
    
    # 获取用户输入
    user_input = input("按下 's' 键并按 Enter 以保存：")

    # 检查用户输入是否为 's'
    if user_input.lower() == 's':
        # 将 instance_id , category, min_bound, max_bound写入json文件
        interested_keys =['instance_id' , 'category', 'min_bound', 'max_bound']
        filtered_instances = [{key: data[key] for key in interested_keys} for data in instance_list]
        #序列化filtered_instances 以便于存储
        filtered_instances = json.dumps(filtered_instances,default = serialize_instance,indent=None)
        # 这里可以添加保存数据的代码，例如保存到文件
        save_path =os.path.join(save_folder,scene_name + ".json")
        with open(save_path, 'w') as json_file:
            json.dump(filtered_instances, json_file)
        print("数据已保存")
    else:
        print("未保存数据")

def serialize_instance(instance):
    if (isinstance(instance,np.ndarray)):
        return instance.tolist()

def transform_aabb(aabb, transformation_matrix):
    # 将最小边界和最大边界转换为齐次坐标
    min_bound_homogeneous = np.concatenate([aabb.min_bound, np.ones(1)])
    max_bound_homogeneous = np.concatenate([aabb.max_bound, np.ones(1)])

    # 应用变换矩阵
    min_bound_transformed = np.dot(transformation_matrix, min_bound_homogeneous)[:-1]
    max_bound_transformed = np.dot(transformation_matrix, max_bound_homogeneous)[:-1]

    # 获取新的最小边界和最大边界
    new_min_bound = np.min(np.stack([min_bound_transformed, max_bound_transformed]), axis=0)
    new_max_bound = np.max(np.stack([min_bound_transformed, max_bound_transformed]), axis=0)

    return o3d.geometry.AxisAlignedBoundingBox(new_min_bound, new_max_bound)                


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_name',type=str, default='Werkraum')
    args = parser.parse_args()
    scene_name = args.scene_name
    get_instance_aabb(scene_name,use_semantic=True)
    aabb_dir = osp.join('D:\code\python\\POSA\output\scenes_ins_aabb',scene_name+'.json')
    with open(aabb_dir,'r') as fp:
        data = json.load(fp)
    data = json.loads(data)
    
       
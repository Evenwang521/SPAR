import bpy
import numpy as np
import os
import os.path as osp
import json
import shutil
import trimesh
import trimesh.scene
import pickle
import open3d as o3d
import pdb
from gen_human import eulerangles
from gen_layout import viz_ultis
from PIL import Image

self_define_scenes =  {'S1_E1', 'S1_E2', 'S2_E1', 'S2_E2'}

scene_body = [
    ["1", "S1_E1", "rp_ethan_posed_012_0_0_00_00"],
    ["1", "S1_E2", "rp_petra_posed_019_0_0_00_06"],
    ["2", "S2_E1", "rp_alexandra_posed_001_0_0_00_01"],
    ["2", "S2_E2", "rp_carla_posed_002_0_0_00_02"],
    ["3", "BasementSittingBooth", "rp_ethan_posed_012_0_0_00_04"],
    ["3", "N0SittingBooth", "rp_ethan_posed_012_0_0_00_02"],
    ["4", "MPH8", "rp_ethan_posed_012_0_0_00_05"],
    ["4", "MPH11", "rp_ethan_posed_012_0_0_00_05"],
    ["4", "MPH16", "rp_ethan_posed_012_0_0_00_05"],
    ["4", "MPH1Library", "rp_corey_posed_005_0_0_00_11"],
    ["4", "N3Library", "rp_corey_posed_005_0_0_00_09"],
    ["4", "Werkraum", "rp_ethan_posed_012_0_0_00_06"],
    ["4", "N3Office", "rp_ethan_posed_012_0_0_00_06"]
]

def merge_one(scenario_id,scene_name,body_name,tag="ours",with_human=True):
    
    #clean scene
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    scenario_base = rf"D:\data\POSA_dir\results\scenario_{scenario_id}"
    temp_path = osp.join(scenario_base,scene_name + "_" + body_name,"temp")
    
    #load scene_mesh
    if scene_name in self_define_scenes:
        scene_path =  osp.join(temp_path,"env.obj") 
        bpy.ops.import_scene.obj(filepath=scene_path,use_split_objects=False)
    else:
        scene_path = osp.join(temp_path,f"{scene_name}.ply") 
        bpy.ops.wm.ply_import(filepath=scene_path,forward_axis='NEGATIVE_Z', up_axis='Y')

    
    #load body_mesh
    if with_human:
        body_path = osp.join(temp_path,f"{body_name}.obj")
        bpy.ops.import_scene.obj(filepath=body_path,use_split_objects=False)
    
    #load ele_mesh
    
    if tag == "ours":
        eles_path_base = osp.join(temp_path,"virtual_elements")
    else:
        eles_path_base = osp.join(temp_path,"virtual_elements_baseline")
    
    for ele in os.listdir(eles_path_base):
        ele_path = osp.join(eles_path_base,ele,ele+'.obj')
        bpy.ops.import_scene.obj(filepath=ele_path,use_split_objects=False)
       
    #export path
    if tag == "ours":
        if with_human :
            output_obj_path = osp.join(scenario_base,scene_name+"_"+body_name,"ours",scene_name+"_"+body_name+".obj")  # 导出 .obj 文件的路径
        else:
            output_obj_path = osp.join(scenario_base,scene_name+"_"+body_name,"ours",scene_name+"_"+body_name+"_pure.obj")  # 导出 .obj 文件的路径
            
        texture_output_dir = osp.join(scenario_base,scene_name+"_"+body_name,"ours","textures")  # 导出纹理的文件夹路径
    else:
        if with_human :
            output_obj_path = osp.join(scenario_base,scene_name + "_" + body_name,"baseline",scene_name+"_"+body_name+".obj")  # 导出 .obj 文件的路径
        else:
            output_obj_path = osp.join(scenario_base,scene_name + "_" + body_name,"baseline",scene_name+"_"+body_name+"_pure.obj")  # 导出 .obj 文件的路径
           
        texture_output_dir = osp.join(scenario_base,scene_name + "_" + body_name,"baseline","textures")  # 导出纹理的文件夹路径

    if not osp.exists(osp.dirname(output_obj_path)):
        os.makedirs(osp.dirname(output_obj_path))
    
    bpy.ops.export_scene.obj(filepath=output_obj_path,
                         use_materials=True,           # Export materials
                         use_triangles=True,           # Export faces as triangles (if needed)
                         use_normals=True,             # Export normals
                         use_uvs=True,                 # Export UVs
                         path_mode='RELATIVE')         # Forces relative paths in .mtl
    
    print(f"All objects have been merged and exported to {output_obj_path}.")
    
    if os.path.exists(texture_output_dir) and os.path.isdir(texture_output_dir):
        shutil.rmtree(texture_output_dir)
        print(f"Deleted folder: {texture_output_dir}")
    else:
        print(f"Folder does not exist: {texture_output_dir}")
    
def test_json(scenario_id,scene_name,body_name,tag="ours"):
    
    scenario_base = rf"D:\data\POSA_dir\results\scenario_{scenario_id}"
    #load scene_path
    if scene_name in self_define_scenes:
        scene_path = f"D:\data\ICAR_SE\{scene_name}\env.obj"
    else:
        scene_path = f"D:\data\POSA_dir\scenes\{scene_name}.ply"
        
    #load body_path
    body_path = rf"D:\data\POSA_dir\affordance\Layout\{scene_name}\meshes\{body_name}.obj"
        
    #load ele matrix
    if tag == "ours":
        json_path = osp.join(scenario_base,scene_name + "_" + body_name,"virtual_eles.json")
    else:
        json_path = osp.join(scenario_base,scene_name + "_" + body_name,"virtual_eles_baseline.json")
    
    with open(json_path,'r') as f:
        eles_data = json.load(f) 
    
    #import objs
    
    geometries = []
    #import ele objs
    #load ele mesh
    ele_base = rf"D:\data\POSA_dir\virtual_elements\scenario_{scenario_id}"
     
    for obj_data in eles_data :
        name = obj_data["name"]
        mat = obj_data["mat"]
        ele_path = osp.join(ele_base,name,name +'.obj')
        mesh = trimesh.load(ele_path,force='mesh')
        mesh.apply_transform(np.array(mat))
        geometries.append(mesh)
        
        
    #import scene_obj
    scene_mesh = trimesh.load(scene_path,force="mesh")  
    geometries.append(scene_mesh)
    
    #import body_obj
    body_mesh = trimesh.load(body_path,force="mesh") 
    geometries.append(body_mesh)
    
    scene = trimesh.Scene()
    scene.add_geometry(geometries)
    scene.show()

def transform_obj(scenario_id,scene_name,body_name,tag="ours"):
    
    # get transform matrix for scene_mesh and body_mesh
    # load body_pkl  
    pkl_file_path = osp.join(rf"D:\data\POSA_dir\affordance\Layout\{scene_name}\pkls\{body_name}.pkl")
    with open(pkl_file_path,'rb') as f:
        body_info = pickle.load(f)
    curr_orient_norm,_,_,r_shoulder,_ = viz_ultis.viz_body_info(body_info) 
    axis = np.cross(np.array([.0, -1.0, .0]), curr_orient_norm) # 旋转轴
    axis_norm = axis/np.linalg.norm(axis)
    angle = np.arccos(np.dot(np.array([.0, -1.0, 0.0]), curr_orient_norm)) #旋转角
    rot_w2rc = np.array(o3d.geometry.get_rotation_matrix_from_axis_angle(axis_norm * angle))
    mat = np.eye(4)
    mat[:3,:3] = rot_w2rc.T
    mat[:3,3] = -rot_w2rc.T @ r_shoulder.reshape(3,)
    
    # scene_mesh
    scenario_base = rf"D:\data\POSA_dir\results\scenario_{scenario_id}"
    
    #load scene_path
    if scene_name in self_define_scenes:
        scene_path = f"D:\data\ICAR_SE\{scene_name}\env.obj"
    else:
        scene_path = f"D:\data\POSA_dir\scenes\{scene_name}.ply"
    
    # temp file
    temp_path = osp.join(scenario_base,scene_name + "_" + body_name,"temp")
    if not osp.exists(temp_path):
        os.makedirs(temp_path)     
    #transform scene_mesh
    scene_mesh = trimesh.load(scene_path,force="mesh")
    scene_mesh.apply_transform(mat)
    if scene_name in self_define_scenes:
        scene_mesh.export(osp.join(temp_path,"env.obj"))
        shutil.copy(f"D:\data\ICAR_SE\{scene_name}\env.mtl",temp_path)
    else:
        scene_mesh.export(osp.join(temp_path,f"{scene_name}.ply"))
        
    #transform body_mesh
    body_path = rf"D:\data\POSA_dir\affordance\Layout\{scene_name}\meshes\{body_name}.obj"
    body_mesh = trimesh.load(body_path)
    body_mesh.apply_transform(mat)
    body_mesh.export(osp.join(temp_path,f"{body_name}.obj"))
    
    # ele_mesh
    if tag == "ours":
        eles_base_path = osp.join(temp_path ,"virtual_elements")
        eles_trans_path = osp.join(scenario_base,scene_name + "_" + body_name,"virtual_eles.json")
    else:
        eles_base_path = osp.join(temp_path ,"virtual_elements_baseline")
        eles_trans_path = osp.join(scenario_base,scene_name + "_" + body_name,"virtual_eles_baseline.json")
    
    with open(eles_trans_path,'r') as f:
        eles_data = json.load(f)
        
    for data in eles_data:
        ele_name = data["name"]
        ele_mat = data["mat"] 
        ele_mesh = trimesh.load(rf"D:\data\POSA_dir\virtual_elements\scenario_{scenario_id}\{ele_name}\{ele_name}.obj")
        ele_mesh.apply_transform(ele_mat)
        ele_path = osp.join(eles_base_path,ele_name)
        if not osp.exists(ele_path):
            os.makedirs(ele_path)
        ele_mesh.export(f"{ele_path}\{ele_name}.obj")
        shutil.copy(rf"D:\data\POSA_dir\virtual_elements\scenario_{scenario_id}\{ele_name}\{ele_name}.mtl",ele_path)
        
    # test
    # geometries = []
    # temp_path = osp.join(scenario_base,scene_name + "_" + body_name,"temp")
    # if scene_name in self_define_scenes:
    #     scene_path =  osp.join(temp_path,"env.obj") 
    # else:
    #     scene_path = osp.join(temp_path,f"{scene_name}.ply") 
    # scene_mesh = trimesh.load(scene_path,force="mesh")
    # body_path = osp.join(temp_path,f"{body_name}.obj")
    # body_mesh = trimesh.load(body_path,force="mesh")
    # geometries.append(scene_mesh)
    # geometries.append(body_mesh)
    
    # if tag == "ours":
    #     eles_path_base = osp.join(temp_path,"virtual_elements")
    # else:
    #     eles_path_base = osp.join(temp_path,"virtual_elements_baseline")
    
    # for ele in os.listdir(eles_path_base):
    #     ele_path = osp.join(eles_path_base,ele,ele+'.obj')
    #     ele_mesh = trimesh.load(ele_path,force="mesh")
    #     geometries.append(ele_mesh)
    # scene = trimesh.Scene()
    # scene.add_geometry(geometries)
    # scene.show()       

def main():
    for i in range(4,len(scene_body)):
        scenario_id = scene_body[i][0]
        scene_name = scene_body[i][1]
        body_name = scene_body[i][2]
        # transform_obj(scenario_id,scene_name,body_name,"baseline")
        merge_one(scenario_id,scene_name,body_name,"ours",True)
        # merge_one(scenario_id,scene_name,body_name,"baseline",True)
        # merge_one(scenario_id,scene_name,body_name,"ours",False)
        # merge_one(scenario_id,scene_name,body_name,"baseline",False)
        print(f"scenario {scenario_id} 's {scene_name} is processing !")

def export():
    #把需要参与评估的结果放入一个文件夹中
    export_base = r"D:\data\POSA_dir\results\evaluation"
    for i in range(1,5):
        scenario_id = str(i)
        scenario_path = rf"D:\data\POSA_dir\results\scenario_{scenario_id}"
        e_scenario_path = osp.join(export_base,f"scenario_{scenario_id}")
        if not osp.exists(e_scenario_path):
            os.makedirs(e_scenario_path)
        for file_name in os.listdir(scenario_path):
            file_base_ours = osp.join(scenario_path,file_name,"ours")
            file_base_baseline = osp.join(scenario_path,file_name,"baseline")
            file_base_temp = osp.join(scenario_path,file_name,"temp")
            
            shutil.copytree(file_base_temp, osp.join(e_scenario_path,file_name,"temp"))
            shutil.copy(osp.join(scenario_path,file_name,"cam_0.json"),osp.join(e_scenario_path,file_name))
            
            e_file_base_ours = osp.join(e_scenario_path,file_name,"ours")
            if not osp.exists(e_file_base_ours):
                os.makedirs(e_file_base_ours)
            e_file_base_baseline = osp.join(e_scenario_path,file_name,"baseline")
            if not osp.exists(e_file_base_baseline):
                os.makedirs(e_file_base_baseline)
    
            pdb.set_trace()
            for file_name in os.listdir(file_base_ours):
                file_name_without_ext = os.path.splitext(file_name)[0]
                if file_name_without_ext.endswith('pure'):
                    file_path = os.path.join(file_base_ours, file_name)
                    shutil.copy(file_path, e_file_base_ours)
                    print(f"Copied: {file_name} to {e_file_base_ours}")
                
            for file_name in os.listdir(file_base_baseline):
                file_name_without_ext = os.path.splitext(file_name)[0]
                if file_name_without_ext.endswith('pure'):
                    file_path = os.path.join(file_base_baseline, file_name)
                    shutil.copy(file_path, e_file_base_baseline)
                    print(f"Copied: {file_name} to {e_file_base_baseline}")

def create_image_grid():
    """
    将图像按网格布局拼接成拼图并保存

    :param image_paths: 图片路径列表
    :param grid_size: 网格的行数和列数 (rows, cols)
    :param save_path: 拼图保存路径
    """
    img_base = r"D:\data\POSA_dir\results\images"
    store_base = r"D:\data\POSA_dir\results\merged_imgs"
    if not osp.exists(store_base):
        os.makedirs(store_base)
    for file_name in os.listdir(img_base):
        imgs_path = osp.join(img_base,file_name)
        merge_path = osp.join(store_base,file_name+".png")
        img_width = 1920 * 4 + 60
        img_height = 1080
        grid_image = Image.new('RGBA', (img_width, img_height),(0,0,0,0))
        img_base_0 = Image.open(osp.join(imgs_path,"cam_0_baseline_pure0001.png"))
        grid_image.paste(img_base_0, (0, 0))
        img_base_3 = Image.open(osp.join(imgs_path,"cam_3_baseline0001.png"))
        grid_image.paste(img_base_3, (1920, 0))
        img_ours_0 = Image.open(osp.join(imgs_path,"cam_0_ours_pure0001.png"))
        grid_image.paste(img_ours_0, (1920 * 2 + 60 , 0))
        img_ours_3 = Image.open(osp.join(imgs_path,"cam_3_ours0001.png"))
        grid_image.paste(img_ours_3, (1920 * 3 + 60, 0))
        grid_image.save(merge_path)
        
        print(f"拼图保存为: {merge_path}")

def exp_images():
    #把需要参与评估的结果放入一个文件夹中
    det_base = r"D:\data\POSA_dir\results\images"
    if not osp.exists(det_base):
        os.makedirs(det_base)
    for i in range(1,5):
        scenario_id = str(i)
        scenario_path = rf"D:\data\POSA_dir\results\scenario_{scenario_id}"
        # d_scenario_path = osp.join(export_base,f"scenario_{scenario_id}")
        # if not osp.exists(d_scenario_path):
        #     os.makedirs(e_scenario_path)
        for file_name in os.listdir(scenario_path):
            
            file_base_temp = osp.join(scenario_path,file_name,"temp","images")
            det_path = osp.join(det_base,file_name)
            shutil.copytree(file_base_temp, det_path)
            
def exp_sdf():
    return 0
             
if __name__ == "__main__":
    # merge_one("1","S1_E1","rp_ethan_posed_012_0_0_00_00")
    # test_json("2","S2_E2","rp_carla_posed_002_0_0_00_02")
    # transform_obj("1","S1_E1","rp_ethan_posed_012_0_0_00_00")
    # main()
    # export()
    # exp_images()
    create_image_grid()
    
  

 
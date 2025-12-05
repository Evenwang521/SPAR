import numpy as np
import json
import open3d as o3d
import os
import pdb
import re

category_mapping = {
    "floor": 2,
    "chair": 3,
    "table": 5,
    "sofa": 10,
    "seating": 34,
    "cabinet": 7,
    "clothrack": 36,
    "computer": 39,
    "printer": 39,
    "monitor": 39,
    "box": 39,
    "shredder": 39,
    "whiteboard": 39,
    "server": 39,
    "cpu":39,
    "printer_hp": 39,
    "printer_canon": 39,
    "sofa_d": 10,
    "sofa_s": 10,
    "loveseat":10,
    "dispenser":39
    }

def load_matrices_from_npy(file_path):
    # Load the dictionary from the .npy file
    data = np.load(file_path, allow_pickle=True).item()
    return data

def get_mesh_path_from_name(obj_name, base_path):
    # Extract the letter prefix from the object name
    prefix = ''.join([char for char in obj_name if (char.isalpha() or char == '_')])
    # Generate the mesh path based on the prefix
    mesh_path = os.path.join(base_path, f"{prefix}",f"{prefix}.obj")
    return mesh_path

def compute_aabb_from_mesh(mesh_path, transform_matrix):
    # Load the mesh using Open3D
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.rotate(transform_matrix[:3, :3],center=(0,0,0))
    mesh.translate(transform_matrix[:3, 3])
    # Compute the axis-aligned bounding box (AABB)
    aabb = mesh.get_axis_aligned_bounding_box()
    return aabb

def save_obb_to_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

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

def main(scene_name):
    npy_file = f"D:\data\ICAR_SE\{scene_name}\objs_transform.npy"
    base_mesh_path = f"D:\data\ICAR_SE\{scene_name}"
    output_json = f"D:\code\python\POSA\output\scenes_ins_aabb\{scene_name}.json"
    
    # Load the transformation matrices and object names
    data = load_matrices_from_npy(npy_file)
    
    # List to store OBB data
    obb_list = []
    pdb.set_trace()
    for instance_id, (obj_name, transform_matrix) in enumerate(data.items()):
        # Generate mesh path from object name
        mesh_path = get_mesh_path_from_name(obj_name, base_mesh_path)
        
        # Check if mesh file exists
        if not os.path.exists(mesh_path):
            print(f"Mesh file not found: {mesh_path}")
            continue
        
        # Compute the OBB from the mesh and transformation matrix
        aabb = compute_aabb_from_mesh(mesh_path, transform_matrix)
        

        # Store the OBB min and max coordinates
        obb_entry = {
            "instance_id": instance_id,
            "category": category_mapping[re.sub(r'\d', '', obj_name)],  # Assuming the category is encoded in the first letter of the object name
            "min_bound": aabb.min_bound.tolist(),
            "max_bound": aabb.max_bound.tolist()
        }
        obb_list.append(obb_entry)
    
    # Save the OBB data to a JSON file
    save_obb_to_json(obb_list, output_json)
    print(f"OBB data saved to {output_json}")

def viz(scene_name):
    
    geometries = []
    scene_mesh = o3d.io.read_triangle_mesh(f"D:/data/ICAR_SE/{scene_name}/env.obj")
    scene_aabb = f"D:\code\python\POSA\output\scenes_ins_aabb\{scene_name}.json"
    geometries.append(scene_mesh)
    
    with open(scene_aabb,'r') as f:
        ins_aabb = json.load(f)
    # ins_aabb = json.loads(data).items()
    
    for ins in ins_aabb: 
        # id = ins['instance_id']
        min_bound = ins["min_bound"]
        max_bound = ins["max_bound"]
        vertices,lines = get_lines_obb(np.array([min_bound,max_bound]))
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(vertices)
        lineset.lines = o3d.utility.Vector2iVector(lines) 
        line_color = np.array([255,0,0])/255.0
        lineset.colors = o3d.utility.Vector3dVector(np.array([line_color for _ in range(len(lines))]))
        geometries.append(lineset)
    o3d.visualization.draw_geometries(geometries)
     
if __name__ == "__main__":
    # pdb.set_trace()
    # main("S1_E2")
    # viz("S1_E2")
    # main("S2_E1")
    # viz("S2_E1")
    main("S2_E2")
    viz("S2_E2")

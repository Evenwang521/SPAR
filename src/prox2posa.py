"""
Transformation between PROX and POSA scene assets.
"""

import pickle
import numpy as np
import trimesh
import os
scene_folder = "D:\data\PROXE_ExpansionPack\scenes_semantics" # PROX COINS 
proxe_folder ="D:\data\POSA_dir\scenes"
scene_registration_file = "D:\code\python\POSA\\registration.pkl"
scene_names = ["BasementSittingBooth", "MPH11", "MPH112", "MPH16", "MPH1Library", "MPH8",
               "N0SittingBooth", "N0Sofa", "N3Library", "N3Office", "N3OpenArea", "Werkraum"]

def scene_registration(scene_name):
    PROX_scene = trimesh.load_mesh(os.path.join(scene_folder, scene_name + '.ply'))
    POSA_scene = trimesh.load_mesh(os.path.join(proxe_folder, scene_name + '.ply'))
    transform, cost = trimesh.registration.mesh_other(POSA_scene, PROX_scene)
    return transform

def prox_to_posa(scene_name, points):
    transform = np.linalg.inv(POSA_to_PROX_transform[scene_name])
    return np.dot(transform[:3, :3], points.T).T + transform[:3, 3].reshape((1, 3))

if not os.path.exists(scene_registration_file):
    POSA_to_PROX_transform = {}
    for scene_name in scene_names:
        print("registration of " + scene_name +" is computing")
        POSA_to_PROX_transform[scene_name] = scene_registration(scene_name)
    with open(scene_registration_file, 'wb') as file:
        pickle.dump(POSA_to_PROX_transform, file)
    print("registration is saving!")
    
with open(scene_registration_file, 'rb') as file:
    POSA_to_PROX_transform = pickle.load(file)
    print("registration is loading!")

if __name__ == '__main__':
    
    for scene_name in scene_names:
        print(scene_name + "is showing!")
        PROX_scene = trimesh.load_mesh(os.path.join(scene_folder, scene_name + '.ply'))
        num_vertex = len(PROX_scene.vertices)
        PROX_scene.visual.vertex_colors = np.array([[255, 0, 0, 255]]*num_vertex, dtype=np.uint8)
        
        POSA_scene = trimesh.load_mesh(os.path.join(proxe_folder, scene_name + '.ply'))
        POSA_scene.visual.vertex_colors = np.array([[0, 255, 0, 255]]*num_vertex, dtype=np.uint8)
        (PROX_scene + POSA_scene.apply_transform(POSA_to_PROX_transform[scene_name])).show()
        
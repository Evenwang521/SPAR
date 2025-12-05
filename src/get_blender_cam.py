import os
import os.path as osp
import bpy
import numpy as np
import json


export_base = "D:/data/POSA_dir/results"

scene_name = 'Werkraum'
body_name = 'rp_ethan_posed_012_0_0_00_06'
scenario = 'scenario_2'
json_path = osp.join(export_base,scenario,scene_name+'_'+body_name,'cam_3.json')

camera = bpy.context.scene.camera

if camera is not None and camera.type == 'CAMERA':
    cam_trans = np.array(camera.matrix_world.copy())
    cam_fov = camera.data.angle
    cam_param = {"cur_model":cam_trans.tolist(),"fov":cam_fov / np.pi * 180}
    
    with open(json_path,'w') as f :
        json.dump(cam_param,f)
    
    
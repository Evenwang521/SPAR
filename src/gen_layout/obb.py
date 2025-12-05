import numpy as np
import torch
import torch.optim as Optim
import open3d as o3d
from tqdm import tqdm
import pdb


EDGES = (
    [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
    [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
    [1, 2], [3, 4], [5, 6], [7, 8]   # lines along z-axis
)

UNIT_BOX = np.array([
    [0., 0., 0.],
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5],
])

class OBB():
    def __init__(self,scale,rot_mat,t,vertices):
        
        self._scale = scale       # tensor(1,3)
        self._rotation = rot_mat  # tensor(3,3)
        self._translation = t     # tensor(3)
        self._vertices = vertices # tensor(n,3)
        # 为了梯度等各种问题，定义时强制赋值，避免计算
         
def sample_points(box,num_samples):
    '''在box中随机采样num_points个点

    Parameters:
    box (OBB)       : 包围盒
    num_samples (int): 采样点数

    Returns:
    tensor (num_samples,3): 采样点
    '''
    point = (torch.rand(num_samples,3)-0.5) 
    point = point * box._scale.detach()
    
    point = torch.matmul(box._rotation, point.t()).t() + box._translation
    
    return point

def inside_pts(box,pts):
    """计算pts中所有的点是否在box中

    Args:
        box (OBB): 包围盒
        pts (tensor(n,3)): 采样点
        
    Returns:
        tensor(1) 在包围盒中的点数
    """
    # pts_w = torch.matmul((box._rotation).t(),(pts-box._translation).t()).t() + box._translation
    pts_w = torch.matmul((box._rotation).t(),(pts-box._translation).t()).t() 
  
    inside_mask_x = torch.where(torch.abs(pts_w[:,0]) < box._scale[0]/2.0,1.0,0.0)
    inside_mask_y = torch.where(torch.abs(pts_w[:,1]) < box._scale[1]/2.0,1.0,0.0)
    inside_mask_z = torch.where(torch.abs(pts_w[:,2]) < box._scale[2]/2.0,1.0,0.0)            
    inside_mask = inside_mask_x * inside_mask_y * inside_mask_z
     
    return torch.sum(inside_mask)

def iou_sample_estimate(box1,box2,num_samples):
    '''估计box1 和 box2 的相交程度

    Args:
        box1 (OBB): 
        box2 (OBB): 
        scale_boxes (n,3) :Boxes的scale
        num_samples (int): 采样点数
        
    Returns:
        两个包围盒的相交比例
    '''
    scale_boxes = torch.cat((box1._scale.detach(),box2._scale.detach()),dim=0)
    sample_pts_box1 = sample_points(box1,num_samples)
    sample_pts_box2 = sample_points(box2,num_samples)
    num_inside_box1 = inside_pts(box1,sample_pts_box2)
    num_inside_box2 = inside_pts(box2,sample_pts_box1)
    volume1 = scale_boxes[0][0] * scale_boxes[0][1] * scale_boxes[0][2]
    volume2 = scale_boxes[1][0] * scale_boxes[1][1] * scale_boxes[1][2]
    intersection_estimate  = (volume1 * num_inside_box1 + volume2 * num_inside_box2 ) /2.0
    union_estimate = volume1 * num_samples + volume2 * num_samples - intersection_estimate
    
    return intersection_estimate / union_estimate

def iou_sdf_estimate(box1,box2,num_samples):
    '''计算box2采样点距离box1的SDF值之和

    Args:
        box1 (OBB): SDF包围盒
        box2 (OBB): 采样包围盒
        num_samples (int): 采样点数
        
    Returns:
        iou_sdf_estimate (_tensor(1,)_) : SDF估计IoU
    '''
    scale_boxes = torch.cat((box1._scale.detach(),box2._scale.detach()),dim=0)
    
    sample_pts_box2 = sample_points(box2,num_samples)
    sample_pts_box22box1 = torch.matmul(box1._rotation.t(),(sample_pts_box2 - box1._translation).t()).t() 
    dist_min_x_1 = -scale_boxes[0][0] - sample_pts_box22box1[:,0]
    dist_max_x_1 = sample_pts_box22box1[:,0] - scale_boxes[0][0]
    dist_min_y_1 = -scale_boxes[0][1] - sample_pts_box22box1[:,1]
    dist_max_y_1 = sample_pts_box22box1[:,1] - scale_boxes[0][1]
    dist_min_z_1 = -scale_boxes[0][2] - sample_pts_box22box1[:,2]
    dist_max_z_1 = sample_pts_box22box1[:,2] - scale_boxes[0][2]
    dist_1 = torch.stack((dist_min_x_1,dist_max_x_1,dist_min_y_1,dist_max_y_1,dist_min_z_1,dist_max_z_1),dim=1) #tensor(n,6)
    max_dist_1,_= torch.max(dist_1,dim=1)
    sdf_1 = torch.where(max_dist_1 < .0,max_dist_1,torch.tensor(.0,dtype=torch.float32))
    
    sample_pts_box1 = sample_points(box1,num_samples)
    sample_pts_box12box2 = torch.matmul(box2._rotation.t(),(sample_pts_box1 - box2._translation).t()).t() 
    dist_min_x_2 = -scale_boxes[1][0] - sample_pts_box12box2[:,0]
    dist_max_x_2 = sample_pts_box12box2[:,0] - scale_boxes[1][0]
    dist_min_y_2 = -scale_boxes[1][1] - sample_pts_box12box2[:,1]
    dist_max_y_2 = sample_pts_box12box2[:,1] - scale_boxes[1][1]
    dist_min_z_2 = -scale_boxes[1][2] - sample_pts_box12box2[:,2]
    dist_max_z_2 = sample_pts_box12box2[:,2] - scale_boxes[1][2]
    dist_2 = torch.stack((dist_min_x_2,dist_max_x_2,dist_min_y_2,dist_max_y_2,dist_min_z_2,dist_max_z_2),dim=1) #tensor(n,6)
    max_dist_2,_= torch.max(dist_2,dim=1)
    sdf_2 = torch.where(max_dist_2 < .0,max_dist_2,torch.tensor(.0,dtype=torch.float32))
    
    return torch.max(torch.sum(torch.abs(sdf_1))/num_samples,torch.sum(torch.abs(sdf_2))/num_samples)

def iou_convexhull_estimate(box1,box2):
    '''通过计算交点和凸包的方式计算IoU

    Args:
        box1 (_Box_): 
        box2 (_Box_): 
    '''
    return 

def viz(box1,box2):
    '''可视化两个box

    Args:
        box1 (OBB): 
        box2 (OBB): 
    '''
    
    vertices1 = box1._vertices.detach().numpy()
    vertices2 = box2._vertices.detach().numpy()
    
    lineset1 = o3d.geometry.LineSet()
    lineset1.points = o3d.utility.Vector3dVector(vertices1)
    lineset1.lines = o3d.utility.Vector2iVector(EDGES) 
    line_color1 = np.array([255,0,0])/255.0
    lineset1.colors = o3d.utility.Vector3dVector(np.array([line_color1 for _ in range(len(EDGES))]))
    
    lineset2 = o3d.geometry.LineSet()
    lineset2.points = o3d.utility.Vector3dVector(vertices2)
    lineset2.lines = o3d.utility.Vector2iVector(EDGES) 
    line_color2 = np.array([0,0,255])/255.0
    lineset2.colors = o3d.utility.Vector3dVector(np.array([line_color2 for _ in range(len(EDGES))]))
    
    o3d.visualization.draw_geometries([lineset1,lineset2], mesh_show_wireframe=False,mesh_show_back_face=True)
    
def test():
    scale = torch.tensor([[1.,1.,1.]],dtype=torch.float32)
    rot_mat = torch.eye(3)
    t = torch.zeros(3)
    vertices = torch.tensor(UNIT_BOX,dtype=torch.float32)
    box1 = OBB(scale,rot_mat,t,vertices)
    
    angle = 45.0 / 180.0 * np.pi
    axis = np.array([1.0,2.0,3.0]) 
    axis_norm = torch.tensor(axis/np.linalg.norm(axis))
    rot_mat_n = torch.tensor(o3d.geometry.get_rotation_matrix_from_axis_angle(axis_norm * angle),dtype=torch.float32)
    t_n = torch.tensor([0.1,0.2,0.3],dtype=torch.float32)
    vertices_n = torch.matmul(rot_mat_n,torch.tensor(UNIT_BOX,dtype=torch.float32).t()).t() + t_n
    box2 = OBB(scale,rot_mat_n,t_n,vertices_n)

    print(iou_estimate(box1,box2,10000))

def opt():
    scale = torch.tensor([[1.,1.,1.]],dtype=torch.float32)
    rot_mat = torch.eye(3)
    t = torch.zeros(3)
    vertices = torch.tensor(UNIT_BOX,dtype=torch.float32)
    box1 = OBB(scale,rot_mat,t,vertices)
    
    angle = 45.0 / 180.0 * np.pi
    axis = np.array([1.0,2.0,3.0]) 
    axis_norm = torch.tensor(axis/np.linalg.norm(axis))
    rot_mat_n = torch.tensor(o3d.geometry.get_rotation_matrix_from_axis_angle(axis_norm * angle),dtype=torch.float32)
    t_n = torch.tensor([0.1,0.2,0.3],dtype=torch.float32)
    scale2 = torch.tensor([[0.3,0.1,0.3]],dtype=torch.float32)
    vertices_n = torch.matmul(rot_mat_n,(torch.tensor(UNIT_BOX,dtype=torch.float32)*scale2.squeeze()).t()).t() + t_n
    box2 = OBB(scale2,rot_mat_n,t_n,vertices_n)
    
    viz(box1,box2)
    #计算scale张量

    # scale_boxes = torch.tensor([[1.,1.,1.],[0.3,0.1,0.3]],dtype=torch.float32)
    
    opt_pos = torch.tensor([0.1000, 0.2000, 0.3000],dtype=torch.float32, requires_grad=True)
    opt_param = [opt_pos]
        
    # 构建优化器
    optimizer = Optim.Adam(opt_param, lr=0.05)
    lr_schedule = Optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    #迭代求解
    # 开始优化
    num_iter = 50  
    pdb.set_trace()
    for epoch in tqdm(range(num_iter)):       
        box2._translation = opt_pos
        # loss = iou_sdf_estimate(box1,box2,10000)
        loss = iou_sdf_estimate(box2,box1,10000)
        loss.backward()
        lr_schedule.step()
        optimizer.step()
        optimizer.zero_grad()
        print("loss of epoch {} is {}".format(epoch,loss))
    pdb.set_trace()
    #可视化
    box2._vertices = torch.matmul(box2._rotation,(torch.tensor(UNIT_BOX,dtype=torch.float32)  * box2._scale.squeeze()).t()).t() + box2._translation
    viz(box1,box2)
   
if __name__ == "__main__":
    pdb.set_trace()
    # test()
    opt()
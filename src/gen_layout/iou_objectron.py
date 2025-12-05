"""The Intersection Over Union (IoU) for 3D oriented bounding boxes."""

import numpy as np
import scipy.spatial as sp
from scipy.spatial.transform import Rotation as rotation_util
import pdb
import torchgeometry as tgm
import torch
import open3d as o3d
import torch.optim as Optim
from tqdm import tqdm
import convexhull
 
_PLANE_THICKNESS_EPSILON = 0.000001
_POINT_IN_FRONT_OF_PLANE = 1
_POINT_ON_PLANE = 0
_POINT_BEHIND_PLANE = -1

EDGES = (
    [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
    [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
    [1, 2], [3, 4], [5, 6], [7, 8]   # lines along z-axis
)

# The vertices are ordered according to the left-hand rule, so the normal
# vector of each face will point inward the box.
FACES = np.array([
    [5, 6, 8, 7],  # +x on yz plane
    [1, 3, 4, 2],  # -x on yz plane
    [3, 7, 8, 4],  # +y on xz plane = top
    [1, 2, 6, 5],  # -y on xz plane
    [2, 4, 8, 6],  # +z on xy plane = front
    [1, 5, 7, 3],  # -z on xy plane
])

UNIT_BOX = np.asarray([
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

NUM_KEYPOINTS = 9
FRONT_FACE_ID = 4
TOP_FACE_ID = 2

class Box(object):
  """General 3D Oriented Bounding Box."""

  def __init__(self, vertices=None):
    if vertices is None:
      vertices = self.scaled_axis_aligned_vertices(np.array([1., 1., 1.]))
    self._vertices = vertices
    self._rotation = torch.eye(3)
    self._translation = torch.zeros((3))
    self._scale = self.get_scale() #默认情况下不进行scale
    self._transformation = torch.eye(4)
    self._volume = self.get_volume()
  
  def scaled_axis_aligned_vertices(self, scale):
    """Returns an axis-aligned set of verticies for a box of the given scale.

    Args:
      scale: A 3*1 vector, specifiying the size of the box in x-y-z dimension.
    """
    w = scale[0] / 2.
    h = scale[1] / 2.
    d = scale[2] / 2.

    # Define the local coordinate system, w.r.t. the center of the box
    aabb = torch.tensor([[0., 0., 0.],
                         [-w, -h, -d], 
                         [-w, -h, d], 
                         [-w, h, -d],
                         [-w, h, d], 
                         [w, -h, -d], 
                         [w, -h, d], 
                         [w, h, -d],
                         [w,h, d]],dtype=torch.float32)
    aabb.requires_grad_(True)
    return aabb
    
  def get_scale(self):
    # x = self._vertices[5, :] - self._vertices[1, :]
    # y = self._vertices[3, :] - self._vertices[1, :]
    # z = self._vertices[2, :] - self._vertices[1, :]
    max_pt ,_ = torch.max(self._vertices,dim=0)
    min_pt ,_ = torch.min(self._vertices,dim=0)
    scale = max_pt - min_pt
    # scale[0] = torch.norm(x)
    # scale[1] = torch.norm(y)
    # scale[2] = torch.norm(z)
    
    return scale.reshape(3,1).detach()
  
  def get_volume(self):
    return self._scale[0] * self._scale[1] * self._scale[2]
  
  def apply_tranform(self,transform):
    '''
    change the translation of itself
    '''
    self._transformation = transform
    self._rotation = transform[:3,:3]
    self._translation = transform[:3,3]
    self._vertices = torch.matmul(self._rotation,self.vertices.t()).t() + self._translation
   
  @classmethod
  def from_transformation(cls, rotation, translation, scale):
    """Constructs an oriented bounding box from transformation and scale."""
    if rotation.shape != 3 and rotation.shape != (3,3):
      raise ValueError('Unsupported rotation, only 3x1 euler angles or 3x3 ' +
                       'rotation matrices are supported. ')
    if rotation.shape == 3:
      rotation = rotation_util.from_rotvec(rotation.tolist()).as_dcm()
    scaled_identity_box = cls.scaled_axis_aligned_vertices(cls,scale)
    vertices = torch.matmul(rotation,scaled_identity_box.t()).t() + translation

    return cls(vertices=vertices)
  
  def apply_transformation(self, transformation):
    """Applies transformation on the box.

    Group multiplication is the same as rotation concatenation. Therefore return
    new box with SE3(R * R2, T + R * T2); Where R2 and T2 are existing rotation
    and translation. Note we do not change the scale.

    Args:
      transformation: a 4x4 transformation matrix.

    Returns:
       transformed box.
    """
    if transformation.shape != (4, 4):
      raise ValueError('Transformation should be a 4x4 matrix.')
    
    new_rotation = torch.matmul(transformation[:3, :3], self.rotation)
    new_translation = transformation[:3, 3] + (
        torch.matmul(transformation[:3, :3], self.translation))
    return Box.from_transformation(new_rotation, new_translation, self.scale)

  def __repr__(self):
    representation = 'Box: '
    for i in range(NUM_KEYPOINTS):
      representation += '[{0}: {1}, {2}, {3}]'.format(i, self.vertices[i, 0],
                                                      self.vertices[i, 1],
                                                      self.vertices[i, 2])
    return representation

  def __len__(self):
    return NUM_KEYPOINTS

  def __name__(self):
    return 'Box'

  def inside(self, point):
    """Tests whether a given point is inside the box.

      Brings the 3D point into the local coordinate of the box. In the local
      coordinate, the looks like an axis-aligned bounding box. Next checks if
      the box contains the point.
    Args:
      point: A 3*1 numpy vector.

    Returns:
      True if the point is inside the box, False otherwise.
    """
    inv_trans = torch.inverse(self.transformation)
    scale = self.scale
    point_w = torch.matmul(inv_trans[:3, :3], point) + inv_trans[:3, 3]
    for i in range(3):
      if abs(point_w[i]) > scale[i] / 2.0:
        return False
    return True 
     
  def num_inside_pts(self, points):
    """Tests whether a given point is inside the box.

      Brings the 3D point into the local coordinate of the box. In the local
      coordinate, the looks like an axis-aligned bounding box. Next checks if
      the box contains the point.
    Args:
      point: A 3*1 numpy vector.

    Returns:
      True if the point is inside the box, False otherwise.
    """
    """修改成判断一堆张量的形式

    Returns:
        _type_: _description_
    """
    inv_trans = torch.inverse(self.transformation)
    scale = self.scale.reshape(1,3)
    points_w = torch.matmul(inv_trans[:3, :3], points.t()).t() + inv_trans[:3, 3]
    # for i in range(3):
    #   if abs(point_w[i]) > scale[i] / 2.0:
    #     return False
    # return True 
    
    # return torch.sum(torch.le(torch.abs(points_w),(scale) / 2.0)))
    return torch.sum(torch.le(torch.abs(points_w),(torch.tensor([1.,1.0,1.],dtype=torch.float32) / 2.0)))
                
    # inside_mask = inside_mask.float()
    # inside_mask.retain_grad()
    # 统计不在立方体内的点数
    # count_outside = torch.sum(inside_mask)
    # return count_outside

  def sample(self,num_samples=10000):
    """Samples a 3D point uniformly inside this box."""
    point = (torch.rand(num_samples,3)-0.5) 
    point[:,0] = point[:,0] * self.scale[0]
    point[:,1] = point[:,1] * self.scale[1]
    point[:,2] = point[:,2] * self.scale[2]
    
    point = torch.matmul(self.rotation, point.t()).t() + self.translation
    return point

  @property
  def vertices(self):
    return self._vertices

  @property
  def rotation(self):
    if self._rotation is None:
      self._rotation, self._translation, self._scale = self.fit(self._vertices)
    return self._rotation

  @property
  def translation(self):
    if self._translation is None:
      self._rotation, self._translation, self._scale = self.fit(self._vertices)
    return self._translation

  @property
  def scale(self):
    if self._scale is None:
      self._rotation, self._translation, self._scale = self.fit(self._vertices)
    return self._scale

  @property
  def volume(self):
    """Compute the volume of the parallelpiped or the box.

      For the boxes, this is equivalent to np.prod(self.scale). However for
      parallelpiped, this is more involved. Viewing the box as a linear function
      we can estimate the volume using a determinant. This is equivalent to
      sp.ConvexHull(self._vertices).volume

    Returns:
      volume (float)
    """
    if self._volume is None:
      i = self._vertices[2, :] - self._vertices[1, :]
      j = self._vertices[3, :] - self._vertices[1, :]
      k = self._vertices[5, :] - self._vertices[1, :]
      sys = np.array([i, j, k])
      self._volume = abs(np.linalg.det(sys))
    return self._volume

  @property
  def transformation(self):
    if self._rotation is None:
      self._rotation, self._translation, self._scale = self.fit(self._vertices)
    if self._transformation is None:
      self._transformation = np.identity(4)
      self._transformation[:3, :3] = self._rotation
      self._transformation[:3, 3] = self._translation
    return self._transformation

  def get_ground_plane(self, gravity_axis=1):
    """Get ground plane under the box."""

    gravity = np.zeros(3)
    gravity[gravity_axis] = 1

    def get_face_normal(face, center):
      """Get a normal vector to the given face of the box."""
      v1 = self.vertices[face[0], :] - center
      v2 = self.vertices[face[1], :] - center
      normal = np.cross(v1, v2)
      return normal

    def get_face_center(face):
      """Get the center point of the face of the box."""
      center = np.zeros(3)
      for vertex in face:
        center += self.vertices[vertex, :]
      center /= len(face)
      return center

    ground_plane_id = 0
    ground_plane_error = 10.

    # The ground plane is defined as a plane aligned with gravity.
    # gravity is the (0, 1, 0) vector in the world coordinate system.
    for i in [0, 2, 4]:
      face = FACES[i, :]
      center = get_face_center(face)
      normal = get_face_normal(face, center)
      w = np.cross(gravity, normal)
      w_sq_norm = np.linalg.norm(w)
      if w_sq_norm < ground_plane_error:
        ground_plane_error = w_sq_norm
        ground_plane_id = i

    face = FACES[ground_plane_id, :]
    center = get_face_center(face)
    normal = get_face_normal(face, center)

    # For each face, we also have a parallel face that it's normal is also
    # aligned with gravity vector. We pick the face with lower height (y-value).
    # The parallel to face 0 is 1, face 2 is 3, and face 4 is 5.
    parallel_face_id = ground_plane_id + 1
    parallel_face = FACES[parallel_face_id]
    parallel_face_center = get_face_center(parallel_face)
    parallel_face_normal = get_face_normal(parallel_face, parallel_face_center)
    if parallel_face_center[gravity_axis] < center[gravity_axis]:
      center = parallel_face_center
      normal = parallel_face_normal
    return center, normal
    
class IoU(object):
  """General Intersection Over Union cost for Oriented 3D bounding boxes."""

  def __init__(self, box1, box2):
    self._box1 = box1
    self._box2 = box2
    self._intersection_points = None
    # self._convexhull = None

  def get_intersection_points(self):
    self._compute_intersection_points(self._box1, self._box2)
    self._compute_intersection_points(self._box2, self._box1)
    return self._intersection_points
    
  def iou(self):
    """Computes the exact IoU using Sutherland-Hodgman algorithm."""
    self._compute_intersection_points(self._box1, self._box2)
    self._compute_intersection_points(self._box2, self._box1)
    if self._intersection_points != None:
      # pdb.set_trace()
      # m_convexhull = ConvexHull(self._intersection_points.detach())
      # m_convexhull.create()
      # intersection_volume = m_convexhull.volume()
      # print("volume of my_own_hull is {}".format(intersection_volume.item()))
      intersection_volume = sp.ConvexHull(self._intersection_points.detach().numpy()).volume
      # print("volume of sp.spatial.convexhull is {}".format(intersection_volume))
      box1_volume = self._box1.volume
      box2_volume = self._box2.volume
      union_volume = box1_volume + box2_volume - intersection_volume
      return intersection_volume / union_volume
    else:
      return 0.

  def iou_sampling(self, num_samples=10000):
    """Computes intersection over union by sampling points.

    Generate n samples inside each box and check if those samples are inside
    the other box. Each box has a different volume, therefore the number of
    samples in box1 is estimating a different volume than box2. To address
    this issue, we normalize the iou estimation based on the ratio of the
    volume of the two boxes.

    Args:
      num_samples: Number of generated samples in each box

    Returns:
      IoU Estimate (float)
    """
    p1 = self._box1.sample(num_samples) 
    p2 = self._box2.sample(num_samples) 
    box1_volume = self._box1.volume
    box2_volume = self._box2.volume
    box1_intersection_estimate = self._box1.num_inside_pts(p2)
    box2_intersection_estimate = self._box2.num_inside_pts(p1)
  
    intersection_volume_estimate = (
        box1_volume * box1_intersection_estimate +
        box2_volume * box2_intersection_estimate) / 2.0
    union_volume_estimate = (box1_volume * num_samples + box2_volume *
                             num_samples) - intersection_volume_estimate
    iou_estimate = intersection_volume_estimate / union_volume_estimate
    return iou_estimate

  def _compute_intersection_points(self, box_src, box_template):
    """Computes the intersection of two boxes."""
    # Transform the source box to be axis-aligned
    inter_pts = []
    inv_transform = torch.inverse(box_src.transformation)
    box_src_axis_aligned = box_src.apply_transformation(inv_transform)
    template_in_src_coord = box_template.apply_transformation(inv_transform)
    for face in range(len(FACES)):
      indices = FACES[face, :]
      poly = [template_in_src_coord.vertices[indices[i], :] for i in range(4)]
      clip = self.intersect_box_poly(box_src_axis_aligned, poly)
      for point in clip:
        # Transform the intersection point back to the world coordinate
        point_w = torch.matmul(box_src.rotation, point) + box_src.translation
        inter_pts.append(point_w)

    for point_id in range(NUM_KEYPOINTS):
      v = template_in_src_coord.vertices[point_id, :]
      if box_src_axis_aligned.inside(v):
        point_w = torch.matmul(box_src.rotation, v) + box_src.translation
        inter_pts.append(point_w)
        
    inter_pts = torch.stack(inter_pts)
    
    if self._intersection_points == None:
      self._intersection_points = inter_pts 
    else:
      self._intersection_points = torch.cat((self._intersection_points,inter_pts),dim=0)   
    # viz_interaction(box_src_axis_aligned,template_in_src_coord,inter_pts.numpy())
    
  def intersect_box_poly(self, box, poly):
    """Clips the polygon against the faces of the axis-aligned box."""
    for axis in range(3):
      poly = self._clip_poly(poly, box.vertices[1, :], 1.0, axis)
      poly = self._clip_poly(poly, box.vertices[8, :], -1.0, axis)
    return poly

  def _clip_poly(self, poly, plane, normal, axis):
    """Clips the polygon with the plane using the Sutherland-Hodgman algorithm.

    See en.wikipedia.org/wiki/Sutherland-Hodgman_algorithm for the overview of
    the Sutherland-Hodgman algorithm. Here we adopted a robust implementation
    from "Real-Time Collision Detection", by Christer Ericson, page 370.

    Args:
      poly: List of 3D vertices defining the polygon.
      plane: The 3D vertices of the (2D) axis-aligned plane.
      normal: normal
      axis: A tuple defining a 2D axis.

    Returns:
      List of 3D vertices of the clipped polygon.
    """
    # The vertices of the clipped polygon are stored in the result list.
    result = []
    if len(poly) <= 1:
      return result

    # polygon is fully located on clipping plane
    poly_in_plane = True

    # Test all the edges in the polygon against the clipping plane.
    for i, current_poly_point in enumerate(poly):
      prev_poly_point = poly[(i + len(poly) - 1) % len(poly)]
      d1 = self._classify_point_to_plane(prev_poly_point, plane, normal, axis)
      d2 = self._classify_point_to_plane(current_poly_point, plane, normal,
                                         axis)
      if d2 == _POINT_BEHIND_PLANE:
        poly_in_plane = False
        if d1 == _POINT_IN_FRONT_OF_PLANE:
          intersection = self._intersect(plane, prev_poly_point,
                                         current_poly_point, axis)
          result.append(intersection)
        elif d1 == _POINT_ON_PLANE:
          if not result or (not np.array_equal(result[-1], prev_poly_point)):
            result.append(prev_poly_point)
      elif d2 == _POINT_IN_FRONT_OF_PLANE:
        poly_in_plane = False
        if d1 == _POINT_BEHIND_PLANE:
          intersection = self._intersect(plane, prev_poly_point,
                                         current_poly_point, axis)
          result.append(intersection)
        elif d1 == _POINT_ON_PLANE:
          if not result or (not np.array_equal(result[-1], prev_poly_point)):
            result.append(prev_poly_point)

        result.append(current_poly_point)
      else:
        if d1 != _POINT_ON_PLANE:
          result.append(current_poly_point)

    if poly_in_plane:
      return poly
    else:
      return result

  def _intersect(self, plane, prev_point, current_point, axis):
    """Computes the intersection of a line with an axis-aligned plane.

    Args:
      plane: Formulated as two 3D points on the plane.
      prev_point: The point on the edge of the line.
      current_point: The other end of the line.
      axis: A tuple defining a 2D axis.

    Returns:
      A 3D point intersection of the poly edge with the plane.
    """
    alpha = (current_point[axis] - plane[axis]) / (
        current_point[axis] - prev_point[axis])
    # Compute the intersecting points using linear interpolation (lerp)
    intersection_point = alpha * prev_point + (1.0 - alpha) * current_point
    return intersection_point

  def _inside(self, plane, point, axis):
    """Check whether a given point is on a 2D plane."""
    # Cross products to determine the side of the plane the point lie.
    x, y = axis
    u = plane[0] - point
    v = plane[1] - point

    a = u[x] * v[y]
    b = u[y] * v[x]
    return a >= b

  def _classify_point_to_plane(self, point, plane, normal, axis):
    """Classify position of a point w.r.t the given plane.

    See Real-Time Collision Detection, by Christer Ericson, page 364.

    Args:
      point: 3x1 vector indicating the point
      plane: 3x1 vector indicating a point on the plane
      normal: scalar (+1, or -1) indicating the normal to the vector
      axis: scalar (0, 1, or 2) indicating the xyz axis

    Returns:
      Side: which side of the plane the point is located.
    """
    signed_distance = normal * (point[axis] - plane[axis])
    if signed_distance > _PLANE_THICKNESS_EPSILON:
      return _POINT_IN_FRONT_OF_PLANE
    elif signed_distance < -_PLANE_THICKNESS_EPSILON:
      return _POINT_BEHIND_PLANE
    else:
      return _POINT_ON_PLANE

  @property
  def intersection_points(self):
    return self._intersection_points

def viz(box1,box2):
    vertices1 = box1.vertices.detach().numpy()
    vertices2 = box2.vertices.detach().numpy()
    
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

def viz_interaction(box1,box2,inter_pts):
  
  vertices1 = box1.vertices.detach().numpy()
  vertices2 = box2.vertices.detach().numpy()
  
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
  
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(inter_pts.reshape(-1,3))
  pcd_color = np.array([0.,0.,0.])/255.0
  pcd.paint_uniform_color(pcd_color)
  
  o3d.visualization.draw_geometries([lineset1,lineset2,pcd], mesh_show_wireframe=False,mesh_show_back_face=True)

def compute_loss(ops,box1,box2):
    # n_transform = torch.eye(4)
    # n_transform[:3,3] = ops
    # box2.apply_tranform(n_transform)
    # pdb.set_trace()
    box2._vertices += ops
    box2._translaion = ops
    box2._transformation[:3,3] = ops
    iou = IoU(box1,box2)
    return  iou.iou_sampling()
  
def opt():
    #定义一个tranform，用轴角公式
    angle = 45.0 / 180.0 * np.pi
    axis = np.array([1.0,2.0,3.0]) 
    axis_norm = torch.tensor(axis/np.linalg.norm(axis))
    n_rotation = tgm.angle_axis_to_rotation_matrix(angle * axis_norm.reshape(-1,3))[:,:3,:3].squeeze(0)
    n_translate = torch.tensor([0.1,0.2,0.3],dtype=torch.float32)
    n_transform = torch.eye(4)
    n_transform[:3, :3] = n_rotation
    n_transform[:3, 3] = n_translate
    
    #初始化两个相互碰撞的包围盒：一个 AABB 一个OBB
    box1 = Box()
    box2 = Box()
    pdb.set_trace()
    box2.apply_tranform(n_transform) #经验证，该函数实现无误
    #可视化
    # viz(box1,box2)
    print(box2.translation)
    
    #将包围盒的位置作为优化变量，不做朝向的调整，只优化OBB的位置
    
    
    init_pos = torch.tensor([0.1000, 0.2000, 0.3000],dtype=torch.float32)
    opt_pos = init_pos.clone().detach()
    opt_pos.requires_grad = True
    opt_param = [opt_pos]
        
    # 构建优化器
    optimizer = Optim.Adam(opt_param, lr=0.05)
    lr_schedule = Optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    #迭代求解
    # 开始优化
    num_iter = 50  
    for epoch in tqdm(range(num_iter)):
        optimizer.zero_grad()
        loss = compute_loss(opt_pos,box1,box2)
        loss.backward()
        lr_schedule.step()
        optimizer.step()
        print("loss of epoch {} is {}".format(epoch,loss))
    pdb.set_trace()
    #可视化
    viz(box1,box2)

def test_convexhull():   
  #定义一个tranform，用轴角公式
  angle = 45.0 / 180.0 * np.pi
  axis = np.array([1.0,2.0,3.0]) 
  axis_norm = torch.tensor(axis/np.linalg.norm(axis))
  n_rotation = tgm.angle_axis_to_rotation_matrix(angle * axis_norm.reshape(-1,3))[:,:3,:3].squeeze(0)
  n_translate = torch.tensor([0.1,0.2,0.3],dtype=torch.float32)
  n_transform = torch.eye(4)
  n_transform[:3, :3] = n_rotation
  n_transform[:3, 3] = n_translate

  #初始化两个相互碰撞的包围盒：一个 AABB 一个OBB
  box1 = Box()
  box2 = Box()
  pdb.set_trace()
  box2.apply_tranform(n_transform) #经验证，该函数实现无误
  #可视化
  viz(box1,box2)
  
  #计算交点
  iou = IoU(box1,box2)
  intersection_pts = iou.get_intersection_points()
  #可视化交点
  viz_interaction(box1,box2,intersection_pts.detach().numpy())
  
  # 构建凸包
  m_convexhull = convexhull.ConvexHull(intersection_pts)
  m_convexhull.create()
  m_convexhull.get_faces()
  vol = m_convexhull.volume()
  print(vol)
  
def test_sample():   
  #定义一个tranform，用轴角公式
  angle = 45.0 / 180.0 * np.pi
  axis = np.array([1.0,2.0,3.0]) 
  axis_norm = torch.tensor(axis/np.linalg.norm(axis))
  n_rotation = tgm.angle_axis_to_rotation_matrix(angle * axis_norm.reshape(-1,3))[:,:3,:3].squeeze(0)
  n_translate = torch.tensor([0.1,0.2,0.3],dtype=torch.float32)
  n_transform = torch.eye(4)
  n_transform[:3, :3] = n_rotation
  n_transform[:3, 3] = n_translate

  #初始化两个相互碰撞的包围盒：一个 AABB 一个OBB
  box1 = Box()
  box2 = Box()
  box2.apply_tranform(n_transform) #经验证，该函数实现无误
  #可视化
  # viz(box1,box2)
  
  # 散点求
  iou = IoU(box1,box2)
  intersection_iou = iou.iou()
  print("intersection_iou is {}".format(intersection_iou))
  pdb.set_trace()
  sample_iou = iou.iou_sampling()
  print("sampling_iou is {}".format(sample_iou))

if __name__ == "__main__":
  pdb.set_trace()
  test_convexhull()
  # test_sample()
  # opt()
  

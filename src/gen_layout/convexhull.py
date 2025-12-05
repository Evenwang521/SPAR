import numpy as np
import torch
import pdb

class Face:
    def __init__(self, a, b, c, ok):
        self.a = a
        self.b = b
        self.c = c
        self.ok = ok

class ConvexHull:
    def __init__(self,pts):
        self._pts = pts  #初始点   tensor (n,3) 
        self._num_pts = pts.shape[0]
        self._Faces  = []   #凸包表面 tensor (N,3) 表示N个面的三个顶点索引
        self._num_faces = 0
        self._g = torch.zeros(self._num_pts,self._num_pts,dtype=torch.int)
        self._volume = None    #凸包表面

    def is_coplanar(self, p, f):
        '''
        判断点p与三角形ABC的关系
        '''
        ab = self._pts[f.b] - self._pts[f.a]
        ac = self._pts[f.c] - self._pts[f.a]
        ap = p - self._pts[f.a]
        return torch.dot(torch.cross(ab,ac),ap)
    
    def is_collinear(self,a, b, c):
        '''
        判断a,b,c是否共面
        '''
        ab = b - a
        ac = c - a
        return torch.norm(torch.cross(ab,ac)) <= 1e-8
  
    def update(self, p, now):
        self._Faces[now].ok = False
        self.deal(p, self._Faces[now].b, self._Faces[now].a)
        self.deal(p, self._Faces[now].c, self._Faces[now].b)
        self.deal(p, self._Faces[now].a, self._Faces[now].c)
    
    def deal(self, p, a, b):
        
        f = self._g[a][b]  # 搜索与该边相邻的另一个平面
        add = Face(0,0,0,False)
        if self._Faces[f].ok:
            if self.is_coplanar(self._pts[p], self._Faces[f]) > 1e-8:
                self.update(p, f)
            else:
                add.a = b
                add.b = a
                add.c = p  
                add.ok = True
                self._g[p][b] = self._g[a][p] = self._g[b][a] = self._num_faces
                self._Faces.append(add)
                # print("faces {} is vert {} ; vert {} ; vert {}".format(self._num_faces,add.a,add.b,add.c))
                self._num_faces += 1
   
    def create(self):
        
        if self._num_pts < 4 :
            self._volume = .0
            return
        
        #确保init_pts的前三个点不共线
        for i in range(2, self._num_pts):
            if not self.is_collinear(self._pts[0], self._pts[1], self._pts[i]):
                self._pts[2],self._pts[i] = self._pts[i],self._pts[2]
                break
           
        #确保init_pts的前四个点不共面
        for i in range(3,self._num_pts):
            temp_f = Face(0,1,2,False)
            if torch.abs(self.is_coplanar(self._pts[i],temp_f)) > 1e-8:
                self._pts[3],self._pts[i] = self._pts[i],self._pts[3]
                break
            
        #从初始四个点构建凸包
        for i in range(4):
            add = Face((i + 1) % 4, (i + 2) % 4, (i + 3) % 4, True)
            if self.is_coplanar(self._pts[i], add) > 0:
                add.b, add.c = add.c, add.b
            self._g[add.a][add.b] = self._g[add.b][add.c] = self._g[add.c][add.a] = self._num_faces
            self._Faces.append(add)
            self._num_faces += 1
        
        #遍历其余的点，如果在四个面外，就
        for i in range(4, self._num_pts):
            for j in range(self._num_faces):
                if self._Faces[j].ok and (self.is_coplanar(self._pts[i], self._Faces[j]) > 1e-8):
                    self.update(i, j)
                    break
                
        tmp = self._num_faces  
        self._num_faces = 0
        for i in range(tmp):
            if self._Faces[i].ok:
                self._Faces[self._num_faces] = self._Faces[i]
                self._num_faces += 1
                
    def volume(self):
        center = torch.mean(self._pts, dim=0)
        volume = torch.tensor(.0)
        for i in range(self._num_faces):
            volume += (self.is_coplanar(center,self._Faces[i]))
        self._volume = torch.abs(volume)/6.0
        return self._volume
    
    def get_faces(self):
        for i in range(self._num_faces):
            print("faces {} is {} ; {} ; {}".format(i,self._Faces[i].a,self._Faces[i].b,self._Faces[i].c))

if __name__ == "__main__":
    unit_box = np.asarray([
    [0., 0., 0.],
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, -0.5],
    [1, 1, 1],
    ])
    pdb.set_trace()
    init_pts = torch.tensor([[ 2.6764e-01,  5.0000e-01, -3.1966e-01],
        [ 5.0000e-01,  1.5007e-01, -4.5884e-01],
        [ 5.0000e-01,  5.0000e-01,  2.1703e-01],
        [ 8.5155e-02, -5.0000e-01,  5.0000e-01],
        [ 3.2932e-06, -5.0000e-01,  3.0332e-01],
        [-2.2183e-01, -4.6806e-01, -1.4735e-01],
        [-5.0000e-01, -4.9138e-02,  1.9260e-02],
        [-5.0000e-01,  1.9977e-01,  5.0000e-01],
        [-5.0000e-01,  3.5432e-01,  5.0000e-01],
        [-5.0000e-01,  5.0000e-01,  1.3371e-01],
        [-2.8061e-01,  5.0000e-01,  5.0000e-01],
        [ 5.0000e-01, -2.4621e-01,  5.0000e-01],
        [ 5.0000e-01,  1.3554e-01, -4.5987e-01],
        [-2.2183e-01, -4.6806e-01, -1.4735e-01],
        [ 3.3006e-06, -5.0000e-01,  3.0332e-01],
        [ 1.1780e-01, -5.0000e-01,  5.0000e-01],
        [-5.0000e-01,  5.0000e-01,  5.8178e-02],
        [-5.0000e-01, -4.9138e-02,  1.9260e-02],
        [-2.2183e-01, -4.6806e-01, -1.4735e-01],
        [ 5.0000e-01,  1.3554e-01, -4.5987e-01],
        [ 5.0000e-01,  1.5007e-01, -4.5884e-01],
        [ 2.6764e-01,  5.0000e-01, -3.1966e-01],
        [ 1.0000e-01,  2.0000e-01,  3.0000e-01],
        [-2.2183e-01, -4.6806e-01, -1.4735e-01],
        [ 5.0000e-01,  1.5007e-01, -4.5884e-01],
        [ 5.0000e-01,  1.3554e-01, -4.5987e-01],
        [ 5.0000e-01, -2.4621e-01,  5.0000e-01],
        [ 5.0000e-01,  5.0000e-01,  5.0000e-01],
        [ 5.0000e-01,  5.0000e-01,  2.1703e-01],
        [-5.0000e-01, -4.9138e-02,  1.9260e-02],
        [-5.0000e-01,  5.0000e-01,  5.8178e-02],
        [-5.0000e-01,  5.0000e-01,  1.3371e-01],
        [-5.0000e-01,  3.5432e-01,  5.0000e-01],
        [-5.0000e-01,  1.9977e-01,  5.0000e-01],
        [-5.0000e-01,  5.0000e-01,  1.3371e-01],
        [-5.0000e-01,  5.0000e-01,  5.8178e-02],
        [ 2.6764e-01,  5.0000e-01, -3.1966e-01],
        [ 5.0000e-01,  5.0000e-01,  2.1703e-01],
        [ 5.0000e-01,  5.0000e-01,  5.0000e-01],
        [-2.8061e-01,  5.0000e-01,  5.0000e-01],
        [ 4.2021e-06, -5.0000e-01,  3.0333e-01],
        [ 8.5155e-02, -5.0000e-01,  5.0000e-01],
        [ 1.1780e-01, -5.0000e-01,  5.0000e-01],
        [ 1.1780e-01, -5.0000e-01,  5.0000e-01],
        [ 8.5155e-02, -5.0000e-01,  5.0000e-01],
        [-5.0000e-01,  1.9977e-01,  5.0000e-01],
        [-5.0000e-01,  3.5432e-01,  5.0000e-01],
        [-2.8061e-01,  5.0000e-01,  5.0000e-01],
        [ 5.0000e-01,  5.0000e-01,  5.0000e-01],
        [ 5.0000e-01, -2.4621e-01,  5.0000e-01],
        [ 1.4901e-08,  0.0000e+00, -2.9802e-08],
        [ 5.0000e-01,  5.0000e-01,  5.0000e-01]])
    # init_pts = torch.tensor(unit_box,dtype=torch.float32)
    convexhull = ConvexHull(init_pts)
    convexhull.create()
    pdb.set_trace()
    vol = convexhull.volume()
    # convexhull.get_faces()
    print(vol)


from PIL import Image
from klampt.math import vectorops,so3,se3
import numpy as np
import random

class RGBDCamera:
    """A class to store camera information.
    
    Attributes:
        color_intrinsics (dict): a map from intrinsic parameter names to
            values. For the color image.
        depth_intrinsics (dict): a map from intrinsic parameter names to
            values. For the depth image.
        depth_image_scale (float): division by this value converts from depth image values to meters
        depth_minimum (float): the minimum depth, in meters
        depth_maximum (float): the maximum depth, in meters
        T_color_to_depth (se3 element): the extrinsic map from color camera to
            depth camera (currently not supported)
        T_depth_to_color (se3 element): the extrinsic map from depth camera to
            color camera (currently not supported)
    """
    def __init__(self):
        self.color_intrinsics = None
        self.depth_intrinsics = None
        self.depth_image_scale = None
        self.depth_minimum = None
        self.depth_maximum = None
        self.T_color_to_depth = None
        self.T_depth_to_color = None

    def load_intrinsics(self,color_fn,depth_fn,format='realsense'):
        """
        Args
            color_fn (str): color intrinsics file
            depth_fn (str): depth intrinsics file
            format (str): can be 
                'json': just loads structure directly
                'realsense': json with ppx and ppy as the cx and cy values
        """
        import json
        with open(color_fn,'r') as f:
            self.color_intrinsics = json.load(f)
        if format=='realsense':
            self.color_intrinsics['cx'] = self.color_intrinsics['ppx']
            self.color_intrinsics['cy'] = self.color_intrinsics['ppy']
            del self.color_intrinsics['ppx']
            del self.color_intrinsics['ppy']
        with open(depth_fn,'r') as f:
            self.depth_intrinsics = json.load(f)
        if format=='realsense':
            self.depth_intrinsics['cx'] = self.depth_intrinsics['ppx']
            self.depth_intrinsics['cy'] = self.depth_intrinsics['ppy']
            del self.depth_intrinsics['ppx']
            del self.depth_intrinsics['ppy']
    
    def save_intrinsics(self,color_fn,depth_fn,format='realsense'):
        """
        Args
            color_fn (str): color intrinsics file
            depth_fn (str): depth intrinsics file
            format (str): can be 
                'json': just saves structure directly
                'realsense': json with ppx and ppy as the cx and cy values
        """
        import json
        import copy
        jsonobj = copy.copy(self.color_intrinsics)
        if format=='realsense':
            jsonobj['ppx'] = jsonobj['cx']
            jsonobj['ppy'] = jsonobj['cy']
            del jsonobj['cx']
            del jsonobj['cy']
        with open(color_fn,'w') as f:
            json.dump(jsonobj,f)
        jsonobj = copy.copy(self.depth_intrinsics)
        if format=='realsense':
            jsonobj['ppx'] = jsonobj['cx']
            jsonobj['ppy'] = jsonobj['cy']
            del jsonobj['cx']
            del jsonobj['cy']
        with open(depth_fn,'r') as f:
            json.dump(jsonobj,f)
        
    

class RGBDScan:
    """Represents a single RGB-D scan with optional ground truth and estimated
    transforms.
    
    Attributes:
        camera (RGBDCamera): the camera taking this scan, if known.
        timestamp (float): the time at which this was taken (application-dependent)
        rgb (np.ndarray): the RGB image, shape (h,w,3)
        depth (np.ndarray): the depth image, shape (h,w)
        T_ground_truth (klampt se3 element): the ground truth pose
        T_estimate (klampt se3 element): the estimated pose
    """
    def __init__(self,rgb=None,depth=None,timestamp=None,T_ground_truth=None,T_estimate=None):
        self.camera = None
        self.rgb = rgb
        self.depth = depth
        self.timestamp = timestamp
        self.T_ground_truth = T_ground_truth
        self.T_estimate = T_estimate

    def plot(self,plt,figsize=(14,4),**options):
        """Plots using matplotlib."""
        fig = plt.figure(figsize=figsize,**options)
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(self.rgb)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(self.depth)

    def get_point_cloud(self,camera=None,
        colors=True,normals=True,structured=False,
        near_filter=None,far_filter=None,format='numpy'):
        """Converts an RGBD scan to a point cloud in camera coordinates.
        
        Points whose Z value is less than near_filter or greater than far_filter
        will be excluded.
        
        if format=='numpy':
            If colors=True and normals=True, the result is an N x 9 Numpy array, giving
            colors and normals.  I.e, the matrix has columns X Y Z R G B Nx Ny Nz. 
            R, G, B are in the range [0,1]
            
            If colors=True and normals=False, only the first 6 columns are returned.
            If colors=False and normals=True, only the first 3 and last 3 columns are returned.
            If colors=False and normals=False, only the X,Y,Z columns are returned.
        if format == 'PointCloud':
            A Klampt PointCloud is returned.

        If structured=True, all points are returned, even ones with missing depth
        and those outside of the near/far filter.  Their rows will be zero 
        """
        if camera is None:
            camera = self.camera 
        if camera is None or camera.depth_intrinsics is None:
            raise ValueError("Need a camera's intrinsics to be defined")
        if self.depth is None:
            raise ValueError("Depth is not defined for this scan")
        if colors and self.rgb is None:
            print("Warning, requested colors but rgb information is not defined for this scan")
            colors = False
        if colors and normals:
            channels = range(9)
        elif colors:
            channels = range(6)
        elif normals:
            channels = list(range(3)) + list(range(6,9))
        else:
            channels = range(3)
        if near_filter is None:
            near_filter = camera.depth_minimum
        else:
            near_filter = max(near_filter,camera.depth_minimum)
        if far_filter is None:
            far_filter = camera.depth_maximum
        else:
            far_filter = min(far_filter,camera.depth_maximum)
        fx,fy = camera.depth_intrinsics['fx'],camera.depth_intrinsics['fy']
        cx,cy = camera.depth_intrinsics['cx'],camera.depth_intrinsics['cy']
        depth_scale = camera.depth_image_scale
        h,w = self.depth.shape[0],self.depth.shape[1]
        Z = self.depth * (1.0/ depth_scale)
        X, Y = np.meshgrid(np.array(range(w))-cx,np.array(range(h))-cy)
        X *= Z*(1.0/fx)
        Y *= Z*(1.0/fy)
        points = np.stack((X,Y,Z),-1).reshape(w*h,3)
        #compute colors
        if colors:
            if camera.T_color_to_depth is not None:
                raise NotImplementedError("TODO: map from color to depth frames")
            colors = (self.rgb / 255.0).reshape(w*h,3)
            have_colors = True
        else:
            colors = np.empty((w*h,3))
            have_colors = False
        if normals:
            #compute normals from image
            #What's all this fuss? getting good normals in the presence of missing depth readings
            gxc = (Z[:,2:]-Z[:,:-2])/(X[:,2:]-X[:,:-2])
            gxn = (Z[:,1:]-Z[:,:-1])/(X[:,1:]-X[:,:-1])
            gyc = (Z[2:,:]-Z[:-2,:])/(Y[2:,:]-Y[:-2,:])
            gyn = (Z[1:,:]-Z[:-1,:])/(Y[1:,:]-Y[:-1,:])
            nancol = np.full((h,1),np.nan)
            gxc = np.hstack((nancol,gxc,nancol))
            gxp = np.hstack((nancol,gxn))
            gxn = np.hstack((gxn,nancol))
            gx = np.where(np.isnan(gxc),np.where(np.isnan(gxn),np.where(np.isnan(gxp),0,gxp),gxn),gxc)
            nanrow = np.full((1,w),np.nan)
            gyc = np.vstack((nanrow,gyc,nanrow))
            gyp = np.vstack((nanrow,gyn))
            gyn = np.vstack((gyn,nanrow))
            gy = np.where(np.isnan(gyc),np.where(np.isnan(gyn),np.where(np.isnan(gyp),0,gyp),gyn),gyc)
            normals = np.stack((gx,gy,-np.ones((h,w))),-1).reshape(w*h,3)
            np.nan_to_num(normals,copy=False)
            normals = normals / np.linalg.norm(normals,axis=1)[:,np.newaxis]
            np.nan_to_num(normals,copy=False)
            have_normals = True
        else:
            normals = np.empty((w*h,3))
            have_normals = False
        #join them all up
        points = np.hstack((points,colors,normals))
        #depth filtering
        indices = np.logical_and(points[:,2] >= near_filter,points[:,2] <= far_filter)
        #print(np.sum(indices),"Points pass the depth test")
        #print("Depth range",points[:,2].min(),points[:,2].max())
        if structured:
            res = points[:,channels]
            res[~indices,:]=0
        else:
            res = points[indices.nonzero()[0]][:,channels]
        if format == 'PointCloud':
            from klampt import PointCloud
            pc = PointCloud()
            pc.setPoints(res.shape[0],res[:,:3].flatten())
            pstart = 0
            if have_colors:
                pc.addProperty('r')
                pc.addProperty('g')
                pc.addProperty('b')
                pc.setProperties(0,res[:,3])
                pc.setProperties(1,res[:,4])
                pc.setProperties(2,res[:,5])
                pstart = 3
            if have_normals:
                pc.addProperty('normal_x')
                pc.addProperty('normal_y')
                pc.addProperty('normal_z')
                pc.setProperties(0+pstart,res[:,6])
                pc.setProperties(1+pstart,res[:,7])
                pc.setProperties(2+pstart,res[:,8])
            return pc
        return res


def plot_point_cloud(pc,plt,colors=True,normals=False,figsize=(12,12),**options):
    """Plots a point cloud using matplotlib."""    
    from mpl_toolkits import mplot3d
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=figsize,**options)
    ax = plt.axes(projection='3d')
    if colors or pc.shape[1] < 6:
        ax.scatter(pc[:,0],pc[:,1],pc[:,2],s=1,c=pc[:,2])
    else:
        ax.scatter(pc[:,0],pc[:,1],pc[:,2],s=1,c=pc[:,3:6])
    if normals:
        if not isinstance(normals,int):
            normals = 250
        inds = random.choices(range(pc.shape[0]),k=normals)
        ax.quiver(pc[inds,0],pc[inds,1],pc[inds,2],pc[inds,6],pc[inds,7],pc[inds,8],length=0.05,normalize=False)
    zmax = np.max(pc[:,2])
    bmin = np.amin(pc[:,:3],0)
    bmax = np.amax(pc[:,:3],0)
    dmax = np.amax(bmax-bmin)
    c = (bmin+bmax)*0.5
    ax.set_xlim([c[0]-dmax*0.5,c[0]+dmax*0.5])
    ax.set_ylim([c[1]-dmax*0.5,c[1]+dmax*0.5])
    ax.set_zlim([c[2]-dmax*0.5,c[2]+dmax*0.5])


def transform_point_cloud(pc,T,point_channels=[0,1,2],normal_channels=[6,7,8]):
    """Given a point cloud `pc` and a transform T, apply the transform
    to the point cloud (in place).
    
    Args:
        pc (np.ndarray): an N x M numpy array, with N points and M
            channels.
        T (klampt se3 element): a Klamp't se3 element representing
            the transform to apply.
        point_channels (list of 3 ints): The channel indices (columns)
             in pc corresponding to the point data.
        normal_channels (list of 3 ints): The channels indices(columns)
            in pc corresponding to the normal data.  If this is None
            or an index is >= M, just ignore.
        """
    N,M = pc.shape
    assert len(point_channels) == 3
    for i in point_channels:
        assert i < M,"Invalid point_channel"
    
    for i in range(N):
        point_data = pc[i,:]
        point = point_data[point_channels]
        point_data[point_channels] = se3.apply(T,point)
        
    if normal_channels is not None and normal_channels[0] < M:
        for i in normal_channels:
            assert i < M,"Invalid normal_channel"
        for i in range(N):
            point_data = pc[i,:]
            point = point_data[normal_channels]
            point_data[normal_channels] = so3.apply(T[0],point)



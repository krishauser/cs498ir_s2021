import numpy as np
from klampt import vis
from klampt.model.geometry import fit_plane,fit_plane3
from klampt.model.trajectory import Trajectory
from klampt.math import vectorops
from klampt.io import numpy_convert
from klampt import PointCloud
import random
import sys
sys.path.append("../common")
from rgbd import *
from rgbd_realsense import load_rgbd_dataset
import json

PROBLEM = '1a'
PROBLEM = '1b'
PROBLEM = '1c'
DONT_EXTRACT = False #if you just want to see the point clouds, turn this to true

def extract_planes_ransac_a(pc,N=100,m=3,inlier_threshold=0.01,inlier_count=20000):
    """Uses RANSAC to determine which planes make up the scene

    Args:
        pc: an Nx3 numpy array of points
        N: the number of iterations used to sample planes
        m: the number of points to fit on each iteration
        inlier_threshold: the distance between plane / point to consider
            it an inlier
        inlier_count: consider a plane to be an inlier (and output it!) if this
            many points are inliers
    
    Returns:
        list of lists of int: a list of lists of point indices that belong to
        planes. If `plane_indices` is the result, each entry represents a plane,
        and the plane equation can be obtained using `fit_plane(pc[plane_indices[i]])`.
    """
    #to fit a plane through 3 points:
    #(a,b,c,d) = fit_plane3(p1,p2,p3)

    #to fit a plane through N>=3 points:
    #(a,b,c,d) = fit_plane([p1,p2,p3,p4])
    planes = []
    planes.append([0,1,2,3,4])
    planes.append([5,6,7,8])
    return planes
    
def extract_planes_ransac_b(pc,N=100,m=3,inlier_threshold=0.01,inlier_count=20000):
    """Uses RANSAC to determine which planes make up the scene

    Args:
        pc: an Nx3 numpy array of points
        N: the number of iterations used to sample planes
        m: the number of points to fit on each iteration
        inlier_threshold: the distance between plane / point to consider
            it an inlier
        inlier_count: consider a plane to be an inlier (and output it!) if this
            many points are inliers
    
    Returns:
        list of lists of int: a list of lists of point indices that belong to
        planes. If `plane_indices` is the result, each entry represents a plane,
        and the plane equation can be obtained using `fit_plane(pc[plane_indices[i]])`.
    """
    #to fit a plane through 3 points:
    #(a,b,c,d) = fit_plane3(p1,p2,p3)

    #to fit a plane through N>=3 points:
    #(a,b,c,d) = fit_plane([p1,p2,p3,p4])
    planes = []
    planes.append([0,1,2,3,4])
    planes.append([5,6,7,8])
    return planes

def extract_planes_ransac_c(pc,N=100,m=3,inlier_threshold=0.015,inlier_count=50000):
    """Uses RANSAC to determine which planes make up the scene

    Args:
        pc: an Nx3 numpy array of points
        N: the number of iterations used to sample planes
        m: the number of points to fit on each iteration
        inlier_threshold: the distance between plane / point to consider
            it an inlier
        inlier_count: consider a plane to be an inlier (and output it!) if this
            many points are inliers
    
    Returns:
        list of lists of int: a list of lists of point indices that belong to
        planes. If `plane_indices` is the result, each entry represents a plane,
        and the plane equation can be obtained using `fit_plane(pc[plane_indices[i]])`.
    """
    #to fit a plane through 3 points:
    #(a,b,c,d) = fit_plane3(p1,p2,p3)

    #to fit a plane through N>=3 points:
    #(a,b,c,d) = fit_plane([p1,p2,p3,p4])
    planes.append([0,1,2,3,4])
    planes.append([5,6,7,8])
    return planes

if __name__ == '__main__':
    scans = load_rgbd_dataset('calibration')
    planesets = []
    for scanno,s in enumerate(scans):
        pc = s.get_point_cloud(colors=True,normals=True,structured=True,format='PointCloud')
        vis.clear()
        vis.setWindowTitle("Scan "+str(scanno))
        vis.add("PC",pc)
        if not DONT_EXTRACT:
            pc2 = s.get_point_cloud(colors=False,normals=False,structured=True)
            if PROBLEM=='1a':
                planes = extract_planes_ransac_a(pc2)
            elif PROBLEM=='1b':
                planes = extract_planes_ransac_b(pc2)
            else:
                planes = extract_planes_ransac_c(pc2)
            planesets.append(planes)
            for j,plane in enumerate(planes):
                color = (random.random(),random.random(),random.random())
                for i in plane:
                    pc.setProperty(i,0,color[0])
                    pc.setProperty(i,1,color[1])
                    pc.setProperty(i,2,color[2])
                plane_eqn = fit_plane(pc2[plane])
                centroid = np.average(pc2[plane],axis=0).tolist()
                assert len(centroid)==3
                vis.add("Plane "+str(j),Trajectory(milestones=[centroid,vectorops.madd(centroid,plane_eqn[:3],0.1)]),color=(1,1,0,1))
        vis.dialog()
    if not DONT_EXTRACT:
        print("Dumping plane identities to planesets.json")
        with open("planesets.json","w") as f:
            json.dump(planesets,f)


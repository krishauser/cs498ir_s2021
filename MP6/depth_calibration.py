import numpy as np
from klampt.math import vectorops,se3
from klampt.model.geometry import fit_plane
from scipy.optimize import minimize
import random
import sys
sys.path.append("../common")
from rgbd import *
from rgbd_realsense import load_rgbd_dataset,sr300_factory_calib
import json

#PROBLEM = '2a'
#PROBLEM = '2b'
PROBLEM = '2c'
CALIBRATION = 'spec'
#CALIBRATION = 'camera'
SUBSET_PLANES = 1000

def mutual_orthogonality(scans,planesets,fx,fy,cx,cy):
    cost = 0
    for scanno,(scan,planeset) in enumerate(zip(scans,planesets)):
        if len(planeset)<=1: continue
        #TODO: set up the point cloud that would have been obtained using the given intrinsic parameters
        pc = scan.get_point_cloud(colors=False,normals=False,structured=True)
        w,h = 640,480
        plane_normals = []
        for plane in planeset:
            plane_eqn = fit_plane(pc[plane])
            plane_normals.append(np.array(plane_eqn[:3]))
        #do something with the cost
    return cost

def calibrate_intrinics_fxfy(scans,planesets):
    cam = scans[0].camera
    fx = cam.depth_intrinsics['fx']
    fy = cam.depth_intrinsics['fy']
    cx = cam.depth_intrinsics['cx']
    cy = cam.depth_intrinsics['cy']
    print("TODO...")

def calibrate_intrinics_all(scans,planesets):
    cam = scans[0].camera
    fx = cam.depth_intrinsics['fx']
    fy = cam.depth_intrinsics['fy']
    cx = cam.depth_intrinsics['cx']
    cy = cam.depth_intrinsics['cy']
    print("TODO...")

if __name__ == '__main__':
    with open("planesets.json","r") as f:
        planesets = json.load(f)
    if SUBSET_PLANES is not None:
        #select a smaller subset of the planesets
        for i,planeset in enumerate(planesets):
            for j,plane in enumerate(planeset):
                if len(plane) > SUBSET_PLANES:
                    planeset[j] = list(random.sample(plane,SUBSET_PLANES))
    scans = load_rgbd_dataset('calibration')
    assert len(planesets) == len(scans)
    #reset to factory calibration
    if CALIBRATION=='spec':
        cam = sr300_factory_calib
        for s in scans:
            s.camera = cam
    else:
        #use calibration from camera
        cam = scans[0].camera
    if PROBLEM == '2a':
        fx = cam.depth_intrinsics['fx']
        fy = cam.depth_intrinsics['fy']
        cx = cam.depth_intrinsics['cx']
        cy = cam.depth_intrinsics['cy']
        print("Cost():",mutual_orthogonality(scans,planesets,fx,fy,cx,cy))
        print()
        print("Some random testing...")
        print("Cost(fx,fy*1.01):",mutual_orthogonality(scans,planesets,fx*1.01,fy*1.01,cx,cy))
        print("Cost(fx,fy*1.1):",mutual_orthogonality(scans,planesets,fx*1.1,fy*1.1,cx,cy))
        print("Cost(fx,fy*1.2):",mutual_orthogonality(scans,planesets,fx*1.2,fy*1.2,cx,cy))
        print("Cost(fx,fy*.99):",mutual_orthogonality(scans,planesets,fx*0.99,fy*0.99,cx,cy))
        print("Cost(fx,fy*.95):",mutual_orthogonality(scans,planesets,fx*0.95,fy*0.95,cx,cy))
        print("Cost(fx,fy*.9):",mutual_orthogonality(scans,planesets,fx*0.9,fy*0.9,cx,cy))
        print("Cost(cx offset):",mutual_orthogonality(scans,planesets,fx,fy,cx+10,cy+30))
    elif PROBLEM == '2b':
        res = calibrate_intrinics_fxfy(scans,planesets)
    else:
        res = calibrate_intrinics_all(scans,planesets)

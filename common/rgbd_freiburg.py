from rgbd import *
from PIL import Image
import numpy as np
from klampt.model.trajectory import SE3Trajectory
from klampt.math import so3

kinect_factory_calib = RGBDCamera()
kinect_factory_calib.color_intrinsics = {
    'width':640,
    'height':640,
    'fx':525.0,  # focal length x
    'fy':525.0,  # focal length y
    'cx':319.5,  # optical center x
    'cy':239.5  # optical center y
    }
kinect_factory_calib.depth_intrinsics = kinect_factory_calib.color_intrinsics
kinect_factory_calib.depth_image_scale = 5000 # for the 16-bit PNG files


def _load_time_stamped_file(fn):
    """Returns a pair (times,items)"""
    with open(fn) as f:
        times = []
        items = []
        for l in f.readlines():
            l = l.strip()
            if l.startswith('#'):
                continue
            v = l.split()
            times.append(float(v[0]))
            if len(v) == 2:
                items.append(v[1])
            else:
                items.append(v[1:])
        return times,items
    raise IOError("Unable to load "+fn)
    
def load_rgbd_dataset(dataset):
    """Returns a list of RGBDScans from a folder"""
    ground_truth_t,ground_truth_transforms = _load_time_stamped_file(dataset+'/groundtruth.txt')
    ground_truth_transforms = [[float(v) for v in T] for T in ground_truth_transforms]
    #convert to Klamp't SE3 elements
    for i,trans_quat in enumerate(ground_truth_transforms):
        qx,qy,qz,qw = trans_quat[3:]
        ground_truth_transforms[i] = (so3.from_quaternion((qw,qx,qy,qz)),trans_quat[:3])
    ground_truth_path = SE3Trajectory(ground_truth_t,ground_truth_transforms)
    rgb_t,rgb_files = _load_time_stamped_file(dataset+'/rgb.txt')
    depth_t,depth_files = _load_time_stamped_file(dataset+'/depth.txt')
    depth_t = np.array(depth_t)
    scans = []
    last_ind = -1
    for t,file in zip(rgb_t,rgb_files):
        im = Image.open(dataset+'/'+file)
        rgb_array  = np.asarray(im)
        ind = np.abs(depth_t-t).argmin()
        if ind > last_ind+1:
            ind = last_ind+1
        last_ind = ind
        #print("Time",t,"Depth index",ind,"Distance",np.abs(depth_t-t)[ind])
        depth_file = depth_files[ind]
        im = Image.open(dataset+'/'+depth_file)
        depth_array  = np.asarray(im)
        T = ground_truth_path.eval_se3(t)
        scans.append(RGBDScan(rgb_array,depth_array,timestamp=t,T_ground_truth=T))
        scans[-1].camera = kinect_factory_calib
    return scans

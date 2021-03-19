from rgbd import *
import os
import copy
from PIL import Image
import numpy as np

sr300_factory_calib = RGBDCamera()
sr300_factory_calib.color_intrinsics = {
    'width':640,
    'height':480,
    'fx':698.0,  # focal length x
    'fy':698.0,  # focal length y
    'cx':319.5,  # optical center x
    'cy':239.5  # optical center y
    }
sr300_factory_calib.depth_intrinsics = sr300_factory_calib.color_intrinsics
sr300_factory_calib.depth_image_scale = 8000  #8000 values per meter
sr300_factory_calib.depth_minimum = 0.15
sr300_factory_calib.depth_maximum = 2.0
    
def load_rgbd_dataset(dataset):
    """Returns a list of RGBDScans from a folder"""
    scans = dict()
    cam = sr300_factory_calib
    for fn in os.listdir(dataset):
        if fn.startswith('color') and fn.endswith('.png'):
            index = int(fn[6:10])
        elif fn.startswith('depth_aligned') and fn.endswith('.png'):
            index = int(fn[14:18])
        elif fn=='color_intrinsics.json':
            cam = copy.deepcopy(cam)
            #note: depth is aligned to color, so the depth has the same intrinsics as color
            cam.load_intrinsics(os.path.join(dataset,fn),os.path.join(dataset,fn))
            continue
        else:
            continue
        if index not in scans:
            scans[index] = RGBDScan()
        abspath = os.path.join(dataset,fn)
        
        print("Loading image",abspath)
        im = Image.open(abspath)
        imarray  = np.asarray(im)
        if fn.startswith('color'):
            scans[index].rgb = imarray
        else:
            scans[index].depth = imarray

    assert cam.color_intrinsics is not None
    assert cam.depth_intrinsics is not None
    scanlist = []
    for k in sorted(scans.keys()):
        scans[k].camera = cam
        scanlist.append(scans[k])
    return scanlist

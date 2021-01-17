from gripper import GripperInfo
from grasp import *
from klampt import Geometry3D,PointCloud,RigidObjectModel
from klampt.math import vectorops,so3,se3
from klampt.model import ik
from klampt.model import contact
try:
    from klampt.model.sensing import fit_plane,fit_plane_centroid
except ImportError:
    from klampt.model.geometry import fit_plane,fit_plane_centroid
from klampt import io
from klampt.io import numpy_convert
import copy
import json
import math
import numpy as np


class GraspFlatAreaSampler(GraspSamplerBase):
    """A GraspSamplerBase subclass that will find flat areas in a
    point cloud for a vacuum gripper.
    """
    def __init__(self,gripper,roughness_penalty,vertical_penalty):
        self._gripper = gripper
        assert gripper.primary_axis is not None,"Gripper needs a primary axis"
        assert gripper.opening_span is not None,"Gripper needs an opening span"
        self.roughness_penalty = roughness_penalty
        self.vertical_penalty = vertical_penalty
        if not callable(roughness_penalty):
            self.roughness_penalty = lambda var:var*roughness_penalty
        if not callable(vertical_penalty):
            self.vertical_penalty = lambda angle:angle*vertical_penalty
        self.pc = None
        self.pc_xform = None
        self.options = None
        self.index = None

    def gripper(self):
        return self._gripper

    def init(self,scene,object,hints):
        """Needs object to contain a structured PointCloud."""
        if not isinstance(object,(RigidObjectModel,Geometry3D,PointCloud)):
            print("Need to pass an object as a RigidObjectModel, Geometry3D, or PointCloud")
            return False
        if isinstance(object,RigidObjectModel):
            return self.init(scene,object.geometry(),hints)
        pc = None
        xform = None
        if isinstance(object,Geometry3D):
            pc = object.getPointCloud()
            xform = object.getCurrentTransform()
        else:
            pc = object
            xform = se3.identity()
        self.pc = pc
        self.pc_xform = xform

        #now look through PC and find flat parts
        #do a spatial hash
        from collections import defaultdict
        estimation_knn = 6
        pts = numpy_convert.to_numpy(pc)
        N = pts.shape[0]
        positions = pts[:,:3]
        normals = np.zeros((N,3))
        indices = (positions * (1.0/self._gripper.opening_span)).astype(int)
        pt_hash = defaultdict(list)
        for i,(ind,p) in enumerate(zip(indices,positions)):
            pt_hash[ind].append((i,p))
        options = []
        for (ind,iplist) in pt_hash.items():
            if len(iplist) < estimation_knn:
                pass
            else:
                pindices = [ip[0] for ip in iplist]
                pts = [ip[1] for ip in iplist]
                c,n = fit_plane_centroid(pts)
                if n[2] < 0:
                    n = vectorops.mul(n,-1)
                verticality = self.vertical_penalty(math.acos(n[2]))
                var = sum(vectorops.dot(vectorops.sub(p,c),n)**2 for p in pts)
                roughness = self.roughness_penalty(var)
                options.append((cn,n,verticality + roughness))
        if len(options) == 0:
            return False
        self.options = options.sorted(key=lambda x:-x[2])
        self.index = 0
        return True

    def next(self):
        """Returns the next Grasp from the database."""
        if self.options is None:
            return None
        
        if self.index >= len(self.options):
            self.options = None
            return None

        c,n,score = self.options(self.index)
        self.index += 1
        cworld = se3.apply(self.pc_xform,c)
        nworld = so3.apply(self.pc_xform[0],n)
        objective = IKObjective()
        objective.setLinks(self.gripper.link)
        objective.setFixedPoint(self.gripper.center,cworld)
        objective.setAxialRotConstraint(self.gripper.primary_axis,vectorops.mul(nworld,-1))
        return Grasp(objective,score=score)
    
    def score(self):
        """Returns the top score.
        """
        if self.options is None: return 0
        if self.index >= len(self.options):
            return 0
        return math.exp(-self.options[self.index][2])

    
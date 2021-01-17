from gripper import GripperInfo
from grasp import *
from klampt.model import ik
from klampt.math import vectorops,so3,se3
from klampt.model import contact
from klampt import io
import copy
import json


class GraspDatabase:
    """A database of grasps, loadable from disk"""
    def __init__(self,gripper,fn=None):
        if not isinstance(gripper,GripperInfo):
            raise ValueError("gripper needs to be a GripperInfo")
        self.gripper = gripper
        GripperInfo.register(gripper)
        self.objects = []
        self.object_to_grasps = dict()
        if fn is not None:
            self.load(fn)

    def print_info(self):
        print("Grasp database statistics:")
        print("Gripper:",self.gripper.name)
        for (o,gs) in self.object_to_grasps:
            print("Object",o,":",len(gs),"grasps")

    def load(self,fn):
        with open(fn,'r') as f:
            jsonobj = json.load(f)
        self.objects = jsonobj['objects']
        self.object_to_grasps = dict()
        for o,gs in jsonobj['object_to_grasps'].items():
            grasps = []
            for g in gs:
                gparsed = Grasp(None)
                gparsed.fromJson(g)
                grasps.append(gparsed)
            self.object_to_grasps[o] = grasps
        return True

    def save(self,fn):
        jsonobj = dict()
        jsonobj['objects'] = self.objects
        grasp_dict = dict()
        for o,gs in self.object_to_grasps.items():
            gsjson = [g.toJson() for g in gs]
            grasp_dict[o] = gsjson
        jsonobj['object_to_grasps'] = grasp_dict
        with open(fn,'w') as f:
            json.dump(jsonobj,f)
        return True

    def add_object(self,name):
        self.objects.append(name)
        self.object_to_grasps[name] = []

    def add_grasp(self,object,grasp):
        """Adds a new Grasp (assumed object centric) to the database for the
        given object.
        """
        if isinstance(grasp,Grasp):
            if object not in self.object_to_grasps:
            self.add_object(name)
            self.object_to_grasps[object].append(grasp)
        else:
            raise ValueError("grasp needs to be a Grasp")

    def get_sampler(self,robot):
        """Returns a GraspDatabaseSampler for this robot."""
        return GraspDatabaseSampler(robot,self.gripper,self.object_to_grasps)


class GraspDatabaseSampler(GraspSamplerBase):
    """A GraspSamplerBase subclass that will read from a dict of
    object-centric Grasp's.

    Args:
        robot (RobotModel): the robot.
        gripper (GripperInfo): the gripper
        object_to_grasps (dict of str -> list): for each object, gives a list
            of Grasp templates.
    """
    def __init__(self,robot,gripper,object_to_grasps):
        self._robot = robot
        self._gripper = gripper
        self._object_to_grasps = object_to_grasps
        self._target_object = None
        self._matching_object = None
        self._matching_xform = None
        self._grasp_index = None

    def object_match(self,object_source,object_target):
        """Determine whether object_source is a match to object_target.
        If they match, return a transform from the reference frame of
        object_source to object_target.  Otherwise, return None.

        Default implementation: determine whether the name of
        object_source matches object_target.name exactly.
        """
        if object_source != object_target.name:
            return se3.identity()
        return None

    def gripper(self):
        return self._gripper

    def init(self,scene,object,hints):
        """Checks for either an exact match or if object_match(o,object)
        exists"""
        if object.name in self._object_to_grasps:
            self._target_object = object
            self._matching_object = object.name
            self._matching_xform = se3.identity()
            self._grasp_index = 0
            return True
        for o,g in self._object_to_grasps.items():
            xform = self.object_match(o,object)
            if xform is not None:
                self._target_object = object
                self._matching_object = o
                self._matching_xform = xform
                self._grasp_index = 0
                return True
        return False

    def next(self):
        """Returns the next Grasp from the database."""
        if self._matching_object is None:
            return None
        grasps = self._object_to_grasps[self._matching_object]
        if self._grasp_index >= len(grasps):
            self._matching_object = None
            return None
        grasp = grasps[self._grasp_index]
        self._grasp_index += 1
        return grasp.get_transformed(se3.mul(self._target_object.getTransform(),self._matching_xform))
    
    def score(self):
        """Returns a score going from 1 to 0 as the number of grasps
        gets exhausted.
        """
        if self._matching_object is None: return 0
        grasps = self._object_to_grasps[self._matching_object]
        if self._grasp_index >= len(grasps):
            return 0
        return 1.0 - self._grasp_index/float(len(grasps))

    
from klampt import WorldModel,Geometry3D,GeometricPrimitive
from klampt.math import vectorops,se3,so3
from klampt.model import sensing
from klampt.model.trajectory import Trajectory
from klampt.io import resource
from klampt import vis
import sys
import random
import math
import os
import pickle
import numpy as np
sys.path.append("../common")
import grasp
from world_generator import ycb_objects,make_ycb_object,save_world
import problem1
import problem2

class AntipodalGrasp:
    """A structure containing information about antipodal grasps.
    
    Attributes:
        center (3-vector): the center of the fingers (object coordinates).
        axis (3-vector): the direction of the line through the
            fingers (object coordinates).
        approach (3-vector, optional): the direction that the fingers
            should move forward to acquire the grasp.
        finger_width (float, optional): the width that the gripper should
            open between the fingers.
        contact1 (ContactPoint, optional): a point of contact on the
            object.
        contact2 (ContactPoint, optional): another point of contact on the
            object.
    """
    def __init__(self,center,axis):
        self.center = center
        self.axis = axis
        self.approach = None
        self.finger_width = None
        self.contact1 = None
        self.contact2 = None

    def add_to_vis(self,name,color=(1,0,0,1)):
        finger_radius = 0.02
        if self.finger_width == None:
            w = 0.05
        else:
            w = self.finger_width*0.5+finger_radius
        a = vectorops.madd(self.center,self.axis,w)
        b = vectorops.madd(self.center,self.axis,-w)
        vis.add(name,Trajectory(milestones=[a,b]),color=color)
        if self.approach is not None:
            vis.add(name+"_approach",Trajectory(milestones=[self.center,vectorops.madd(self.center,self.approach,0.05)]),color=(1,0.5,0,1))
        normallen = 0.05
        if self.contact1 is not None:
            vis.add(name+"cp1",self.contact1.x,color=(1,1,0,1),size=0.01)
            vis.add(name+"cp1_normal",Trajectory(milestones=[self.contact1.x,vectorops.madd(self.contact1.x,self.contact1.n,normallen)]),color=(1,1,0,1))
        if self.contact2 is not None:
            vis.add(name+"cp2",self.contact2.x,color=(1,1,0,1),size=0.01)
            vis.add(name+"cp2_normal",Trajectory(milestones=[self.contact2.x,vectorops.madd(self.contact2.x,self.contact2.n,normallen)]),color=(1,1,0,1))

def grasp_from_contacts(contact1,contact2):
    """Helper: if you have two contacts, this returns an AntipodalGrasp"""
    d = vectorops.unit(vectorops.sub(contact2.x,contact1.x))
    grasp = AntipodalGrasp(vectorops.interpolate(contact1.x,contact2.x,0.5),d)
    grasp.finger_width = vectorops.distance(contact1.x,contact2.x)
    grasp.contact1 = contact1
    grasp.contact2 = contact2
    return grasp


class ImageBasedGraspPredictor:
    def __init__(self):
        #TODO: Problem 3B initialize your model here

    def generate(self,camera_xform,camera_intrinsics,color_image,depth_image,k='auto'):
        """Returns a list of k AntipodalGrasps in world space according
        to the given color and depth images.  Returns a list of k scores
        as well.
        """
        #TODO: implement me for Problem3B
        #tip: make your life easy and call code from problem2.py directly
        pixel = np.unravel_index(depth_image.argmin(), depth_image.shape)
        depth = depth_image[pixel]
        fx,fy,cx,cy = camera_intrinsics['fx'],camera_intrinsics['fy'],camera_intrinsics['cx'],camera_intrinsics['cy']
        y,x = pixel
        Z = depth
        X = (x - cx)/fx*Z
        Y = (y - cy)/fy*Z
        grasp_closest = AntipodalGrasp(se3.apply(camera_xform,[X,Y,Z]),[1,0,0])
        score = 1.0
        return [grasp_closest],[score]
    
def grasp_plan_main():
    world = WorldModel()
    world.readFile("camera.rob")
    robot = world.robot(0)
    sensor = robot.sensor(0)
    world.readFile("table.xml")
    xform = resource.get("table_camera_00.xform",type='RigidTransform')
    sensing.set_sensor_xform(sensor,xform)
    box = GeometricPrimitive()
    box.setAABB([0,0,0],[1,1,1])
    box = resource.get('table.geom','GeometricPrimitive',default=box,world=world,doedit='auto')
    bmin,bmax = [v for v in box.properties[0:3]],[v for v in box.properties[3:6]]
    nobj = 5
    obj_fns = []
    for o in range(nobj):
        fn = random.choice(ycb_objects)
        obj = make_ycb_object(fn,world)
        #TODO: you might want to mess with colors here too
        obj.appearance().setSilhouette(0)
        obj_fns.append(fn)
    for i in range(world.numTerrains()):
        #TODO: you might want to mess with colors here too
        world.terrain(i).appearance().setSilhouette(0)
    problem1.arrange_objects(world,obj_fns,bmin,bmax)

    intrinsics = dict()
    w = int(sensor.getSetting('xres'))
    h = int(sensor.getSetting('yres'))
    xfov = float(sensor.getSetting('xfov'))
    yfov = float(sensor.getSetting('yfov'))
    intrinsics['cx'] = w/2
    intrinsics['cy'] = h/2
    intrinsics['fx'] = math.tan(xfov*0.5)*h*2
    intrinsics['fy'] = math.tan(xfov*0.5)*h*2
    print("Intrinsics",intrinsics)
    planner = ImageBasedGraspPredictor()
    def do_grasp_plan(event=None,world=world,sensor=sensor,planner=planner,camera_xform=xform,camera_intrinsics=intrinsics):
        sensor.kinematicReset()
        sensor.kinematicSimulate(world,0.01)
        rgb,depth = sensing.camera_to_images(sensor)
        grasps,scores = planner.generate(camera_xform,camera_intrinsics,rgb,depth)
        for i,(g,s) in enumerate(zip(grasps,scores)):
            color = (1-s,s,0,1)
            g.add_to_vis("grasp_"+str(i),color=color)

    def resample_objects(event=None,world=world,obj_fns=obj_fns,bmin=bmin,bmax=bmax):
        problem1.arrange_objects(world,obj_fns,bmin,bmax)

    vis.add("world",world)
    vis.add("sensor",sensor)
    vis.addAction(do_grasp_plan,'Run grasp planner','p')
    vis.addAction(resample_objects,'Sample new arrangement','s')
    vis.run()

if __name__ == '__main__':
    grasp_plan_main()
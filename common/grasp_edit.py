"""Allows visual editing of grasps.
"""

from klampt import WorldModel
from klampt import vis
from klampt.math import vectorops,so3,se3
from klampt.model import ik
from grasp import Grasp
import time
import os
import json
import copy

def grasp_edit_ui(gripper,object,grasp=None):
    assert gripper.klampt_model is not None
    world = WorldModel()
    res = world.readFile(gripper.klampt_model)
    if not res:
        raise ValueError("Unable to load klampt model")
    robot = world.robot(0)
    base_link = robot.link(gripper.base_link)
    base_xform = base_link.getTransform()
    base_xform0 = base_link.getTransform()
    parent_xform = se3.identity()
    if base_link.getParent() >= 0:
        parent_xform = robot.link(base_link.getParent()).getTransform()
    if grasp is not None:
        base_xform = grasp.ik_constraint.closestMatch(*base_xform)
        base_link.setParentTransform(*se3.mul(se3.inv(parent_xform),base_xform))
        robot.setConfig(gripper.set_finger_config(robot.getConfig(),grasp.finger_config))
    q0 = robot.getConfig()
    grob = gripper.get_subrobot(robot)
    grob._links = [l for l in grob._links if l != gripper.base_link]

    #set up visualizer
    oldWindow = vis.getWindow()
    if oldWindow is None:
        oldWindow = vis.createWindow()
    vis.createWindow()
    vis.add("gripper",grob)
    vis.edit("gripper")
    vis.add("object",object)
    vis.add("base_xform",base_xform)
    vis.edit("base_xform")

    def make_grasp():
        return Grasp(ik.objective(base_link,R=base_xform[0],t=base_xform[1]),gripper.finger_links,gripper.get_finger_config(robot.getConfig()))

    #add hooks
    robot_appearances = [robot.link(i).appearance().clone() for i in range(robot.numLinks())]
    robot_shown = [True]
    def toggle_robot(arg=0,data=robot_shown):
        vis.lock()
        if data[0]:
            for i in range(robot.numLinks()):
                if i not in grob._links and i != gripper.base_link:
                    robot.link(i).appearance().setDraw(False)
            data[0] = False
        else:
            for i in range(robot.numLinks()):
                if i not in grob._links and i != gripper.base_link:
                    robot.link(i).appearance().set(robot_appearances[i])
            data[0] = True
        vis.unlock()
    def randomize():
        print("TODO")
    def reset():
        vis.lock()
        robot.setConfig(q0)
        base_link.setParentTransform(*se3.mul(se3.inv(parent_xform),base_xform0))
        vis.unlock()
        vis.add("base_xform",base_xform0)
        vis.edit("base_xform")
        vis.setItemConfig("gripper",grob.getConfig())
    def save():
        fmt = gripper.name+"_"+object.getName()+'_grasp_%d.json'
        ind = 0
        while os.path.exists(fmt%(ind,)):
            ind += 1
        fn = fmt%(ind,)
        g = make_grasp()
        print("Saving grasp to",fn)
        with open(fn,'w') as f:
            json.dump(g.toJson(),f)

    vis.addAction(toggle_robot,'Toggle show robot','v')
    vis.addAction(randomize,'Randomize','r')
    vis.addAction(reset,'Reset','0')
    vis.addAction(save,'Save to disk','s')

    def loop_setup():
        vis.show()

    def loop_callback():
        global base_xform
        xform = vis.getItemConfig("base_xform")
        base_xform = (xform[:9],xform[9:])
        vis.lock()
        base_link.setParentTransform(*se3.mul(se3.inv(parent_xform),base_xform))
        vis.unlock()
    
    def loop_cleanup():
        vis.show(False)

    vis.loop(setup=loop_setup,callback=loop_callback,cleanup=loop_cleanup)
    # this works with Linux/Windows, but not Mac
    # loop_setup()
    # while vis.shown():
    #     loop_callback()
    #     time.sleep(0.02)
    # loop_cleanup()

    g = make_grasp()
    #restore RobotModel
    base_link.setParentTransform(*se3.mul(se3.inv(parent_xform),base_xform0))
    vis.setWindow(oldWindow)
    return g

if __name__ == '__main__':
    import sys
    import os
    if len(sys.argv) < 3:
        print("USAGE: python grasp_edit gripper_name OBJECT_FILE [grasp file]")
        exit(1)
        
    from known_grippers import *
    g = GripperInfo.get(sys.argv[1])
    w = WorldModel()
    res = w.readFile(sys.argv[2])
    if not res:
        basename = os.path.splitext(os.path.basename(sys.argv[2]))[0]
        obj = w.makeRigidObject(basename)
        if not obj.loadFile(sys.argv[2]):
            if not obj.geometry().loadFile(sys.argv[2]):
                print("Unable to read object",sys.argv[2])
                exit(1)
    obj = w.rigidObject(0)
    grasp = None
    if len(sys.argv) >= 4:
        with open(sys.argv[3],'r') as f:
            jsonobj = json.load(f)
        grasp = Grasp()
        grasp.fromJson(jsonobj)
    grasp = grasp_edit_ui(g,obj,grasp)
    print("Resulting grasp:")
    print(json.dumps(grasp.toJson(), indent=2))

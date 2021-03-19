from klampt import WorldModel
from klampt.io import resource
from klampt import vis 
from klampt.model.trajectory import RobotTrajectory
from klampt.math import vectorops,so3,se3
import pick
import math
import random
import sys
sys.path.append("../common")
import grasp
import grasp_database
from known_grippers import *

PROBLEM = '2a'
#PROBLEM = '2b'
#PROBLEM = '2c'

if __name__ == '__main__':
    #load the robot / world file
    fn = "problem2.xml"
    world = WorldModel()
    res = world.readFile(fn)
    if not res:
        print("Unable to read file",fn)
        exit(0)

    #load the gripper info and grasp database
    source_gripper = robotiq_85
    target_gripper = robotiq_85_kinova_gen3
    db = grasp_database.GraspDatabase(source_gripper)
    if not db.load("../data/grasps/robotiq_85_sampled_grasp_db.json"):
        raise RuntimeError("Can't load grasp database?")

    robot = world.robot(0)
    #need to fix the spin joints somewhat
    qmin,qmax = robot.getJointLimits()
    for i in range(len(qmin)):
        if qmax[i] - qmin[i] > math.pi*2:
            qmin[i] = -float('inf')
            qmax[i] = float('inf')
    robot.setJointLimits(qmin,qmax)

    obj = world.rigidObject(0)

    #add the world elements individually to the visualization
    vis.add("world",world)
    qstart = resource.get("start.config",world=world)
    robot.setConfig(qstart)
    try:
        orig_grasps = db.object_to_grasps[obj.getName()]
    except Exception:
        raise RuntimeError("Can't get grasps for object {}".format(obj.getName()))

    #transform all the grasps to use the kinova arm gripper and object transform
    grasps = [grasp.get_transformed(obj.getTransform()).transfer(source_gripper,target_gripper) for grasp in orig_grasps]

    #this is just to do a nicer visualization... add the grasp and gripper to the visualizer
    if PROBLEM == '2a':
        w2 = WorldModel()
        w2.loadFile(source_gripper.klampt_model)
        source_gripper_model = w2.robot(0)
        grasp_index = 2
        grasp = grasps[grasp_index]
        Tgripper = grasp.ik_constraint.closestMatch(*se3.identity())
        source_gripper_model.setConfig(orig_grasps[grasp_index].set_finger_config(source_gripper_model.getConfig()))
        source_gripper.add_to_vis(source_gripper_model,animate=False,base_xform=Tgripper)
        grasp.add_to_vis("grasp")
    else:
        for i,grasp in enumerate(grasps):
            grasp.add_to_vis("grasp"+str(i))

    def planTriggered():
        global world,robot,obj,target_gripper,grasp,grasps
        qstart = robot.getConfig()
        if PROBLEM == '2a':
            res = pick.plan_pick_one(world,robot,obj,target_gripper,grasp)
        elif PROBLEM == '2b':
            res = pick.plan_pick_grasps(world,robot,obj,target_gripper,grasps)
        else:
            res = pick.plan_pick_multistep(world,robot,obj,target_gripper,grasps)
        if res is None:
            print("Unable to plan pick")
        else:
            (transit,approach,lift) = res
            traj = transit
            traj = traj.concat(approach,relative=True,jumpPolicy='jump')
            traj = traj.concat(lift,relative=True,jumpPolicy='jump')
            vis.add("traj",traj,endEffectors=[9])
            vis.animate(vis.getItemName(robot),traj)
        robot.setConfig(qstart)

    vis.addAction(planTriggered,"Plan grasp",'p')

    if PROBLEM == '2a':
        def shiftGrasp(amt):
            global grasp,grasps,grasp_index
            grasp_index += amt
            if grasp_index >= len(grasps):
                grasp_index = 0
            elif grasp_index < 0:
                grasp_index = len(grasps)-1
            print("Grasp",grasp_index)
            grasp = grasps[grasp_index]
            Tgripper = grasp.ik_constraint.closestMatch(*se3.identity())
            source_gripper_model.setConfig(orig_grasps[grasp_index].set_finger_config(source_gripper_model.getConfig()))
            source_gripper.add_to_vis(source_gripper_model,animate=False,base_xform=Tgripper)
            grasp.add_to_vis("grasp")

        vis.addAction(lambda :shiftGrasp(1),"Next grasp",'g')
    vis.run()

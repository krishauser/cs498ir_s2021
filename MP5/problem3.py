from klampt import WorldModel,Simulator
from klampt.io import resource
from klampt import vis 
from klampt.model.trajectory import Trajectory,RobotTrajectory,path_to_trajectory
from klampt.math import vectorops,so3,se3
import pick
import place
import math
import random
import time
import sys
sys.path.append("../common")
import grasp
import grasp_database
from known_grippers import *

PROBLEM = '3a'
#PROBLEM = '3b'
#PROBLEM = '3c'

PHYSICS_SIMULATION = False  #not implemented correctly yet

if __name__ == '__main__':
    #load the robot / world file
    if PROBLEM=='3a':
        fn = "problem3a.xml"
    else:
        fn = "problem3b.xml"
    world = WorldModel()
    res = world.readFile(fn)
    if not res:
        print("Unable to read file",fn)
        exit(0)
    for i in range(world.numRigidObjects()):
        obj = world.rigidObject(i)
        #this will perform a reasonable center of mass / inertia estimate
        m = obj.getMass()
        m.estimate(obj.geometry(),mass=0.454,surfaceFraction=0.2)
        obj.setMass(m)

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

    #these describe the box dimensions
    goal_bounds = [(-0.08,-0.68,0.4),(0.48,-0.32,0.5)]

    #you can play around with which object you select for problem 3b...
    obj = world.rigidObject(0)
    gripper_link = robot.link(9)
    if PROBLEM=='3a':
        qstart = resource.get("transfer_start.config",world=world)
        Tobj = resource.get("transfer_obj_pose.xform",world=world,referenceObject=obj)
        obj.setTransform(*Tobj)
        qgoal = resource.get("transfer_goal.config",world=world)
        robot.setConfig(qstart)
        Tobject_grasp = se3.mul(se3.inv(gripper_link.getTransform()),Tobj)
        vis.add("goal",qgoal,color=(1,0,0,0.5))
        robot.setConfig(qgoal)
        Tobj_goal = se3.mul(robot.link(9).getTransform(),Tobject_grasp)
        vis.add("Tobj_goal",Tobj_goal)
        robot.setConfig(qstart)
    elif PROBLEM == '3b':
        #determine a grasp pose via picking
        orig_grasps = db.object_to_grasps[obj.getName()]
        grasps = [grasp.get_transformed(obj.getTransform()).transfer(source_gripper,target_gripper) for grasp in orig_grasps]
        res = pick.plan_pick_grasps(world,robot,obj,target_gripper,grasps)
        if res is None:
            raise RuntimeError("Couldn't find a pick plan?")
        transit,approach,lift = res
        qgrasp = approach.milestones[-1]
        #get the grasp transform
        robot.setConfig(qgrasp)
        Tobj = obj.getTransform()
        Tobject_grasp = se3.mul(se3.inv(gripper_link.getTransform()),Tobj)

        robot.setConfig(lift.milestones[-1])
        qstart = robot.getConfig()
        obj.setTransform(*se3.mul(gripper_link.getTransform(),Tobject_grasp))
    else:
        qstart = robot.getConfig()

    #add the world elements individually to the visualization
    vis.add("world",world)

    solved_trajectory = None
    trajectory_is_transfer = None
    next_item_to_pick = 0
    def planTriggered():
        global world,robot,obj,target_gripper,solved_trajectory,trajectory_is_transfer,Tobject_grasp,obj,next_item_to_pick,qstart
        if PROBLEM == '3a':
            robot.setConfig(qstart)
            res = place.transfer_plan(world,robot,qgoal,obj,Tobject_grasp)
            if res is None:
                print("Unable to plan transfer")
            else:
                traj = RobotTrajectory(robot,milestones=res)
                vis.add("traj",traj,endEffectors=[gripper_link.index])
                solved_trajectory = traj
            robot.setConfig(qstart)
        elif PROBLEM == '3b':
            res = place.plan_place(world,robot,obj,Tobject_grasp,target_gripper,goal_bounds)
            if res is None:
                print("Unable to plan place")
            else:
                (transfer,lower,retract) = res
                traj = transfer
                traj = traj.concat(lower,relative=True,jumpPolicy='jump')
                traj = traj.concat(retract,relative=True,jumpPolicy='jump')
                vis.add("traj",traj,endEffectors=[gripper_link.index])
                solved_trajectory = traj
            robot.setConfig(qstart)
        else:
            robot.setConfig(qstart)
            obj = world.rigidObject(next_item_to_pick)
            Tobj0 = obj.getTransform()
            print("STARTING TO PICK OBJECT",obj.getName())
            orig_grasps = db.object_to_grasps[obj.getName()]
            grasps = [grasp.get_transformed(obj.getTransform()).transfer(source_gripper,target_gripper) for grasp in orig_grasps]
            res = pick.plan_pick_multistep(world,robot,obj,target_gripper,grasps)
            if res is None:
                print("Unable to plan pick")
            else:
                transit,approach,lift = res

                qgrasp = approach.milestones[-1]
                #get the grasp transform
                robot.setConfig(qgrasp)
                Tobj = obj.getTransform()
                Tobject_grasp = se3.mul(se3.inv(gripper_link.getTransform()),Tobj)
                
                robot.setConfig(lift.milestones[-1])
                res = place.plan_place(world,robot,obj,Tobject_grasp,target_gripper,goal_bounds)
                if res is None:
                    print("Unable to plan place")
                else:
                    (transfer,lower,retract) = res
                    trajectory_is_transfer = Trajectory()
                    trajectory_is_transfer.times.append(0)
                    trajectory_is_transfer.milestones.append([0])
                    traj = transit
                    traj = traj.concat(approach,relative=True,jumpPolicy='jump')
                    trajectory_is_transfer.times.append(traj.endTime())
                    trajectory_is_transfer.times.append(traj.endTime())
                    trajectory_is_transfer.milestones.append([0])
                    trajectory_is_transfer.milestones.append([1])
                    traj = traj.concat(lift,relative=True,jumpPolicy='jump')
                    traj = traj.concat(transfer,relative=True,jumpPolicy='jump')
                    traj = traj.concat(lower,relative=True,jumpPolicy='jump')
                    trajectory_is_transfer.times.append(traj.endTime())
                    trajectory_is_transfer.times.append(traj.endTime())
                    trajectory_is_transfer.milestones.append([1])
                    trajectory_is_transfer.milestones.append([0])
                    traj = traj.concat(retract,relative=True,jumpPolicy='jump')
                    trajectory_is_transfer.times.append(traj.endTime())
                    trajectory_is_transfer.milestones.append([0])
                    solved_trajectory = traj

                    obj.setTransform(*Tobj0)

                    vis.add("traj",traj,endEffectors=[gripper_link.index])
            robot.setConfig(qstart)

    vis.addAction(planTriggered,"Plan grasp",'p')

    executing_plan = False
    execute_start_time = None
    def executePlan():
        global solved_trajectory,trajectory_is_transfer,executing_plan,execute_start_time
        if solved_trajectory is None:
            return
        executing_plan = True
        if PHYSICS_SIMULATION:
            execute_start_time = 0
            solved_trajectory = path_to_trajectory(solved_trajectory,timing='robot',smoothing=None)
            solved_trajectory.times = [10*t for t in solved_trajectory.times]
        else:
            execute_start_time = time.time()

    vis.addAction(executePlan,"Execute plan",'e')

    sim = Simulator(world)
    sim_dt = 0.02
    was_grasping = False
    def loop_callback():
        global was_grasping,Tobject_grasp,solved_trajectory,trajectory_is_transfer,executing_plan,execute_start_time,qstart,next_item_to_pick
        if not executing_plan:
            return
        if PHYSICS_SIMULATION:
            execute_start_time += sim_dt
            t = execute_start_time
        else:
            t = time.time()-execute_start_time
        vis.addText("time","Time %.3f"%(t),position=(10,10))
        if PROBLEM == '3c':
            qstart = solved_trajectory.eval(t)
            if PHYSICS_SIMULATION:
                #sim.controller(0).setPIDCommand(qstart,solved_trajectory.deriv(t))
                #sim.controller(0).setMilestone(qstart)
                sim.controller(0).setLinear(qstart,sim_dt*5)
                sim.simulate(sim_dt)
                sim.updateWorld()
            else :
                robot.setConfig(qstart)
                during_transfer = trajectory_is_transfer.eval(t)[0]
                if not was_grasping:
                    #pick up object
                    Tobject_grasp = se3.mul(se3.inv(gripper_link.getTransform()),obj.getTransform())
                    was_grasping = True
                if during_transfer:
                    obj.setTransform(*se3.mul(robot.link(9).getTransform(),Tobject_grasp))
                else:
                    was_grasping = False
            if t > solved_trajectory.duration():
                executing_plan = False
                solved_trajectory = None
                next_item_to_pick += 1
        else:
            robot.setConfig(solved_trajectory.eval(t,'loop'))
            obj.setTransform(*se3.mul(robot.link(9).getTransform(),Tobject_grasp))
        
    vis.loop(callback=loop_callback)

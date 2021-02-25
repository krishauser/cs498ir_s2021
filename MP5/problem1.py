from klampt import WorldModel
from klampt.io import resource
from klampt import vis 
from klampt.model import trajectory
import planning
import math

PROBLEM = '1a'
#PROBLEM = '1b'

if __name__ == '__main__':
    #load the robot / world file
    fn = "problem1.xml"
    world = WorldModel()
    res = world.readFile(fn)
    if not res:
        print("Unable to read file",fn)
        exit(0)

    robot = world.robot(0)
    #need to fix the spin joints somewhat
    qmin,qmax = robot.getJointLimits()
    for i in range(len(qmin)):
        if qmax[i] - qmin[i] > math.pi*2:
            qmin[i] = -float('inf')
            qmax[i] = float('inf')
    robot.setJointLimits(qmin,qmax)

    qstart = resource.get("start.config",world=world)

    #add the world elements individually to the visualization
    vis.add("world",world)
    vis.add("start",qstart,color=(0,1,0,0.5))
    qgoal = resource.get("goal.config",world=world)
    #qgoal = resource.get("goal_easy.config",world=world)
    robot.setConfig(qgoal)
    vis.edit(vis.getItemName(robot))
    def planTriggered():
        global world,robot
        qtgt = vis.getItemConfig(vis.getItemName(robot))
        qstart = vis.getItemConfig("start")
        robot.setConfig(qstart)
        if PROBLEM == '1a':
            path = planning.feasible_plan(world,robot,qtgt)
        else:
            path = planning.optimizing_plan(world,robot,qtgt)
            
        if path is not None:
            ptraj = trajectory.RobotTrajectory(robot,milestones=path)
            ptraj.times = [t / len(ptraj.times) * 5.0 for t in ptraj.times]
            #this function should be used for creating a C1 path to send to a robot controller
            traj = trajectory.path_to_trajectory(ptraj,timing='robot',smoothing=None)
            #show the path in the visualizer, repeating for 60 seconds
            vis.animate("start",traj)
            vis.add("traj",traj,endeffectors=[9])

    vis.addAction(planTriggered,"Plan to target",'p')
    vis.run()

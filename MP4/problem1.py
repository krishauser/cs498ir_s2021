from klampt import WorldModel,RobotModel,RobotModelLink,Geometry3D,GeometricPrimitive
from klampt.math import vectorops,so3,se3
from klampt.model import ik
from klampt.io import numpy_convert
from klampt import vis
import math
import numpy as np
import sys
import os
if sys.path[-1] != "../common":
    sys.path.append("../common")
import merge_geometry

lower_corner = (-1.1,-1.1,-0.7)
upper_corner = (1.1,1.1,1.5)

def collision_free(robot,obstacles):
    #TODO: do environment collision checking
    if robot.selfCollides():
        return False
    return True

def calculate_workspace_free(robot,obstacles,end_effector,point_local):
    """Calculate the reachable workspace of the end effector point whose coordinates
    are `point_local` on link `end_effector`.  Ensure that the robot is collision
    free with itself and with environment obstacles
    """
    global lower_corner,upper_corner
    resolution = (20,20,20)
    cellsize = vectorops.div(vectorops.sub(upper_corner,lower_corner),resolution)
    invcellsize = vectorops.div(resolution,vectorops.sub(upper_corner,lower_corner))
    
    reachable = np.zeros(resolution)
    #TODO: your code here
    feasible = collision_free(robot,obstacles)
    if feasible:
        wp = end_effector.getWorldPosition(point_local)
        index = [int(math.floor(v)) for v in vectorops.mul(vectorops.sub(wp,lower_corner),invcellsize)]
        if all(i>=0 and i<r for (i,r) in zip(index,resolution)):
            reachable[tuple(index)] = 1.0
    return reachable

def calculate_workspace_axis(robot,obstacles,end_effector,point_local,axis_local,axis_world):
    global lower_corner,upper_corner
    resolution = (15,15,15)
    cellsize = vectorops.div(vectorops.sub(upper_corner,lower_corner),resolution)
    invcellsize = vectorops.div(resolution,vectorops.sub(upper_corner,lower_corner))
    
    reachable = np.zeros(resolution)
    #TODO: your code here
    return reachable

w = WorldModel()
w.readFile("../data/robots/kinova_gen3_7dof.urdf")
w.readFile("../data/terrains/plane.env")
robot = w.robot(0)
obstacles = [w.terrain(0)]

ee_link = 'EndEffector_Link'
ee_local_pos = (0.15,0,0)
ee_local_axis = (1,0,0)

#put a "pen" geometry on the end effector
gpen = Geometry3D()
res = gpen.loadFile("../data/objects/cylinder.off")
assert res
gpen.scale(0.01,0.01,0.15)
gpen.rotate(so3.rotation((0,1,0),math.pi/2))
robot.link(7).geometry().setCurrentTransform(*se3.identity())
gpen.transform(*robot.link(8).getParentTransform())
robot.link(7).geometry().set(merge_geometry.merge_triangle_meshes(gpen,robot.link(7).geometry()))
robot.setConfig(robot.getConfig())

def show_workspace(grid):
    vis.add("world",w)
    res = numpy_convert.from_numpy((lower_corner,upper_corner,grid),'VolumeGrid')
    g_workspace = Geometry3D(res)
    g_surface = g_workspace.convert('TriangleMesh',0.5)
    if g_surface.numElements() != 0:
        vis.add("reachable_boundary",g_surface,color=(1,1,0,0.5))
    else:
        print("Nothing reachable?")

    Tee = robot.link(ee_link).getTransform()
    gpen.setCurrentTransform(*Tee)
    box = GeometricPrimitive()
    box.setAABB(lower_corner,upper_corner)
    gbox = Geometry3D(box)
    #show this if you want to debug the size of the grid domain
    #vis.add("box",gbox,color=(1,1,1,0.2))

    vis.add("pen tip",se3.apply(Tee,ee_local_pos))
    vis.loop()

def problem_1a():
    grid = calculate_workspace_free(robot,obstacles,robot.link(ee_link),ee_local_pos)
    np.save('problem1a.npy',grid)
    show_workspace(grid)

def show_problem_1a():
    grid = np.load('problem1a.npy')
    show_workspace(grid)

def problem_1b():
    grid = calculate_workspace_axis(robot,obstacles,robot.link(ee_link),ee_local_pos,ee_local_axis,(0,0,1))
    np.save('problem1b.npy',grid)
    show_workspace(grid)

def show_problem_1b():
    grid = np.load('problem1b.npy')
    show_workspace(grid)

def problem_1c():
    Wup = np.load('problem1b.npy')
    Wdown = calculate_workspace_axis(robot,obstacles,robot.link(ee_link),ee_local_pos,ee_local_axis,(0,0,-1))
    np.save('problem1c_down.npy',Wdown)
    Wside = calculate_workspace_axis(robot,obstacles,robot.link(ee_link),ee_local_pos,ee_local_axis,(1,0,0))
    np.save('problem1c_forward.npy',Wside)
    
    #TODO: replace this with your own visualization.
    show_workspace(Wup)
    show_workspace(Wdown)
    show_workspace(Wside)

def show_problem_1c():
    Wup = np.load('problem1b.npy')
    Wdown = np.load('problem1c_down.npy')
    Wside = np.load('problem1c_forward.npy')

    #TODO: replace this with your own visualization
    show_workspace(Wup)
    show_workspace(Wdown)
    show_workspace(Wside)


if __name__=='__main__':
    problem_1a()
    #show_problem_1a()
    #problem_1b()
    #show_problem_1b()
    #problem_1c()
    #show_problem_1c()
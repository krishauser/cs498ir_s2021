from klampt import WorldModel,RobotModel,RobotModelLink,Geometry3D,Simulator
from klampt.math import vectorops,so2,so3,se3
from klampt.model import ik
from klampt.model.trajectory import Trajectory
from klampt.control.robotinterface import RobotInterfaceBase
from klampt.control.robotinterfaceutils import RobotInterfaceCompleter
from klampt.control.simrobotinterface import SimPositionControlInterface,KinematicSimControlInterface
from klampt.control.interop import RobotInterfacetoVis
from klampt import vis
import math
import time
import sys
sys.path.append("../common")
import merge_geometry
import svgimport
# import normals
# import stable_faces
# import grasp
# import gripper
# import known_grippers

def get_paths_svg(fn,add_to_vis=True):
    trajs,attrs = svgimport.svg_import_trajectories(fn,center=True,want_attributes=True)
    if add_to_vis:
        for i,(traj,attr) in enumerate(zip(trajs,attrs)):
            name = attr.get('name',"path %d"%i)
            vis.add(name,traj)
            for a,v in attr.items():
                if a != 'name':
                    vis.setAttribute(name,a,v)
    return trajs

def get_paths(add_to_vis=True):
    """Returns a list of Trajectory objects describing the desired end effector
    motion in 3D space"""
    #TODO: design your trajectories here or use get_paths_svg instead
    #return get_paths_svg("mypaths.svg",add_to_vis)

    trajs = []
    trajs.append(Trajectory([0,1],[[0,0,0],[0.2,0,0]]))
    if add_to_vis:
        for i,traj in enumerate(trajs):
            vis.add("path %d"%i,traj)
    return trajs


spin_joints = [1,3,5,7]

def normalize_config(config,reference):
    """For a configuration that may have wildly incorrect continuous-rotation
    joints, this calculates a version of config that represents the same
    configuration but is within +/ pi radians of the reference configuration.
    """
    res = [v for v in config]
    for i in spin_joints:
        if abs(config[i]-reference[i]) > math.pi:
            d = so2.diff(config[i]%(math.pi*2),reference[i])
            res[i] = reference[i] + d
    return res

class DrawingController:
    """A controller to execute cartesian trajectory drawings.

    External caller will call advance(dt) multiple times.  This function should
    update internal state and call self.robot_controller.X() to send commands
    to the simulated robot.

    Attributes:
        robot_controller (RobotInterfaceBase): the interface to the robot's
            controller.  Commands to the controller are non-blocking -- you send
            a command, and to determine when it's done, you must monitor for the
            command's completion.
        robot_model (RobotModel): a model for you to do anything with. This doesn't
            actually move the robot!
        qhome (configuration): a home configuration.
        end_effector (RobotModelLink): the end effector link
        pen_local_tip (3-vector): local coordinates of the pen tip
        pen_local_axis (3-vector): local coordinates of the pen (forward) axis
        plane_axis (3-vector): the outward normal of the plane on which the
            trajectories lie
        lift_amount (float): a distance to lift the pen between tracing.

    """
    def __init__(self,robot_controller,
            robot_model,qhome,paths,
            end_effector,pen_local_tip,pen_local_axis,plane_axis,lift_amount):
        self.robot_model = robot_model
        self.qhome = qhome
        self.paths = paths
        if isinstance(end_effector,RobotModelLink):
            self.end_effector = end_effector
        else:
            self.end_effector = robot_model.link(end_effector)
        self.pen_local_tip = pen_local_tip
        self.pen_local_axis = pen_local_axis
        self.plane_axis = plane_axis
        self.lift_amount = lift_amount
        self.robot_controller = robot_controller
        #store some internal state here
        self.state = 'moveto'
        self.path_index = 0
        self.path_progress = 0
        self.wait = 0


    def advance(self,dt):
        """The state machine logic of advance() should:

        1. move the pen to an approach point near the start point of path 0
          (The approach point should be lifted `lift_amount` distance away from the plane)
        2. lower the pen to the start point of path 0
        3. trace path 0, keeping the pen's local axis paralle to the plane axis
        4. raise the pen by lift_amount
        5. move in a straight joint space path to an approach point near path 1's start point
        6. ...likewise, repeat steps 1-4 for the remaining paths
        7. move back to qhome along a straight joint space path.

        dt is the time step. 
        """
        model = self.robot_model
        controller = self.robot_controller
        qcur = controller.configToKlampt(controller.commandedPosition())
        #TODO: implement me
        if not controller.isMoving():
            self.wait += 1
            if self.wait < 50:
                return
            self.wait = 0
            #done with prior motion
            model.randomizeConfig()
            model.setConfig(normalize_config(model.getConfig(),qcur))  #this is needed to handle the funky spin joints
            q = model.getConfig()
            duration = 2
            controller.setPiecewiseLinear([duration],[controller.configFromKlampt(q)])
            #setPosition does an immediate position command
            #self.robot_controller.setPosition(self.robot_controller.configFromKlampt(q))
        return


def run_cartesian(world,paths):
    sim_world = world
    sim_robot = world.robot(0)
    vis.add("world",world)
    planning_world = world.copy()
    planning_robot = planning_world.robot(0)
    sim = Simulator(sim_world)

    robot_controller = RobotInterfaceCompleter(KinematicSimControlInterface(sim_robot))
    #TODO: Uncomment this if you'd like to test in the physics simulation
    #robot_controller = RobotInterfaceCompleter(SimPositionControlInterface(sim.controller(0),sim))
    if not robot_controller.initialize():
        raise RuntimeError("Can't connect to robot controller")

    ee_link = 'EndEffector_Link'
    ee_local_pos = (0.15,0,0)
    ee_local_axis = (1,0,0)
    lifth = 0.05
    drawing_controller = DrawingController(robot_controller,planning_robot,
        planning_robot.getConfig(),paths,
        ee_link,ee_local_pos,ee_local_axis,(0,0,1),lifth)

    controller_vis = RobotInterfacetoVis(robot_controller)

    #note: this "storage" argument is only necessary for jupyter to keep these around and not destroy them once main() returns
    def callback(robot_controller=robot_controller,drawing_controller=drawing_controller,
        storage=[sim_world,planning_world,sim,controller_vis]):
        start_clock = time.time()
        dt = 1.0/robot_controller.controlRate()

        #advance the controller        
        robot_controller.startStep()
        if robot_controller.status() == 'ok':
            drawing_controller.advance(dt)
            vis.addText("Status",drawing_controller.state,(10,20))
        robot_controller.endStep()
        
        #update the visualization
        sim_robot.setConfig(robot_controller.configToKlampt(robot_controller.sensedPosition()))
        Tee=sim_robot.link(ee_link).getTransform()
        vis.add("pen axis",Trajectory([0,1],[se3.apply(Tee,ee_local_pos),se3.apply(Tee,vectorops.madd(ee_local_pos,ee_local_axis,lifth))]),color=(0,1,0,1))

        controller_vis.update()

        cur_clock = time.time()
        duration = cur_clock - start_clock
        time.sleep(max(0,dt-duration))
    vis.loop(callback=callback)


def main():
    w = WorldModel()
    w.readFile("../data/robots/kinova_gen3_7dof.urdf")
    robot = w.robot(0)
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
    
    trajs = get_paths()
    #place on a reasonable height and offset
    tableh = 0.1
    for traj in trajs:
        for m in traj.milestones:
            m[0] = m[0]*0.4 + 0.35
            m[1] = m[1]*0.4
            m[2] = tableh
            if len(m) == 6:
                m[3] *= 0.4
                m[4] *= 0.4

    return run_cartesian(w,trajs)

if __name__ == '__main__':
    main()
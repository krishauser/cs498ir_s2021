from klampt import WorldModel,RobotModel,RobotModelLink,Geometry3D,Simulator
from klampt.math import vectorops,so3,se3
from klampt.model import ik
from klampt.model.trajectory import Trajectory
from klampt.io import resource
from klampt.control.robotinterface import RobotInterfaceBase
from klampt.control.robotinterfaceutils import RobotInterfaceCompleter
from klampt.control.simrobotinterface import SimPositionControlInterface,KinematicSimControlInterface
from klampt.control.interop import RobotInterfacetoVis
from klampt import vis
import math
import time
import sys
from klampt.vis.visualization import _backend
sys.path.append("../common")
import merge_geometry

class CircleController:
    """A controller to execute cartesian circles.  The pen tip AND the axis
    should revolve around a vertical axis so that the pen traces out a circle of
    the specified radius.

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
        end_effector (RobotModelLink): the end effector link
        pen_local_tip (3-vector): local coordinates of the pen tip
        pen_local_axis (3-vector): local coordinates of the pen (forward) axis
        radius (float): radius of the circle
        period (float): # of seconds per cycle
        time (float): a time accumulator
    """
    def __init__(self,robot_controller,
            robot_model,end_effector,
            pen_local_tip,pen_local_axis,
            radius,period):
        self.robot_controller = robot_controller
        self.robot_model = robot_model
        if isinstance(end_effector,RobotModelLink):
            self.end_effector = end_effector
        else:
            self.end_effector = robot_model.link(end_effector)
        self.pen_local_tip=pen_local_tip
        self.pen_local_axis=pen_local_axis

        self.radius = radius
        self.period = period
        self.time = 0

    def advance(self,dt):
        """Move the end effector in a circle, maintaining its orientation

        dt is the time step. 
        """
        model = self.robot_model
        controller = self.robot_controller
        qcur = controller.configToKlampt(controller.commandedPosition())
     
        # TODO: implement me
        if not controller.isMoving():
            #done with prior motion
            model.randomizeConfig()
            q = model.getConfig()
            duration = 2
            controller.setPiecewiseLinear([duration],[controller.configFromKlampt(q)])
            #setPosition does an immediate position command
            #self.robot_controller.setPosition(self.robot_controller.configFromKlampt(q))
        return




def run_simulation(world):
    value = resource.get('start.config',default=world.robot(0).getConfig(),world=world)
    world.robot(0).setConfig(value)

    sim_world = world
    sim_robot = world.robot(0)
    vis.add("world",world)
    planning_world = world.copy()
    planning_robot = planning_world.robot(0)
    sim = Simulator(sim_world)

    robot_controller = RobotInterfaceCompleter(KinematicSimControlInterface(sim_robot))
    #TODO: Uncomment this when you are ready for testing in the physics simulation
    #robot_controller = RobotInterfaceCompleter(SimPositionControlInterface(sim.controller(0),sim))
    if not robot_controller.initialize():
        raise RuntimeError("Can't connect to robot controller")

    ee_link = 'EndEffector_Link'
    ee_local_pos = (0.15,0,0)
    ee_local_axis = (1,0,0)
    lifth = 0.05
    drawing_controller = CircleController(robot_controller,planning_robot,ee_link,
        ee_local_pos,ee_local_axis,
        radius=0.05,period=5.0)

    controller_vis = RobotInterfacetoVis(robot_controller)
    trace = Trajectory()

    #note: this "storage" argument is only necessary for jupyter to keep these around and not destroy them once main() returns
    def callback(robot_controller=robot_controller,drawing_controller=drawing_controller,trace=trace,
        storage=[sim_world,planning_world,sim,controller_vis]):
        start_clock = time.time()
        dt = 1.0/robot_controller.controlRate()

        #advance the controller        
        robot_controller.startStep()
        drawing_controller.advance(dt)
        robot_controller.endStep()

        #update the visualization
        sim_robot.setConfig(robot_controller.configToKlampt(robot_controller.sensedPosition()))
        Tee=sim_robot.link(ee_link).getTransform()
        wp = se3.apply(Tee,ee_local_pos)

        if len(trace.milestones) == 0 or vectorops.distance(wp,trace.milestones[-1]) > 0.01:
            trace.milestones.append(wp)
            trace.times.append(0)
            if len(trace.milestones)==2:
                vis.add("trace",trace,color=(0,1,1,1),pointSize=0)
            if len(trace.milestones) > 200:
                trace.milestones = trace.milestones[-100:]
                trace.times = trace.times[-100:]
            if len(trace.milestones)>2:
                if _backend=='IPython':
                    vis.remove("trace")
                    vis.add("trace",trace,color=(0,1,1,1),pointSize=0)
                else:
                    vis.dirty("trace")

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

    return run_simulation(w)

if __name__ == '__main__':
    main()
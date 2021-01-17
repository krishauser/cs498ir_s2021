from klampt.math import vectorops,so3,se3
from klampt.model.subrobot import SubRobotModel
from klampt import WorldModel,Geometry3D
import copy

class GripperInfo:
    """Stores basic information describing a gripper and its mounting on
    a robot.

    For a vacuum-type gripper,

    - center should be set to middle of the vacuum at a "tight" seal.
    - primary_axis should be set to the outward direction from the vacuum.
    - opening_span should be set to the diameter of the vacuum seal
    - finger_length should be set to the amount the vacuum should lower from 
      an offset away from the object (the length of the vacuum seal)

    For a parallel-jaw gripper,

    - center should be set to the deepest point within the gripper.
    - primary_axis should be set to the "down" direction for a top-down grasp
      (away from the wrist, usually). 
    - secondary_axis should be set to the axis along which the fingers close
      / open (either sign is OK).
    - finger_length should be set to the distance along primary_axis from center to
      the tips of the gripper's fingers.
    - finger_width should be set to the width of the fingers.
    - finger_depth should be set to the thickness of the fingers along
      secondary_axis
    - opening_span should be set to the fingers' maximum separation.

    For a multi-finger gripper, these elements are less important, but can
    help with general heuristics.

    - center should be set to a point on the "palm"
    - primary_axis should be set to a open direction away from the wrist.
    - secondary_axis should be set to the axis along which the fingers
      close / open in a power grasp (either sign is OK).
    - finger_length should be set to approximately the length of each finger.
    - finger_width should be set to approximately the width of each finger.
    - finger_depth should be set to approximately the thickness of each finger.
    - opening_span should be set to the width of the largest object grippable.

    Attributes:
        name (str): the gripper name
        base_link (int): the index of the gripper's base
        finger_links (list of int): the indices of the gripper's fingers.
        finger_drivers (list of int): the driver indices of the gripper's
            fingers. Can also be a list of list of ints if each finger joint
            can be individually actuated.
        type (str, optional): Specifies the type of gripper. Can be 'vacuum',
            'parallel', 'wrapping', or None (unknown)
        center (list of 3 floats, optional): The "palm" of the gripper.
        primary_axis (list of 3 floats, optional): The local axis of the
            gripper that opposes the "typical" load.  (Unit vector in the
            opposite direction of the load)
        secondary_axis (list of 3 floats, optional): The local axis of the
            gripper perpendicular to the primary that dictates the direction
            of the fingers opening and closing
        finger_length,finger_width,finger_depth (float, optional): dimensions
            of the fingers.
        opening_span (float, optional): the maximum opening span of the gripper.
        closed_config (list of floats, optional): the "logical closed" gripper
            finger config.
        open_config (list of floats, optional): the "logical open" gripper
            finger config.
        klampt_model (str, optional): the Klamp't .rob or .urdf model to which
            this refers to.  Note: this is not necessarily a model of just 
            gripper.  Suggest creating a GripperInfo with name
            NAME+'-fixed' or NAME+'-floating' for automatic
            grasp generation algorithms.
    """

    all_grippers = dict()

    @staticmethod
    def register(gripper):
        GripperInfo.all_grippers[gripper.name] = gripper

    @staticmethod
    def get(name):
        return GripperInfo.all_grippers.get(name,None)


    @staticmethod
    def mounted(gripper,klampt_model,base_link,name=None,
        register=True):
        """From a standalone gripper, return a GripperInfo such that the link
        indices are shifted onto a new robot model.
        """
        if name is None:
            name = gripper.name + "_mounted"
        w = WorldModel()
        w.enableGeometryLoading(False)
        res = w.readFile(klampt_model)
        if not res:
            raise IOError("Unable to load file "+str(klampt_model))
        robot = w.robot(0)
        w.enableGeometryLoading(True)
        if isinstance(base_link,str):
            base_link = robot.link(base_link).index
        shifted_finger_links = [l+base_link for l in gripper.finger_links]
        mount_driver = -1
        for i in range(robot.numDrivers()):
            ls = robot.driver(i).getAffectedLinks()
            if any(l in ls for l in shifted_finger_links):
                mount_driver = i
                break
        if mount_driver < 0:
            raise RuntimeError("Can't find the base driver for the mounted gripper?")
        shifted_finger_drivers = [l+mount_driver for l in gripper.finger_drivers]
        res = copy.copy(gripper)
        res.name = name
        res.base_link = base_link
        res.klampt_model = klampt_model
        res.finger_links = shifted_finger_links
        res.finger_drivers = shifted_finger_drivers
        if register:
            GripperInfo.register(res)
        return res
        

    def __init__(self,name,base_link,finger_links=None,finger_drivers=None,
                    type=None,center=None,primary_axis=None,secondary_axis=None,finger_length=None,finger_width=None,finger_depth=None,opening_span=None,
                    closed_config=None,open_config=None,
                    klampt_model=None,
                    register=True):
        self.name = name
        self.base_link = base_link
        self.finger_links = finger_links if finger_links is not None else []
        self.finger_drivers = finger_drivers if finger_drivers is not None else []
        self.type=type
        self.center = center
        self.primary_axis = primary_axis
        self.secondary_axis = secondary_axis
        self.finger_length = finger_length
        self.finger_width = finger_width
        self.finger_depth = finger_depth
        self.opening_span = opening_span
        self.closed_config = closed_config 
        self.open_config = open_config
        self.klampt_model = klampt_model
        if register:
            GripperInfo.register(self)

    def partway_open_config(self,amount):
        """Returns a finger configuration partway open, with amount in the
        range [0 (closed),1 (fully open)].
        """
        if self.closed_config is None or self.open_config is None:
            raise ValueError("Can't get an opening configuration on a robot that does not define it")
        return vectorops.interpolate(self.closed_config,self.open_config,amount)

    def set_finger_config(self,qrobot,qfinger):
        """Given a full robot config qrobot, returns a config but with the finger
        degrees of freedom fixed to qfinger.
        """
        assert len(qfinger) == len(self.finger_links)
        qf = [v for v in qrobot]
        for (i,v) in zip(self.finger_links,qfinger):
            qf[i] = v
        return qf

    def get_finger_config(self,qrobot):
        """Given a full robot config qrobot, returns a finger config."""
        return [qrobot[i] for i in self.finger_links]

    def descendant_links(self,robot):
        """Returns all links under the base link.  This may be different
        from finger_links if there are some frozen DOFs and you prefer
        to treat a finger configuration as only those DOFS for the active
        links.
        """
        descendants = [False]*robot.numLinks()
        descendants[self.base_link] = True
        for i in range(robot.numLinks()):
            if descendants[robot.link(i).getParent()]:
                descendants[i] = True
        return [i for (i,d) in enumerate(descendants) if d]

    def get_subrobot(self,robot,all_descendants=True):
        """Returns the SubRobotModel of the gripper given a RobotModel.

        If some of the links belonging to the gripper are frozen and not
        part of the DOFs (i.e., part of finger_links), then they will 
        be included in the SubRobotModel if all_descendants=True.  This
        means there may be a discrepancy between the finger configuration
        and the sub-robot configuration.

        Otherwise, they will be excluded and finger configurations will
        map one-to-one to the sub-robot.
        """
        if all_descendants:
            return SubRobotModel(robot,[self.base_link] + self.descendant_links(robot))
        return SubRobotModel(robot,[self.base_link]+list(self.finger_links))

    def get_geometry(self,robot,qfinger=None):
        """Returns a Geometry of the gripper frozen at its configuration.
        If qfinger = None, the current configuration is used.  Otherwise,
        qfinger is a finger configuration.
        """
        if qfinger is not None:
            q0 = robot.getConfig()
            robot.setConfig(self.set_finger_config(q0,qfinger))
        res = Geometry3D()
        res.setGroup()
        Tbase = robot.link(self.base_link).getTransform()
        for i,link in enumerate([self.base_link] + self.descendant_links(robot)):
            Trel = se3.mul(se3.inv(Tbase),robot.link(link).getTransform())
            g = robot.link(link).geometry().clone()
            if not g.empty():
                g.setCurrentTransform(*se3.identity())
                g.transform(*Trel)
            else:
                print("Uh... link",robot.link(link).getName(),"has empty geometry?")
            res.setElement(i,g)
        if qfinger is not None:
            robot.setConfig(q0)
        return res

    def add_to_vis(self,robot=None,animate=True):
        """Adds the gripper to the klampt.vis scene."""
        from klampt import vis
        from klampt import WorldModel,Geometry3D,GeometricPrimitive
        from klampt.model.trajectory import Trajectory
        base_xform = se3.identity()
        prefix = "gripper_"+self.name
        if robot is None and self.klampt_model is not None:
            w = WorldModel()
            if w.readFile(self.klampt_model):
                robot = w.robot(0)
                vis.add(prefix+"_gripper",w)
                robotPath = (prefix+"_gripper",robot.getName())
        elif robot is not None:
            vis.add(prefix+"_gripper",robot)
            robotPath = prefix+"_gripper"
        if robot is not None:
            assert self.base_link >= 0 and self.base_link < robot.numLinks()
            robot.link(self.base_link).appearance().setColor(1,0.75,0.5)
            base_xform = robot.link(self.base_link).getTransform()
            for l in self.finger_links:
                assert l >= 0 and l < robot.numLinks()
                robot.link(l).appearance().setColor(1,1,0.5)
        if robot is not None and animate:
            q0 = robot.getConfig()
            for i in self.finger_drivers:
                if isinstance(i,(list,tuple)):
                    for j in i:
                        robot.driver(j).setValue(1)
                else:
                    robot.driver(i).setValue(1)
            traj = Trajectory([0],[robot.getConfig()])
            for i in self.finger_drivers:
                if isinstance(i,(list,tuple)):
                    for j in i:
                        robot.driver(j).setValue(0)
                        traj.times.append(traj.endTime()+0.5)
                        traj.milestones.append(robot.getConfig())
                else:
                    robot.driver(i).setValue(0)
                    traj.times.append(traj.endTime()+1)
                    traj.milestones.append(robot.getConfig())
            traj.times.append(traj.endTime()+1)
            traj.milestones.append(traj.milestones[0])
            traj.times.append(traj.endTime()+1)
            traj.milestones.append(traj.milestones[0])
            assert len(traj.times) == len(traj.milestones)
            traj.checkValid()
            vis.animate(robotPath,traj)
            robot.setConfig(q0)
        if self.center is not None:
            vis.add(prefix+"_center",se3.apply(base_xform,self.center))
        center_point = (0,0,0) if self.center is None else self.center
        outer_point = (0,0,0)
        if self.primary_axis is not None:
            length = 0.1 if self.finger_length is None else self.finger_length
            outer_point = vectorops.madd(self.center,self.primary_axis,length)
            line = Trajectory([0,1],[self.center,outer_point])
            line.milestones = [se3.apply(base_xform,m) for m in line.milestones]
            vis.add(prefix+"_primary",line,color=(1,0,0,1))
        if self.secondary_axis is not None:
            width = 0.1 if self.opening_span is None else self.opening_span
            line = Trajectory([0,1],[vectorops.madd(outer_point,self.secondary_axis,-0.5*width),vectorops.madd(outer_point,self.secondary_axis,0.5*width)])
            line.milestones = [se3.apply(base_xform,m) for m in line.milestones]
            vis.add(prefix+"_secondary",line,color=(0,1,0,1))
        elif self.opening_span is not None:
            #assume vacuum gripper?
            p = GeometricPrimitive()
            p.setSphere(outer_point,self.opening_span)
            g = Geometry3D()
            g.set(p)
            vis.add(prefix+"_opening",g,color=(0,1,0,0.25))
        #TODO: add finger box

    def remove_from_vis(self):
        """Removes a previously-added gripper from the klampt.vis scene."""
        prefix = "gripper_"+self.name
        from klampt import vis
        try:
            vis.remove(prefix+"_world")
        except Exception:
            pass
        if self.center is not None:
            vis.remove(prefix+"_center")
        if self.primary_axis is not None:
            vis.remove(prefix+"_primary")
        if self.secondary_axis is not None:
            vis.remove(prefix+"_secondary")
        elif self.opening_span is not None:
            vis.remove(prefix+"_opening")
        
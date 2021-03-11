from gripper import GripperInfo
from klampt.model import contact
from klampt.model import ik
from klampt.model.contact import ContactPoint
from klampt import io
from klampt import IKObjective
from klampt.math import se3
import copy

class Grasp:
    """A "fully-qualified" grasp, specifying the entire set of IK and finger
    constraints.  Optional contacts are also given, in world space.
    """
    def __init__(self,ik_constraint=None,finger_links=None,finger_config=None,contacts=None,score=1):
        self.ik_constraint = ik_constraint
        self.finger_links = finger_links if finger_links is not None else []
        self.finger_config = finger_config if finger_config is not None else []
        self.contacts = contacts if contacts is not None else []
        if ik_constraint is not None:
            assert isinstance(ik_constraint,IKObjective)
        for c in self.contacts:
            assert isinstance(c,ContactPoint)
        self.score = score
    
    def __str__(self):
        return "Grasp link={} at {}, fingers={} score={}".format(self.ik_constraint.link(),self.ik_constraint.getPosition()[1],
            ' '.join('%d=%.2f'%(i,v) for (i,v) in zip(self.finger_links,self.finger_config)),self.score)

    def gripper_link(self):
        """Returns the link that is fixed by the IK constraint"""
        if self.ik_constraint is None: return None
        return self.ik_constraint.link()

    def set_finger_config(self,q):
        """Given a full robot config q, returns a config but with the finger
        degrees of freedom fixed.
        """
        qf = [v for v in q]
        for (i,v) in zip(self.finger_links,self.finger_config):
            qf[i] = v
        return qf

    def get_ik_solver(self,robot):
        """Returns a configured IK solver that will try to achieve the
        specified constraints.
        """
        obj = self.ik_constraint.copy()
        obj.robot = robot
        solver = ik.solver(obj)
        q = robot.getConfig()
        robot.setConfig(self.set_finger_config(q))
        active = solver.getActiveDofs()
        marked_active = [False]*robot.numLinks()
        for a in active:
            marked_active[a] = True
        for l in self.finger_links:
            marked_active[l] = False
        active = [i for i,a in enumerate(marked_active) if a]
        solver.setActiveDofs(active)
        return solver

    def get_transformed(self,xform):
        """Returns a copy of self, transformed by xform.
        """
        obj = self.ik_constraint.copy()
        obj.transform(*xform)
        world_contacts = [copy.copy(c) for c in self.contacts]
        for c in world_contacts:
            c.transform(xform)
        return Grasp(obj,self.finger_links,self.finger_config,world_contacts,self.score)

    def sample_fixed_grasp(self,Tref=None):
        """For non-fixed grasps, samples a Grasp with a fully specified base transform."""
        if Tref is None:
            Tref = se3.identity()
        Tfixed = self.ik_constraint.closestMatch(*Tref)
        ik2 = self.ik_constraint.copy()
        ik2.setFixedTransform(self.ik_constraint.link(),*Tfixed)
        return Grasp(ik2,self.finger_links,self.finger_config,self.contacts,self.score)

    def transfer(self,gripper_source,gripper_dest):
        """Creates a copy of this Grasp so that it can be used for another
        gripper.  self must match gripper_source, and the number of finger
        DOFs in gripper_source and gripper_dest must match.
        """
        if gripper_source.base_link != self.ik_constraint.link():
            raise ValueError("Invalid gripper source? base_link doesn't match")
        if self.finger_links != gripper_source.finger_links:
            raise ValueError("Invalid gripper source? finger_links doesn't match")
        if len(gripper_dest.finger_links) != len(gripper_source.finger_links):
            raise ValueError("Invalid gripper destination? finger_links doens't match")
        obj = self.ik_constraint.copy()
        obj.setLinks(gripper_dest.base_link,obj.destLink())
        return Grasp(obj,gripper_dest.finger_links,self.finger_config,self.contacts,self.score)

    def toJson(self):
        """Returns a JSON-compatible object storing all of the grasp data."""
        return {'ik_constraint':io.toJson(self.ik_constraint),'finger_links':self.finger_links,'finger_config':self.finger_config,'score':self.score}

    def fromJson(self,jsonObj):
        """Creates the grasp from a JSON-compatible object previously saved by
        toJson.
        """
        self.ik_constraint = io.fromJson(jsonObj['ik_constraint'])
        self.finger_links = jsonObj['finger_links']
        self.finger_config = jsonObj['finger_config']
        self.score = jsonObj.get('score',0)

    def add_to_vis(self,prefix,hide_label=True):
        from klampt import vis
        vis.add(prefix+"_ik",self.ik_constraint,hide_label=hide_label)
        for i,c in enumerate(self.contacts):
            vis.add(prefix+"_c"+str(i),c,hide_label=hide_label)

    def remove_from_vis(self,prefix):
        from klampt import vis
        vis.remove(prefix+"_ik")
        for i,c in enumerate(self.contacts):
            vis.remove(prefix+"_c"+str(i))


class GraspWithConfig(Grasp):
    """Result from a GraspSamplerBase that may return a robot configuration
    as well as the Grasp.
    """
    def __init__(self,ik_constraint=None,finger_links=None,finger_config=None,robot_config=None,score=0):
        Grasp.__init__(self,ik_constraint,finger_links,finger_config,score)
        self.robot_config = robot_config

    def toJson(self):
        """Returns a JSON-compatible object storing all of the grasp/config
        data.
        """
        res = Grasp.toJson(self)
        res['robot_config'] = self.robot_config
        return res

    def fromJson(self,jsonObj):
        """Creates the GraspWithConfig from a JSON-compatible object previously
        saved by toJson.
        """
        Grasp.fromJson(self,jsonObj)
        self.robot_config = jsonObj.get('robot_config',None)

    def add_to_vis(self,prefix):
        from klampt import vis
        Grasp.add_to_vis(self,prefix)
        vis.add(prefix+"_config",self.robot_config,color=(0,0,1,0.5))

    def remove_from_vis(self,prefix):
        from klampt import vis
        Grasp.remove_from_vis(self,prefix)
        vis.remove(prefix+"_config")


class GraspSamplerBase:
    """An abstract base class that will sample grasps for a given scene.
    """
    def gripper(self):
        """Returns a GripperInfo describing which gripper this works for."""
        raise NotImplementedError()

    def init(self,scene,object,hints):
        """Does some initialization of the grasp generator.  Return True if
        this generator can be applied.

        For object-centric grasp samplers, object should be a RigidObjectModel
        containing the object model in its estimated pose.

        For image-based grasp samplers, object is typically a RigidObjectModel
        containing the point cloud of the scene / object to be grasped. 
        hints can include a point in image space, a point in 3D space,
        a ray, or even a description of the object.
        """
        return True

    def next(self):
        """Returns the next candidate Grasp or GraspWithConfig for the scene
        and object provided in init().  Can return None if the grasp generator
        fails."""
        raise NotImplementedError()

    def score(self):
        """Return a heuristic score that measures the likelihood of a grasp
        being generated for the scene and object provided in init()."""
        raise NotImplementedError()

    def vis_debug(self):
        """Does some visual debugging of the sampler, if possible."""
        return

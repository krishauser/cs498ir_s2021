from klampt import WorldModel
from klampt.model import ik
from klampt.math import so3,se3
from gripper import GripperInfo
from grasp import Grasp
import features
import copy
import math

class GraspParameterSpace:
    """Allows for a vectorized representation of a Grasp.

    Attributes:
        gripper (GripperInfo, optional): the info of the gripper being used.
        template (Grasp): the grasp template.
        template_json (dict): the JSON object belonging to template
        items (list of str or list of list of str): the grasp items referred
            to by the parameter space.  Can be:
            
            - 'transform' (6D rotation_vector + translation concatenated)
            - 'finger_driver_config' (len(gripper.finger_drivers))
            - 'finger_config' (len(gripper.finger_links))
            - ['ikConstraint','endPosition'] (3D),
            - ['ikConstraint','endRotation'] (3D),
            - 'robot_config'
            
            Item strings can also refer to sub-indices, like
            ['ikConstraint']['endPosition'][0] or ['transform'][3].
    
    """
    def __init__(self,template,items,gripper=None):
        self.gripper = gripper
        if isinstance(template,GripperInfo):
            self.gripper = template    
        else:
            self.gripper = None
        if self.gripper is not None:
            w = WorldModel()
            res = w.readFile(self.gripper.klampt_model)
            if not res:
                raise RuntimeError("GraspParameterSpace: Error, could not load gripper model")
            self.world = w
            self.robot = w.robot(0)
        if isinstance(template,GripperInfo):
            fixed_constraint = ik.fixed_objective(self.robot.link(self.gripper.base_link))
            q = self.gripper.closed_config if self.gripper.closed_config is not None else [0]*len(self.gripper.finger_links)
            template = Grasp(fixed_constraint,self.gripper.finger_links,q)
        self.template = template
        self.template_json = template.toJson()
        self.items = items
        self.use_transform = ('transform' in items or any(i[0] == 'transform' for i in items))
        self.use_drivers = ('finger_driver_config' in items or any(i[0] == 'finger_driver_config' for i in items))
        if self.use_transform:
            T = self.template.ik_constraint.closestMatch(*se3.identity())
            self.template_json['transform'] = so3.rotation_vector(T[0]) + T[1]
        if self.use_drivers:
            if self.gripper is None:
                raise ValueError("Cannot use finger_driver_config unless gripper is provided")
            #self.template_json['finger_driver_config'] = [0]*len(self.gripper.finger_drivers)
            self.robot.setConfig(self.gripper.set_finger_config(self.robot.getConfig(),template.finger_config))
            qdriver = [self.robot.driver(i).getValue() for i in self.gripper.finger_drivers]
            self.template_json['finger_driver_config'] = qdriver

        #check
        try:
            nd = self.num_dims()
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError("Invalid items specified")

        self.min_template_json = copy.deepcopy(self.template_json)
        self.max_template_json = copy.deepcopy(self.template_json)
        inf = float('inf')
        self.min_template_json['ik_constraint']['endRotation'] = [-math.pi,-math.pi,-math.pi]
        self.max_template_json['ik_constraint']['endRotation'] = [math.pi,math.pi,math.pi]
        self.min_template_json['ik_constraint']['endPosition'] = [-inf,-inf,-inf]
        self.max_template_json['ik_constraint']['endPosition'] = [inf,inf,inf]
        self.min_template_json['finger_config'] = [-inf]*len(self.template_json['finger_config'])
        self.max_template_json['finger_config'] = [inf]*len(self.template_json['finger_config'])
        if self.use_transform:
            self.min_template_json['transform'] = [-math.pi,-math.pi,-math.pi,-inf,-inf,-inf]
            self.max_template_json['transform'] = [math.pi,math.pi,math.pi,inf,inf,inf]
        
            if len(self.template.finger_links) > 0:
                qmin,qmax = self.robot.getJointLimits()
                self.min_template_json['finger_config'] = [qmin[i] for i in self.template.finger_links]
                self.max_template_json['finger_config'] = [qmax[i] for i in self.template.finger_links]

        if self.use_drivers and len(self.gripper.finger_drivers) > 0:
            qmin,qmax = self.robot.getJointLimits()
            drivers = [self.robot.driver(i) for i in self.gripper.finger_drivers]
            dmin,dmax = [],[]
            for d in drivers:
                dlinks = d.getAffectedLinks()
                if d.getType() == 'affine':
                    scale,offset = d.getAffineCoeffs()
                    ddmin = float('inf')
                    ddmax = -ddmin
                    for i,s,o in zip(dlinks,scale,offset):
                        if (qmin[i]-o)/s < ddmin:
                            ddmin = (qmin[i]-o)/s
                        if (qmax[i]-o)/s > ddmax:
                            ddmax = (qmax[i]-o)/s
                    dmin.append(ddmin)
                    dmax.append(ddmax)
                else:
                    dmin.append(qmin[dlinks[0]])
                    dmax.append(qmax[dlinks[0]])
            self.min_template_json['finger_driver_config'] = dmin
            self.max_template_json['finger_driver_config'] = dmax
    
    def num_dims(self) -> int:
        """Returns the number of dimensions in the parameter space."""
        return len(self.get_features(self.template))
    
    def get_features(self,grasp : Grasp):
        """Converts a grasp into a feature vector.  ``grasp`` must be 
        compatible with ``template``.
        """
        import features
        jsonobj = grasp.toJson()
        #set up special items `transform` and `finger_driver_config`
        if self.use_transform:
            T = grasp.ik_constraint.closestMatch(*se3.identity())
            jsonobj['transform'] = so3.rotation_vector(T[0]) + T[1]
        if self.use_drivers:
            self.robot.setConfig(self.gripper.set_finger_config(self.robot.getConfig(),grasp.finger_config))
            qdriver = [self.robot.driver(i).getValue() for i in self.gripper.finger_drivers]
            jsonobj['finger_driver_config'] = qdriver
        return features.extract(jsonobj,self.items)
    
    def get_grasp(self,feature_vec) -> Grasp:
        """Converts a feature vector to a Grasp.  ``feature_vec`` must
        be of the right length.
        """
        res = copy.deepcopy(self.template_json)
        features.inject(res,self.items,feature_vec)
        g = self.template.__class__()
        g.fromJson(res)
        if self.use_transform:
            Tvec = res['transform']
            rv,t = Tvec[:3],Tvec[3:]
            T = so3.from_rotation_vector(rv),t
            g.ik_constraint.matchDestination(*T)
        if self.use_drivers:
            for i,v in zip(self.gripper.finger_drivers,res['finger_driver_config']):
                self.robot.driver(i).setValue(v)
            q = self.robot.getConfig()
            g.finger_config = [q[i] for i in self.gripper.finger_links]
        return g
    
    def get_bounds(self):
        """Returns bounds on the feature space.  Some of these may be
        infinite.
        """
        return (features.extract(self.min_template_json,self.items),
                features.extract(self.max_template_json,self.items))

    def set_position_bounds(self,pmin,pmax):
        """Restricts the position bounds to the bounding box (pmin,pmax)."""
        if 'transform' in self.min_template_json:
            self.min_template_json['transform'][3:6] = pmin
            self.max_template_json['transform'][3:6] = pmax
        self.min_template_json['ik_constraint']['endPosition'] = pmin
        self.min_template_json['ik_constraint']['endPosition'] = pmax


if __name__ == '__main__':
    from known_grippers import *
    #items = [['ik_constraint','endPosition',0],['ik_constraint','endPosition',1],'finger_config']
    #items = ['transform']
    items = [['transform',3],['transform',4],['transform',5]] #only translation
    #items = ['finger_driver_config']
    space = GraspParameterSpace(robotiq_85,items)
    nd = space.num_dims()
    print(space.get_features(space.template))
    print(space.get_bounds())
    print(space.get_grasp([0]*nd))
    print(space.get_grasp([1]*nd))
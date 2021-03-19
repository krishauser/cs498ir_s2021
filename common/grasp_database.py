from gripper import GripperInfo
from grasp import *
from klampt.model import ik
from klampt.math import vectorops,so3,se3
from klampt.model import contact
from klampt import io
import copy
import json
import os


def _object_name(obj):
    if isinstance(obj,str):
        return obj
    elif hasattr(obj,'name'):
        return obj.name
    elif hasattr(obj,'getName'):
        return obj.getName()
    else:
        raise ValueError("Can't determine name of object {}".format(obj))


class GraspDatabase:
    """A database of grasps, loadable from disk"""
    def __init__(self,gripper,fn=None):
        if not isinstance(gripper,GripperInfo):
            raise ValueError("gripper needs to be a GripperInfo")
        self.gripper = gripper
        GripperInfo.register(gripper)
        self.objects = []
        self.object_to_grasps = dict()
        if fn is not None:
            self.load(fn)

    def print_info(self):
        print("Grasp database statistics:")
        print("Gripper:",self.gripper.name)
        for (o,gs) in self.object_to_grasps:
            print("Object",o,":",len(gs),"grasps")

    def load(self,fn):
        with open(fn,'r') as f:
            jsonobj = json.load(f)
        self.objects = jsonobj['objects']
        self.object_to_grasps = dict()
        for o,gs in jsonobj['object_to_grasps'].items():
            grasps = []
            for g in gs:
                gparsed = Grasp(None)
                gparsed.fromJson(g)
                grasps.append(gparsed)
            self.object_to_grasps[o] = grasps
        return True

    def save(self,fn):
        jsonobj = dict()
        jsonobj['objects'] = self.objects
        grasp_dict = dict()
        for o,gs in self.object_to_grasps.items():
            gsjson = [g.toJson() for g in gs]
            grasp_dict[o] = gsjson
        jsonobj['object_to_grasps'] = grasp_dict
        with open(fn,'w') as f:
            json.dump(jsonobj,f)
        return True

    def loadfolder(self,fn):
        """Reads from a folder containing object folders, each of which contains
        some set of grasp json files."""
        for obj in os.listdir(fn):
            print(obj)
            if os.path.isdir(os.path.join(fn,obj)):
                inobjs = False
                for gfn in os.listdir(os.path.join(fn,obj)):
                    if gfn.endswith('.json') and self.gripper.name in gfn:
                        if not inobjs:
                            if obj not in self.objects:
                                self.objects.append(obj)
                                self.object_to_grasps[obj] = []
                            inobjs = True
                            print("Loading grasps for object",obj,"...")
                        with open(os.path.join(fn,obj,gfn),'r') as f:
                            jsonobj = json.load(f)
                        try:
                            gparsed = Grasp(None)
                            gparsed.fromJson(jsonobj)
                            self.object_to_grasps[obj].append(gparsed)
                        except Exception as e:
                            print("Unable to load",os.path.join(fn,obj,gfn),"as a Grasp")
                            raise

    def add_object(self,name):
        if name in self.objects:
            raise ValueError("Object {} already exists".format(name))
        self.objects.append(name)
        self.object_to_grasps[name] = []

    def add_grasp(self,object,grasp):
        """Adds a new Grasp (assumed object centric) to the database for the
        given object.
        """
        if isinstance(grasp,Grasp):
            oname = _object_name(object)
            if oname not in self.object_to_grasps:
                self.add_object(oname)
            self.object_to_grasps[oname].append(grasp)
        else:
            raise ValueError("grasp needs to be a Grasp")

    def get_sampler(self,robot):
        """Returns a GraspDatabaseSampler for this robot."""
        return GraspDatabaseSampler(robot,self.gripper,self.object_to_grasps)


class GraspDatabaseSampler(GraspSamplerBase):
    """A GraspSamplerBase subclass that will read from a dict of
    object-centric Grasp's.

    Args:
        robot (RobotModel): the robot.
        gripper (GripperInfo): the gripper
        object_to_grasps (dict of str -> list): for each object, gives a list
            of Grasp templates.
    """
    def __init__(self,robot,gripper,object_to_grasps):
        self._robot = robot
        self._gripper = gripper
        self._object_to_grasps = object_to_grasps
        self._target_object = None
        self._matching_object = None
        self._matching_xform = None
        self._grasp_index = None

    def object_match(self,object_source,object_target):
        """Determine whether object_source is a match to object_target.
        If they match, return a transform from the reference frame of
        object_source to object_target.  Otherwise, return None.

        Default implementation: determine whether the name of
        object_source matches object_target.name or object_target.getName()
        exactly.
        """
        if object_source != _object_name(object_target):
            return se3.identity()
        return None

    def gripper(self):
        return self._gripper

    def init(self,scene,object,hints):
        """Checks for either an exact match or if object_match(o,object)
        exists"""
        if _object_name(object) in self._object_to_grasps:
            self._target_object = object
            self._matching_object = _object_name(object)
            self._matching_xform = se3.identity()
            self._grasp_index = 0
            return True
        for o,g in self._object_to_grasps.items():
            xform = self.object_match(o,object)
            if xform is not None:
                self._target_object = object
                self._matching_object = o
                self._matching_xform = xform
                self._grasp_index = 0
                return True
        return False

    def next(self):
        """Returns the next Grasp from the database."""
        if self._matching_object is None:
            return None
        grasps = self._object_to_grasps[self._matching_object]
        if self._grasp_index >= len(grasps):
            self._matching_object = None
            return None
        grasp = grasps[self._grasp_index]
        self._grasp_index += 1
        return grasp.get_transformed(se3.mul(self._target_object.getTransform(),self._matching_xform))
    
    def score(self):
        """Returns a score going from 1 to 0 as the number of grasps
        gets exhausted.
        """
        if self._matching_object is None: return 0
        grasps = self._object_to_grasps[self._matching_object]
        if self._grasp_index >= len(grasps):
            return 0
        return 1.0 - self._grasp_index/float(len(grasps))

def browse_database(gripper,dbfile=None):
    from known_grippers import GripperInfo
    g = GripperInfo.get(gripper)
    if g is None:
        print("Invalid gripper, valid names are",list(GripperInfo.all_grippers.keys()))
        exit(1)
    db = GraspDatabase(g)
    if dbfile is not None:
        if dbfile.endswith('.json'):
            db.load(dbfile)
        else:
            db.loadfolder(dbfile)

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../data'))
    FIND_PATTERNS = [data_dir+"/objects/%s.obj",data_dir+"/objects/ycb-select/%s/textured.obj"]
    def _find_object(name):
        for pat in FIND_PATTERNS:
            fn = pat%(name,)
            if os.path.exists(fn):
                return fn
        return None

    w = WorldModel()
    if not w.readFile(g.klampt_model):
        print("Can't load gripper robot model",g.klampt_model)
        exit(1)
    robot = w.robot(0)
    for o in db.objects:
        fn = _find_object(o)
        if fn is None:
            print("Can't find object",o,"in usual paths...")
            continue
        if w.numRigidObjects()==0:
            obj = w.makeRigidObject(o)
            if not obj.loadFile(fn):
                if not obj.geometry().loadFile(fn):
                    print("Couldn't load object",o,"from",fn)
                    exit(1)
            obj.setTransform(*se3.identity())
    if len(db.objects)==0:
        print("Can't show anything, no objects")
        print("Try adding some grasps to the database using grasp_edit.py")
        exit(0)

    data = dict()
    data['cur_object'] = 0
    data['cur_grasp'] = -1
    data['shown_grasps'] = []
    vis.add(db.objects[data['cur_object']],w.rigidObject(0))
    def shift_object(amt,data=data):
        vis.remove(db.objects[data['cur_object']])
        data['cur_object'] += amt
        if data['cur_object'] >= len(db.objects):
            data['cur_object'] = 0
        elif data['cur_object'] < 0:
            data['cur_object'] = len(db.objects)-1
        if data['cur_object'] >= w.numRigidObjects():
            for i in range(w.numRigidObjects(),data['cur_object']+1):
                o = db.objects[i]
                fn = _find_object(o)
                obj = w.makeRigidObject(o)
                if not obj.loadFile(fn):
                    if not obj.geometry().loadFile(fn):
                        print("Couldn't load object",o,"from",fn)
                        exit(1)
                obj.setTransform(*se3.identity())
        obj = w.rigidObject(data['cur_object'])
        vis.add(db.objects[data['cur_object']],obj)
        shift_grasp(None)

    def shift_grasp(amt,data=data):
        for i,grasp in data['shown_grasps']:
            grasp.remove_from_vis("grasp"+str(i))
        data['shown_grasps'] = []
        all_grasps = db.object_to_grasps[db.objects[data['cur_object']]]
        if amt == None:
            data['cur_grasp'] = -1
        else:
            data['cur_grasp'] += amt
            if data['cur_grasp'] >= len(all_grasps):
                data['cur_grasp'] = -1
            elif data['cur_grasp'] < -1:
                data['cur_grasp'] = len(all_grasps)-1
        if data['cur_grasp']==-1:
            for i,grasp in enumerate(all_grasps):
                grasp.ik_constraint.robot = robot
                grasp.add_to_vis("grasp"+str(i))
                data['shown_grasps'].append((i,grasp))
            print("Showing",len(data['shown_grasps']),"grasps")
        else:
            grasp = all_grasps[data['cur_grasp']]
            grasp.ik_constraint.robot = robot
            grasp.add_to_vis("grasp"+str(data['cur_grasp']))
            Tbase = grasp.ik_constraint.closestMatch(*se3.identity())
            g.add_to_vis(robot,animate=False,base_xform=Tbase)
            robot.setConfig(grasp.set_finger_config(robot.getConfig()))
            data['shown_grasps'].append((data['cur_grasp'],grasp))
            if grasp.score is not None:
                vis.addText("score","Score %.3f"%(grasp.score,),position=(10,10))
            else:
                vis.addText("score","",position=(10,10))

    vis.addAction(lambda: shift_object(1),"Next object",'.')
    vis.addAction(lambda: shift_object(-1),"Prev object",',')
    vis.addAction(lambda: shift_grasp(1),"Next grasp",'=')
    vis.addAction(lambda: shift_grasp(-1),"Prev grasp",'-')
    vis.addAction(lambda: shift_grasp(None),"All grasps",'0')
    vis.add("gripper",w.robot(0))
    vis.run()

if __name__ == '__main__':
    import sys
    from klampt import WorldModel
    from klampt import vis
    if len(sys.argv) < 2:
        print("Usage: python grasp_database.py gripper [FILE]")
        exit(0)
    print(sys.argv)
    gripper = sys.argv[1]
    dbfile = None
    if len(sys.argv) >= 3:
        dbfile = sys.argv[2]
    browse_database(gripper,dbfile)

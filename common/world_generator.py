import os
import json
import random
from klampt.math import vectorops,so3,se3
from klampt import Geometry3D,Mass
from klampt.model.create import primitives
from klampt.model import collide
import math
import platform

DEFAULT_OBJECT_MASS = 1
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'../data'))
YCB_DIR = os.path.join(DATA_DIR,"objects/ycb-select")
ycb_objects = list(d for d in os.listdir(YCB_DIR) if os.path.isdir(os.path.join(YCB_DIR,d)))
ycb_masses = dict()
with open(os.path.join(YCB_DIR,'masses.json'),'r') as f:
    ycb_masses = json.load(f)

def make_ycb_object(objectname,world,textured=True):
    """Adds an object to the world using its geometry / mass properties
    and places it in a default location (x,y)=(0,0) and resting on plane."""
    if objectname == 'random':
        objectname = random.choice(ycb_objects)
    if isinstance(objectname,int):
        objectname = ycb_objects[objectname]

    objmass = ycb_masses.get(objectname,DEFAULT_OBJECT_MASS)
    if textured:
        fn = os.path.join(YCB_DIR,objectname,'textured.obj')
    else:
        fn = os.path.join(YCB_DIR,objectname,'nontextured.ply')

    obj = world.makeRigidObject(objectname)
    if not obj.geometry().loadFile(os.path.relpath(fn)):
        raise IOError("Unable to read geometry from file",fn)
    obj.setTransform(*se3.identity())

    #set mass automatically
    mass = obj.getMass()
    surfaceFraction = 0.5
    mass.estimate(obj.geometry(),objmass,surfaceFraction)
    obj.setMass(mass)

    bmin,bmax = obj.geometry().getBB()
    T = obj.getTransform()
    spacing = 0.006
    T = (T[0],vectorops.add(T[1],(-(bmin[0]+bmax[0])*0.5,-(bmin[1]+bmax[1])*0.5,-bmin[2]+spacing)))
    obj.setTransform(*T)
    if textured:
        obj.appearance().setColor(1,1,1,1.0)
    else:
        obj.appearance().setColor(random.random(),random.random(),random.random(),1.0)
    obj.setName(objectname)
    return obj

def make_primitive_object(world,typename,name=None):
    """Adds an object to the world using its geometry / mass properties
    and places it in a default location (x,y)=(0,0) and resting on plane."""
    if name is None:
        name = typename
    fn = os.path.join(DATA_DIR,"/objects/"+typename+".obj")
    if not world.readFile(fn):
        raise IOError("Unable to read primitive from file",fn)
    obj = world.rigidObject(world.numRigidObjects()-1)
    T = obj.getTransform()
    spacing = 0.006
    bmin,bmax = obj.geometry().getBB()
    T = (T[0],vectorops.add(T[1],(-(bmin[0]+bmax[0])*0.5,-(bmin[1]+bmax[1])*0.5,-bmin[2]+spacing)))
    obj.setTransform(*T)
    obj.appearance().setColor(0.2,0.5,0.7,1.0)
    obj.setName(name)
    return obj

def make_box(world,width,depth,height,wall_thickness=0.005,mass=float('inf')):
    """Makes a new axis-aligned open-top box with its bottom centered at the origin
    with dimensions width x depth x height. Walls have thickness wall_thickness. 
    If mass=inf, then the box is a TerrainModel, otherwise it's a RigidObjectModel
    with automatically determined inertia.
    """
    left = primitives.box(wall_thickness,depth,height,[-width*0.5-wall_thickness*0.5,0,height*0.5])
    right = primitives.box(wall_thickness,depth,height,[width*0.5+wall_thickness*0.5,0,height*0.5])
    front = primitives.box(width,wall_thickness,height,[0,-depth*0.5-wall_thickness*0.5,height*0.5])
    back = primitives.box(width,wall_thickness,height,[0,depth*0.5+wall_thickness*0.5,height*0.5])
    bottom = primitives.box(width,depth,wall_thickness,[0,0,wall_thickness*0.5])
    boxgeom = Geometry3D()
    boxgeom.setGroup()
    for i,elem in enumerate([left,right,front,back,bottom]):
        boxgeom.setElement(i,elem)
    if mass != float('inf'):
        print("Making open-top box a rigid object")
        bmass = Mass()
        bmass.setMass(mass)
        bmass.setCom([0,0,height*0.3])
        bmass.setInertia([width/12,depth/12,height/12])
        box = world.makeRigidObject("box")
        box.geometry().set(boxgeom)
        box.appearance().setColor(0.6,0.3,0.2,1.0)
        box.setMass(bmass)
        return box
    else:
        print("Making open-top box a terrain")
        box = world.makeTerrain("box")
        box.geometry().set(boxgeom)
        box.appearance().setColor(0.6,0.3,0.2,1.0)
        return box

def make_shelf(world,width,depth,height,wall_thickness=0.005,mass=float('inf')):
    """Makes a new axis-aligned "shelf" with its bottom centered at the origin 
    with dimensions width x depth x height. Walls have thickness wall_thickness. 
    If mass=inf, then the box is a TerrainModel, otherwise it's a RigidObjectModel
    with automatically determined inertia.
    """
    left = primitives.box(wall_thickness,depth,height,[-width*0.5-wall_thickness*0.5,0])
    right = primitives.box(wall_thickness,depth,height,[width*0.5+wall_thickness*0.5,0,height*0.5])
    back = primitives.box(width,wall_thickness,height,[0,depth*0.5+wall_thickness*0.5,height*0.5])
    bottom = primitives.box(width,depth,wall_thickness,[0,0,wall_thickness*0.5])
    top = primitives.box(width,depth,wall_thickness,[0,0,height+wall_thickness*0.5])
    shelfgeom = Geometry3D()
    shelfgeom.setGroup()
    for i,elem in enumerate([left,right,back,bottom,top]):
        shelfgeom.setElement(i,elem)
    if mass != float('inf'):
        print("Making shelf box a rigid object")
        bmass = Mass()
        bmass.setMass(mass)
        bmass.setCom([0,depth*0.3,height*0.5])
        bmass.setInertia([width/12,depth/12,height/12])
        shelf = world.makeRigidObject("shelf")
        shelf.geometry().set(shelfgeom)
        shelf.appearance().setColor(0.2,0.6,0.3,1.0)
        shelf.setMass(bmass)
        return shelf
    else:
        shelf = world.makeTerrain("shelf")
        shelf.geometry().set(shelfgeom)
        shelf.appearance().setColor(0.2,0.6,0.3,1.0)
        return shelf

def save_world(world,fn):
    """Does what world.saveFile(fn) should do, but implements
    a Workaround for a world saving bug in Windows..."""
    folder = os.path.splitext(fn)[0]
    print("Saving world to",fn)
    #annoying workaround: bug saving files on Windows 
    try:
        os.makedirs(folder)
    except Exception:
        pass
    for o in range(world.numRigidObjects()):
        name = world.rigidObject(o).getName()
        try:
            os.mkdir(os.path.join(folder,name))
        except Exception:
            pass
    for o in range(world.numTerrains()):
        name = world.terrain(o).getName()
        try:
            os.mkdir(os.path.join(folder,name))
        except Exception:
            pass
    world.saveFile(fn,folder+'/')
    if platform.system()=='Windows':
        #hack for windows path references
        hack_exts = ['obj','env']
        import glob
        for ext in hack_exts:
            for fn in glob.glob(folder+"/*."+ext):
                with open(fn,'r') as f:
                    contents = ''.join(f.readlines())
                replaced = contents.replace('\\','/')
                with open(fn,'w') as f:
                    f.write(replaced)

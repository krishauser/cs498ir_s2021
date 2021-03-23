from klampt import WorldModel,GeometricPrimitive
from klampt.io import resource
from klampt.io import loader
from klampt import vis 
from klampt.math import vectorops,so3,se3
from klampt.model import sensing
import math
import random
import time
import json
import sys
import os
import glob
import numpy as np
from PIL import Image
sys.path.append("../common")
import grasp
import grasp_database
from known_grippers import *
from world_generator import ycb_objects,make_ycb_object,save_world
from stable_faces import stable_faces,debug_stable_faces

#PROBLEM = '1a'
PROBLEM = '1b'

base_files = ['table.xml','box.xml','shelf.xml']
base_boxes = []
ycb_stable_faces = dict()

def sample_object_pose_table(obj,stable_fs,bmin,bmax):
    """Samples a transform of the object so that it lies on in the given
    bounding box bmin,bmax.

    Args:
        obj (RigidObjectModel)
        stable_fs (list of lists): giving the stable faces of the object,
            as in MP2.
        bmin,bmax (3-vectors): the bounding box of the area in which the
            objects should lie.
    """
    table_height = bmin[2] + 0.001
    face = random.choice(stable_fs)
    normal = np.cross(face[1] - face[0],face[2]-face[0])
    normal = normal / np.linalg.norm(normal)
    centroid = np.sum(face,axis=0)/len(face)
    x = random.uniform(bmin[0],bmax[0])
    y = random.uniform(bmin[1],bmax[1])
    z = table_height + np.dot(centroid,normal)
    Rnormal = so3.canonical((-normal).tolist())
    Rx = so3.rotation((1,0,0),random.uniform(0,math.pi*2))
    Rxz = so3.rotation((0,1,0),-math.pi*0.5)
    R = so3.mul(Rxz,so3.mul(Rx,so3.inv(Rnormal)))
    #R*com + t = [x,y,_]
    t = vectorops.sub([x,y,z],so3.apply(R,obj.getMass().getCom()))
    t[2] = z
    obj.setTransform(R,t)

def arrange_objects(world,obj_fns,bmin,bmax,interior=False):
    global ycb_stable_faces
    for o,obj in enumerate(obj_fns):
        if obj not in ycb_stable_faces:
            world.rigidObject(o).setTransform(*se3.identity())
            ycb_stable_faces[obj] = stable_faces(world.rigidObject(o),stability_tol=0.01,merge_tol=0.05)
            if len(ycb_stable_faces[obj]) == 0:
                print("Object",obj,"has no stable faces with robustness 0.01, trying 0.0")
                ycb_stable_faces[obj] = stable_faces(world.rigidObject(o),stability_tol=0.0,merge_tol=0.05)
                #debug_stable_faces(world.rigidObject(o),ycb_stable_faces[obj])
    i = 0
    while i < world.numRigidObjects():
        faces = ycb_stable_faces[obj_fns[i]]
        samples = 0
        feasible = False
        while not feasible and samples < 100:
            sample_object_pose_table(world.rigidObject(i),faces,bmin,bmax)
            
            samples += 1
            feasible = True
            for j in range(i):
                if world.rigidObject(i).geometry().collides(world.rigidObject(j).geometry()):
                    feasible = False
                    break
            if feasible:
                for j in range(world.numTerrains()):
                    if world.rigidObject(i).geometry().collides(world.terrain(j).geometry()):
                        feasible = False
                        break
        if not feasible:
           world.remove(world.rigidObject(i))
           print("Couldn't find feasible placement for",i,"th object")
        else:
            i += 1

def gen_grasp_worlds(N=10):
    if len(base_boxes)==0:
        #edit base boxes
        for basefn in base_files:
            world = WorldModel()
            world.readFile(basefn)
            base_name = os.path.splitext(os.path.basename(basefn))[0]
            box = GeometricPrimitive()
            box.setAABB([0,0,0],[1,1,1])
            base_boxes.append(resource.get(base_name+'.geom','GeometricPrimitive',default=box,world=world,doedit='auto'))
    output_file_pattern = "generated_worlds/world_%04d.xml"
    #TODO: play around with me, problem 1a
    num_objects = [1,1,1,2,2,5]
    for i in range(N):
        nobj = random.choice(num_objects)
        world = WorldModel()
        base_world = random.choice(range(len(base_files)))
        world.readFile(base_files[base_world])
        obj_fns = []
        for o in range(nobj):
            fn = random.choice(ycb_objects)
            obj = make_ycb_object(fn,world)
            obj_fns.append(fn)
        bbox = base_boxes[base_world]
        bmin,bmax = [v for v in bbox.properties[0:3]],[v for v in bbox.properties[3:6]]
        #TODO: Problem 1: arrange objects within box bmin,bmax
        arrange_objects(world,obj_fns,bmin,bmax)

        save_world(world,output_file_pattern%(i,))


def autodetect_world_type(w):
    for i in range(w.numTerrains()):
        if w.terrain(i).getName().startswith('table'):
            return 'table'
        elif w.terrain(i).getName().startswith('box'):
            return 'box'
        elif w.terrain(i).getName().startswith('pod'):
            return 'shelf'
    return 'unknown'

def edit_camera_xform(world_fn,xform=None,title=None):
    """Visual editor of the camera position
    """
    world = WorldModel()
    world.readFile(world_fn)
    world.readFile("camera.rob")
    robot = world.robot(0)
    sensor = robot.sensor(0)
    if xform is not None:
        sensing.set_sensor_xform(sensor,xform)
    vis.createWindow()
    if title is not None:
        vis.setWindowTitle(title)
    vis.resizeWindow(1024,768)
    vis.add("world",world)
    vis.add("sensor",sensor)
    vis.add("sensor_xform",sensing.get_sensor_xform(sensor,robot))
    vis.edit("sensor_xform")
    def update_sensor_xform():
        sensor_xform = vis.getItemConfig("sensor_xform")
        sensor_xform = sensor_xform[:9],sensor_xform[9:]
        sensing.set_sensor_xform(sensor,sensor_xform)
    vis.loop(callback=update_sensor_xform)
    sensor_xform = vis.getItemConfig("sensor_xform")
    return sensor_xform[:9],sensor_xform[9:]

camera_viewpoints = dict()

def gen_grasp_images(max_variations=10):
    """Generates grasp training images for Problem 1b"""
    global camera_viewpoints
    if len(camera_viewpoints) == 0:
        #This code pops up the viewpoint editor
        edited = False
        for base in base_files:
            camera_viewpoints[base] = []
            camera_fn_template = os.path.join("resources",os.path.splitext(base)[0]+'_camera_%02d.xform')
            index = 0
            while True:
                camera_fn = camera_fn_template%(index,)
                if not os.path.exists(camera_fn):
                    break
                xform = loader.load('RigidTransform',camera_fn)
                camera_viewpoints[base].append(xform)
                index += 1
            if len(camera_viewpoints[base]) > 0:
                #TODO: if you want to edit the camera transforms, comment this line out and replace it with "pass"
                continue
                #pass
            edited = True
            for i,xform in enumerate(camera_viewpoints[base]):
                print("Camera transform",base,i)
                sensor_xform = edit_camera_xform(base,xform,title="Camera transform {} {}".format(base,i))
                camera_viewpoints[base][i] = sensor_xform
                camera_fn = camera_fn_template%(i,)
                loader.save(sensor_xform,'RigidTransform',camera_fn)
            while True:
                if len(camera_viewpoints[base]) > 0:
                    print("Do you want to add another? (y/n)")
                    choice = input()
                else:
                    choice = 'y'
                if choice.strip()=='y':
                    sensor_xform = edit_camera_xform(base,title="New camera transform {}".format(base))
                    camera_viewpoints[base].append(sensor_xform)
                    camera_fn = camera_fn_template%(len(camera_viewpoints[base])-1,)
                    loader.save(sensor_xform,'RigidTransform',camera_fn)
                else:
                    break
        if edited:
            print("Quitting; run me again to render images")
            return
    
    #Here's where the main loop begins
    try:
        os.mkdir('image_dataset')
    except Exception:
        pass
    
    try:
        os.remove("image_dataset/metadata.csv")
    except Exception:
        pass
    with open("image_dataset/metadata.csv",'w') as f:
        f.write("world,view,view_transform,color_fn,depth_fn,grasp_fn,variation\n")
    
    total_image_count = 0
    for fn in glob.glob("generated_worlds/world_*.xml"):
        ofs = fn.find('.xml')
        index = fn[ofs-4:ofs]
        world = WorldModel()
        world.readFile(fn)
        wtype = autodetect_world_type(world)
        if wtype == 'unknown':
            print("WARNING: DONT KNOW WORLD TYPE OF",fn)
        world_camera_viewpoints = camera_viewpoints[wtype+'.xml']
        assert len(world_camera_viewpoints) > 0

        #TODO: change appearances
        for i in range(world.numRigidObjects()):
            world.rigidObject(i).appearance().setSilhouette(0)
        for i in range(world.numTerrains()):
            world.terrain(i).appearance().setSilhouette(0)

        world.readFile('camera.rob')
        robot = world.robot(0)
        sensor = robot.sensor(0)
        vis.add("world",world)
        counters = [0,0,total_image_count]
        def loop_through_sensors(world=world,sensor=sensor,
            world_camera_viewpoints=world_camera_viewpoints,
            index=index,counters=counters):
            viewno = counters[0]
            variation = counters[1]
            total_count = counters[2]
            print("Generating data for sensor view",viewno,"variation",variation)
            sensor_xform = world_camera_viewpoints[viewno]

            #TODO: problem 1.B: perturb camera and change colors
            sensing.set_sensor_xform(sensor,sensor_xform)
            
            rgb_filename = "image_dataset/color_%04d_var%04d.png"%(total_count,variation)
            depth_filename = "image_dataset/depth_%04d_var%04d.png"%(total_count,variation)
            grasp_filename = "image_dataset/grasp_%04d_var%04d.png"%(total_count,variation)
   
            #Generate sensor images
            sensor.kinematicReset()
            sensor.kinematicSimulate(world,0.01)
            print("Done with kinematic simulate")
            rgb,depth = sensing.camera_to_images(sensor)
            print("Saving to",rgb_filename,depth_filename)
            Image.fromarray(rgb).save(rgb_filename)
            depth_scale = 8000
            depth_quantized = (depth * depth_scale).astype(np.uint32)
            Image.fromarray(depth_quantized).save(depth_filename)
            
            with open("image_dataset/metadata.csv",'a') as f:
                line = "{},{},{},{},{},{},{}\n".format(index,viewno,loader.write(sensor_xform,'RigidTransform'),os.path.basename(rgb_filename),os.path.basename(depth_filename),os.path.basename(grasp_filename),variation)
                f.write(line)
            
            #calculate grasp map and save it
            grasp_map = make_grasp_map(world,total_count)
            grasp_map_quantized = (grasp_map*255).astype(np.uint8)
            channels = ['score','opening','axis_heading','axis_elevation']
            for i,c in enumerate(channels):
                base,ext=os.path.splitext(grasp_filename)
                fn = base+'_'+c+ext
                Image.fromarray(grasp_map_quantized[:,:,i]).save(fn)
        
            #loop through variations and views
            counters[1] += 1
            if counters[1] >= max_variations:  
                counters[1] = 0
                counters[0] += 1
                if counters[0] >= len(world_camera_viewpoints):
                    vis.show(False)
                counters[2] += 1
        print("Running render loop")
        vis.loop(callback=loop_through_sensors)
        total_image_count = counters[2]


#some globals
gripper_world = None
gripper_robot = None
grasp_db = None
grasps_feasible = dict()

def make_grasp_map(world,id):
    """Estimates a grasp quality and regression map for a sensor image.

    Input:
        world (WorldModel): contains a robot and a sensor

    Output: a 4-channel numpy array of size (w,h,3) with 
    w x h the sensor dimensions and all values in the range [0,1].

    The first channel encodes the quality.
    
    The second encodes the grasp opening amount.
    
    The third and fourth encode the orientation of the grasp
    relative to the camera frame, as a heading and elevation.
    
    Make
    sure to note how you've transformed these quantities to
    [0,1]!  These will be needed for decoding.
    """
    robot = world.robot(0)
    sensor = robot.sensor(0)
    sensor_xform = sensing.get_sensor_xform(sensor,robot)
    w = int(sensor.getSetting("xres"))
    h = int(sensor.getSetting("yres"))
    #You'll be filling this image out
    result = np.zeros((h,w,4))
    result[:,:,3] = 0.5

    #this shows how to go from a point in space to a pixel
    # point = (0,0,0)
    # projected = sensing.camera_project(sensor,robot,point)
    # if projected is None:
    #     pass
    # else:
    #     x,y,z = projected
    #     if x < 0 or x > w or y < 0 or y > h: 
    #         pass
    #     else:
    #         result[int(y),int(x),0]=1
    # result[50,10,0]=1

    #load the gripper info and grasp database
    source_gripper = robotiq_85
    global gripper_robot,gripper_world,grasp_db
    if gripper_robot is None:
        gripper_world = WorldModel()
        if not gripper_world.readFile(source_gripper.klampt_model):
            raise ValueError("Unable to read gripper file "+source_gripper.klampt_model)
        gripper_robot = gripper_world.robot(0)
    gripper_geom_open = source_gripper.get_geometry(gripper_robot,source_gripper.open_config)

    if grasp_db is None:
        db = grasp_database.GraspDatabase(source_gripper)
        if not db.load("../data/grasps/robotiq_85_more_sampled_grasp_db.json"):
            raise RuntimeError("Can't load grasp database?")
        grasp_db = db
    
    for i in range(world.numRigidObjects()):
        obj = world.rigidObject(i)
        ycb_obj = obj.getName()
        if ycb_obj not in grasp_db.object_to_grasps:
            print("Can't find object",ycb_obj,"in database")
            print("Database only contains objects",grasp_db.objects)
            raise ValueError()
        grasps = grasp_db.object_to_grasps[ycb_obj]
        gripper_tip = vectorops.madd(source_gripper.center,source_gripper.primary_axis,source_gripper.finger_length)
        num_grasps_valid = 0
        for gindex,g in enumerate(grasps):
            #TODO: problem 1B: do collision testing of gripper_geom
            found_approach = False
            if (id,ycb_obj,gindex) in grasps_feasible:
                found_approach = grasps_feasible[(id,ycb_obj,gindex)]
            else:
                #this is the Geometry3D corresponding to the robot at the opening configuration
                gripper_geom = source_gripper.get_geometry(gripper_robot,g.finger_config)
                local_pt,world_pt = g.ik_constraint.getPosition()
                local_axis,world_axis = g.ik_constraint.getRotationAxis()
                #TODO: put your code here
                found_approach = True
                grasps_feasible[(id,ycb_obj,gindex)] = found_approach
            if not found_approach:
                continue
            num_grasps_valid += 1
    
            Tfixed = g.ik_constraint.closestMatch(*se3.identity())
            Tworld = se3.mul(obj.getTransform(),Tfixed)
            gworld = se3.apply(Tworld,gripper_tip)
            projected = sensing.camera_project(sensor,robot,gworld)
            if projected is not None:
                x,y,z = projected
                x = int(x)
                y = int(y)
                if x < 0 or x >= w or y < 0 or y >= h: 
                    continue
                gripper_opening_axis = so3.apply(Tworld[0],source_gripper.secondary_axis)
                gripper_opening_axis_cam = so3.apply(so3.inv(sensor_xform[0]),gripper_opening_axis)
                if gripper_opening_axis_cam[1] < 0:
                    gripper_opening_axis_cam = vectorops.mul(gripper_opening_axis_cam,-1)
                gripper_axis_heading = math.atan2(gripper_opening_axis_cam[1],gripper_opening_axis_cam[0])
                xy = vectorops.norm(gripper_opening_axis_cam[:2])
                if xy < 1e-7:
                    gripper_axis_elevation = math.pi*0.5
                else:
                    gripper_axis_elevation = math.atan(gripper_opening_axis_cam[2]/xy)

                score0 = math.exp(-g.score)
                window = 10
                std = 5
                for u in range(-window,window+1):
                    if y+u < 0 or y+u >= h: continue
                    for v in range(-window,window+1):
                        if x+v < 0 or x+v >= w: continue
                        score = score0*math.exp(-(u**2+v**2)/(2*std**2))
                        if score > result[y+u,x+v,0]:
                            result[y+u,x+v,:] = [score,source_gripper.config_to_opening(g.finger_config),
                                gripper_axis_heading/math.pi,gripper_axis_elevation/math.pi+0.5]
        print("{}/{} grasps valid for object {}".format(num_grasps_valid,len(grasps),ycb_obj))

    return result


if __name__ == '__main__':
    if len(sys.argv) > 1:
        PROBLEM = sys.argv[1]
    #load the robot / world file
    if PROBLEM=='1a':
        gen_grasp_worlds()
    elif PROBLEM == '1b':
        gen_grasp_images()
    else:
        raise ValueError("Invalid PROBLEM?")

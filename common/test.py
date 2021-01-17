import world_generator
import grasp_edit
from klampt import WorldModel
from klampt import vis
from klampt.math import vectorops,so3,se3
from klampt.model.create import pile
from klampt.io import resource
from known_grippers import *

#load the world model
w = WorldModel()
if not w.readFile(robotiq_140_trina_left.klampt_model):
    print("Couldn't load",robotiq_140_trina_left.klampt_model)
    exit(1)

import os
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../data'))
resource.setDirectory(os.path.join(data_dir,'resources/TRINA'))
qhome = resource.get('home.config')
w.robot(0).setConfig(qhome)

#create a box with objects in it
box = world_generator.make_box(w,0.5,0.5,0.1)
box.geometry().translate((1,0,0.8))
objects_in_box = []
for i in range(1):
    g = world_generator.make_ycb_object('random',w)
    objects_in_box.append(g)

grasp_edit.grasp_edit_ui(robotiq_140_trina_right,objects_in_box[0])

#visualize the right gripper
robotiq_140_trina_right.add_to_vis(w.robot(0))
#look at the disembodied right gripper
g = robotiq_140_trina_right.get_geometry(w.robot(0))
g.setCurrentTransform(*se3.from_translation([1,0,0]))
vis.add("geom",g,color=(1,0,0,0.5))

#pile.make_object_arrangement(w,box,objects_in_box,max_iterations=1000)
pile.make_object_pile(w,box,objects_in_box,verbose=0,visualize=False)
for i,o1 in enumerate(objects_in_box):
    for (j,o2) in enumerate(objects_in_box[:i]):
        print("Collision",o1.getName(),"-",o2.getName(),"?",o1.geometry().collides(o2.geometry()))

for i,g in enumerate(objects_in_box):
    vis.add("ycb_object"+str(i),g)
vis.add("box",box)

#pop up the visualization
vis.show()
vis.spin(float('inf'))
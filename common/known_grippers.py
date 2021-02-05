"""Registers all known grippers.

If you run this as the main file with no arguments, it will display all the
known grippers.

If you run this with one or more arguments, it will display the named grippers.
"""

from gripper import GripperInfo
import os
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../data'))

robotiq_85 = GripperInfo("robotiq_85",0,[1,2,3,4,5,6,7,8],[0],
    klampt_model=os.path.join(data_dir,"robots/robotiq_85.rob"))
robotiq_85.center = (0,0,0.1)
robotiq_85.primary_axis = (0,0,1)
robotiq_85.secondary_axis = (1,0,0)
robotiq_85.finger_length = 0.06
robotiq_85.finger_depth = 0.01
robotiq_85.finger_width = 0.02
robotiq_85.maximum_span = 0.085 - 0.01
robotiq_85.minimum_span = 0
robotiq_85.open_config = [0]*8
robotiq_85.closed_config = [0.723,0,0.723,-0.723,-0.723,0.723,-0.723,0 ]

robotiq_140 = GripperInfo("robotiq_140",0,[1,2,4,6,7,9],[0],
    klampt_model=os.path.join(data_dir,"robots/robotiq_140.rob"))
robotiq_140.center = (0,0,0.1)
robotiq_140.primary_axis = (0,0,1)
robotiq_140.secondary_axis = (0,1,0)
robotiq_140.finger_length = 0.12
robotiq_140.finger_depth = 0.01
robotiq_140.finger_width = 0.02
robotiq_140.maximum_span = 0.140 - 0.01
robotiq_140.minimum_span = 0
robotiq_140.open_config = [0]*6
robotiq_140.closed_config = [0.7,0.7,0.7,0.7,0.7,0.7]

robotiq_85_kinova_gen3 = GripperInfo.mounted(robotiq_85,os.path.join(data_dir,"robots/kinova_with_robotiq_85.urdf"),"gripper:Link_0","robotiq_85-kinova_gen3")

robotiq_140_trina_left = GripperInfo.mounted(robotiq_140,os.path.join(data_dir,"robots/TRINA.urdf"),"left_gripper:base_link","robotiq_140-trina-left")
robotiq_140_trina_right = GripperInfo.mounted(robotiq_140,os.path.join(data_dir,"robots/TRINA.urdf"),"right_gripper:base_link","robotiq_140-trina-right")

if __name__ == '__main__':
    from klampt import vis
    import sys
    if len(sys.argv) == 1:
        grippers = [i for i in GripperInfo.all_grippers]
        print("ALL GRIPPERS",grippers)
    else:
        grippers = sys.argv[1:]

    for i in grippers:
        g = GripperInfo.get(i)
        print("SHOWING GRIPPER",i)
        g.add_to_vis()
        vis.setWindowTitle(i)
        def setup():
            vis.show()
        def cleanup():
            vis.show(False)
            vis.clear()
        vis.loop(setup=setup,cleanup=cleanup)

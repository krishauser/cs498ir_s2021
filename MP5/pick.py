from planning import *
from klampt.model import collide
from klampt.model import ik
from klampt.model.trajectory import Trajectory,RobotTrajectory
from klampt.math import vectorops,so3,se3

finger_pad_links = ['gripper:Link 4','gripper:Link 6']

def is_collision_free_grasp(world,robot,object):
    #TODO: you might want to fix this to ignore collisions between finger pads and the object
    if robot.selfCollides():
        return False
    for i in range(world.numTerrains()):
        for j in range(robot.numLinks()):
            if robot.link(j).geometry().collides(world.terrain(i).geometry()):
                return False
    for i in range(world.numRigidObjects()):
        for j in range(robot.numLinks()):
            if robot.link(j).geometry().collides(world.rigidObject(i).geometry()):
                return False
    return True

def retract(robot,gripper,amount,local=True):
    """Retracts the robot's gripper by a vector `amount`.

    if local=True, amount is given in local coordinates.  Otherwise, its given in
    world coordinates.
    """
    if not isinstance(gripper,(int,str)):
        gripper = gripper.base_link
    link = robot.link(gripper)
    Tcur = link.getTransform()
    if local:
        amount = so3.apply(Tcur[0],amount)
    obj = ik.objective(link,R=Tcur[0],t=vectorops.add(Tcur[1],amount))
    res = ik.solve(obj)
    if not res:
        return None
    return robot.getConfig()

def plan_pick_one(world,robot,object,gripper,grasp):
    """
    Plans a picking motion for a given object and a specified grasp.

    Arguments:
        world (WorldModel): the world, containing robot, object, and other items that
            will need to be avoided.
        robot (RobotModel): the robot in its current configuration
        object (RigidObjectModel): the object to pick.
        gripper (GripperInfo): the gripper.
        grasp (Grasp): the desired grasp. See common/grasp.py for more information.

    Returns:
        None or (transit,approach,lift): giving the components of the pick motion.
        Each element is a RobotTrajectory.  (Note: to convert a list of milestones
        to a RobotTrajectory, use RobotTrajectory(robot,milestones=milestones)

    Tip:
        vis.debug(q,world=world) will show a configuration.
    """
    qstart = robot.getConfig()
    grasp.ik_constraint.robot = robot  #this makes it more convenient to use the ik module
    
    #TODO solve the IK problem for qgrasp?
    qgrasp = qstart

    qgrasp = grasp.set_finger_config(qgrasp)  #open the fingers the right amount
    qopen = gripper.set_finger_config(qgrasp,gripper.partway_open_config(1))   #open the fingers further

    qpregrasp = qopen   #TODO solve the retraction problem for qpregrasp?

    qstartopen = gripper.set_finger_config(qstart,gripper.partway_open_config(1))  #open the fingers of the start to match qpregrasp
    robot.setConfig(qstartopen)
    transit = feasible_plan(world,robot,qpregrasp)   #decide whether to use feasible_plan or optimizing_plan
    if not transit:
        return None

    #TODO: not a lot of collision checking going on either...

    qlift = qgrasp
    return RobotTrajectory(robot,milestones=[qstart]+transit),RobotTrajectory(robot,milestones=[qpregrasp,qopen,qgrasp],RobotTrajectory(robot,milestones=[qgrasp,qlift])


def plan_pick_grasps(world,robot,object,gripper,grasps):
    """
    Plans a picking motion for a given object and a set of possible grasps, sorted
    in increasing score order.

    Arguments:
        world (WorldModel): the world, containing robot, object, and other items that
            will need to be avoided.
        robot (RobotModel): the robot in its current configuration
        object (RigidObjectModel): the object to pick.
        gripper (GripperInfo): the gripper.
        grasp (Grasp): the desired grasp. See common/grasp.py for more information.

    Returns:
        None or (transit,approach,lift): giving the components of the pick motion.
        Each element is a RobotTrajectory.  (Note: to convert a list of milestones
        to a RobotTrajectory, use RobotTrajectory(robot,milestones=milestones)

    Tip:
        vis.debug(q,world=world) will show a configuration.
    """
    #TODO: implement me


class StepResult:    
    FAIL = 0
    COMPLETE = 1
    CONTINUE = 2
    CHILDREN = 3
    CHILDREN_AND_CONTINUE = 4

class PQNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __lt__(self, other):
        return self.value < other.value

    def __str__(self):
        return str("{} : {}".format(self.key, self.value))

class MultiStepPlanner:
    """A generic multi step planner that can be subclassed to implement multi-step planning
    behavior.

    Subclass will need to:
        - implement choose_item() or provide a sequence of items to solve to the initializer.
        - implement solve_item() to complete items of the plan
        - implement score() for a custom heuristic (lower scores are better)
    
    """
    def __init__(self,items=None):
        self.W = [PQNode({},0)]
        if items is not None:
            self.items = items
        else:
            self.items = []
        self.pending_solutions = []

    def choose_item(self,plan):
        """Returns an item that is not yet complete in plan.  Default loops through items
        provided in the constructor (fixed order), but a subclass can do more sophisticated
        reasoning.

        Args:
            plan (dict): a partial plan.
        """
        for i in self.items:
            if i not in plan:
                return i
        return None

    def on_solve(self,plan,item,soln):
        """Callback that can do logging, etc."""
        pass

    def solve_item(self,plan,item):
        """Attempts to solve the given item in the plan.  Can tell the high-level planner to
        stop this line of reasoning (FAIL), the solution(s) completes the plan (COMPLETE), this item
        needs more time to solve (CONTINUE), several options have been generated and we wish
        stop planning this item (CHILDREN), or options have been generated and we wish to
        continue trying to generate more (CHILDREN_AND_CONTINUE).

        If you want to cache something for this item for future calls, put it under plan['_'+item].

        Args:
            plan (dict): a partial plan
            item (str): an item to solve.

        Returns:
            tuple: a pair (status,children) where status is one of the StepResult codes
            FAIL, COMPLETE, CONTINUE, CHILDREN, CHILDREN_AND_CONTINUE, and
            children is a list of solutions that complete more of the plan.
        """
        return StepResult.FAIL,[]
    
    def score(self,plan):
        """Returns a numeric score for the plan.  Default checks the score"""
        num_solved = len([k for k in plan if not k.startswith('_')])
        return plan.get('_solve_time',0) - 0.2*num_solved

    def assemble_result(self,plan):
        """Turns a dict partial plan into a complete result.  Default just
        returns the plan dict.
        """
        return dict((k,v) for (k,v) in plan.items() if not k.startwith('_'))

    def solve(self,tmax=float('inf')):
        """Solves the whole plan using least-commitment planning. 

        Can be called multiple times to produce multiple solutions.
        """
        import heapq,copy
        if self.pending_solutions:
            soln = self.pending_solutions.pop(0)
            return self.assemble_result(soln)
        tstart = time.time()
        while len(self.W)>0:
            if time.time()-tstart > tmax:
                return None
            node = heapq.heappop(self.W)
            plan = node.key
            prio = node.value
            item = self.choose_item(plan)
            if item is None:
                #no items left, done
                return self.assemble_result(plan)
            #print("Choosing item",item,"priority",prio)
            t0 = time.time()
            status,children = self.solve_item(plan,item)
            t1 = time.time()
            plan.setdefault('_solve_time',0)
            plan['_solve_time'] += t1-t0
            if status == StepResult.FAIL:
                continue
            elif status == StepResult.COMPLETE:
                assert len(children) > 0,"COMPLETE was returned but without an item solution?"
                soln = children[0]
                self.on_solve(plan,item,soln)
                plan[item] = soln
                for soln in children[1:]:
                    self.on_solve(plan,item,soln)
                    child = copy.copy(plan)
                    child[item] = soln
                    self.pending_solutions.append(child)
                return self.assemble_result(plan)
            else:
                if status == StepResult.CHILDREN_AND_CONTINUE or status == StepResult.CONTINUE:
                    heapq.heappush(self.W,PQNode(plan,self.score(plan)))
                for soln in children:
                    self.on_solve(plan,item,soln)
                    child = copy.copy(plan)
                    child[item] = soln
                    child['_solve_time'] = 0
                    heapq.heappush(self.W,PQNode(child,self.score(child)))
                    #print("Child priority",self.score(child))
        return None

class PickPlanner(MultiStepPlanner):
    """For problem 2C
    """
    def __init__(self,world,robot,object,gripper,grasps):
        MultiStepPlanner.__init__(self,['grasp','qgrasp','approach','transit','lift'])
        self.qstart = robot.getConfig()
        self.world=world
        self.robot=robot
        self.object=object
        self.gripper=gripper
        self.grasps=grasps

    def solve_qgrasp(self,grasp):
        #TODO: solve for the grasping configuration 
        return None

    def solve_approach(self,grasp,qgrasp):
        #TODO: solve for the approach 
        return None

    def solve_transit(self,qpregrasp):
        #TODO: solve for the transit path
        return None

    def solve_lift(self,qgrasp):
        #TODO: solve for the lifting configurations
        return None

    def solve_item(self,plan,item):
        """Returns a pair (status,children) where status is one of the codes FAIL,
        COMPLETE, CHILDREN, CHILDREN_AND_SELF, and children is a list of solutions
        that complete more of the plan.
        """
        if item == 'grasp':
            print("Assigning grasps")
            return StepResult.CHILDREN,self.grasps
        if item == 'qgrasp':
            print("Planning IK configuration")
            grasp = plan['grasp']
            result = self.solve_qgrasp(grasp)
            if result is None:
                print("IK solve failed... trying again")
                return StepResult.CONTINUE,[]
            else:
                print("IK solve succeeded, moving on to pregrasp planning")
                return StepResult.CHILDREN_AND_CONTINUE,[result]
        if item == 'approach':
            print("Planning approach")
            grasp = plan['grasp']
            qgrasp = plan['qgrasp']
            result = self.solve_approach(grasp,qgrasp)
            if result is None:
                return StepResult.FAIL,[]
            return StepResult.CHILDREN,[result]
        if item == 'transit':
            print("Transit planning")
            qpregrasp = plan['approach'][0]
            result = self.solve_transit(qpregrasp)
            if result is None:
                print("Transit planning failed")
                return StepResult.CONTINUE,[]
            else:
                print("Transit planning succeeded!")
                return StepResult.CHILDREN,[result]
        if item == 'lift':
            qgrasp = plan['qgrasp']
            result = self.solve_lift(qgrasp)
            if result is None:
                return StepResult.FAIL,[]
            return StepResult.CHILDREN,[result]
        raise ValueError("Invalid item "+item)


    def score(self,plan):
        """Priority score for a partial plan"""
        #TODO: prioritize grasps with low score
        return MultiStepPlanner.score(self,plan)

    def assemble_result(self,plan):
        """Get the results from a partial plan"""
        qstart = self.qstart
        transit = plan['transit']
        approach = plan['approach']        
        qgrasp = plan['qgrasp']
        qlift = plan['lift']
        #TODO: construct the RobotTrajectory triple as in plan_pick_one
        return plan

    
def plan_pick_multistep(world,robot,object,gripper,grasps):
    """
    Plans a picking motion for a given object and a set of possible grasps, sorted
    in increasing score order.

    Arguments:
        world (WorldModel): the world, containing robot, object, and other items that
            will need to be avoided.
        robot (RobotModel): the robot in its current configuration
        object (RigidObjectModel): the object to pick.
        gripper (GripperInfo): the gripper.
        grasp (Grasp): the desired grasp. See common/grasp.py for more information.

    Returns:
        None or (transit,approach,lift): giving the components of the pick motion.
        Each element is a RobotTrajectory.  (Note: to convert a list of milestones
        to a RobotTrajectory, use RobotTrajectory(robot,milestones=milestones)

    Tip:
        vis.debug(q,world=world) will show a configuration.
    """
    qstart = robot.getConfig()
    for grasp in grasps:
        grasp.ik_constraint.robot = robot  #this makes it more convenient to use the ik module
    planner = PickPlanner(world,robot,object,gripper,grasps)
    time_limit = 60
    return planner.solve(time_limit)

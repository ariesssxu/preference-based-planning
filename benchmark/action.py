from video_recorder import video_recorder
from utils import robot_rotate_interpolation, camera_rotate_interpolation

import numpy as np
import random
import os

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject
from omnigibson import object_states
from omnigibson.robots import Fetch
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from omnigibson.utils.asset_utils import get_og_avg_category_specs
from omnigibson.systems import get_system
from omnigibson_modified import TraversableMap
from omnigibson.utils.control_utils import IKSolver
import omnigibson.utils.transform_utils as T
from omnigibson.sensors import VisionSensor

from pyquaternion import Quaternion
from transforms3d.euler import quat2euler
import matplotlib
from utils import (
    robot_rotate_interpolation, 
    camera_rotate_interpolation,
    path_interpolation,
    get_robot_view,
    get_robot_position,
    get_robot_orientation,
    get_camera_orientation,
    get_robot_angle,
    get_camera_angle,
    get_arm_position,
    get_operation_position,
)
matplotlib.use('TkAgg')

# set to True if you want to see the robot moving
# This will influence the moving function 
SHOW_MOVING = True
SHOWING_ARM_MOVING = False

def robot_rotate(robot, obj, obj_grasp=None):
    if isinstance(obj, DatasetObject):
        robot_end = get_robot_angle(obj.get_position(), robot.get_position())
        robot_start = quat2euler(robot.get_orientation()[[3, 0, 1, 2]])[2]
        interpolation_num = 20
        robot_rotate_list = robot_rotate_interpolation(robot_start, robot_end, interpolation_num)
        for i in range(interpolation_num):
            robot.keep_still()
            pos = robot.get_position()
            pos[2] = 0.01
            robot.set_position_orientation(position=pos, orientation=robot_rotate_list[i])
            if isinstance(obj_grasp, DatasetObject):
                obj_grasp.set_position(robot.get_eef_position())
                obj_grasp.keep_still()
            # og.log.info(f"Current orientation: {ori}, target orientation: {end}, start orientation: {start}")
            robot.keep_still()
            og.sim.step()
    else:
        pos = robot.get_position()
        robot_end = get_robot_orientation(obj, pos)
        pos[2] = 0.01
        robot.set_position_orientation(position=pos, orientation=robot_end)
        if isinstance(obj_grasp, DatasetObject):
            obj_grasp.set_position(robot.get_eef_position())
            obj_grasp.keep_still()
        og.sim.step()

def camera_rotate(robot, obj, obj_grasp=None):
    if isinstance(obj, DatasetObject):
        for sensor in robot.sensors.values():
            if isinstance(sensor, VisionSensor):
                camera_position, camera_orientation = sensor.get_position_orientation()
        camera_end = get_camera_angle(obj.get_position(), camera_position)
        camera_angles = quat2euler(camera_orientation[[3, 0, 1, 2]])
        camera_start = camera_angles[0]
        interpolation_num = 20
        camera_rotate_list = camera_rotate_interpolation(camera_angles, camera_start, camera_end, interpolation_num)
        for i in range(interpolation_num):
            robot.keep_still()
            for sensor in robot.sensors.values():
                if isinstance(sensor, VisionSensor):
                    sensor.set_orientation(camera_rotate_list[i])
            if isinstance(obj_grasp, DatasetObject):
                obj_grasp.set_position(robot.get_eef_position())
                obj_grasp.keep_still()
            robot.keep_still()
            og.sim.step()
    else:
        for sensor in robot.sensors.values():
            if isinstance(sensor, VisionSensor):
                camera_position = sensor.get_position()
                camera_angles = quat2euler(sensor.get_orientation()[[3, 0, 1, 2]])
                camera_end = get_camera_orientation(camera_angles, obj, camera_position)
                sensor.set_orientation(camera_end)
        if isinstance(obj_grasp, DatasetObject):
            obj_grasp.set_position(robot.get_eef_position())
            obj_grasp.keep_still()

def move_arm_to_target(pos, robot):
    
    # Create the IK solver -- note that we are controlling both the trunk and the arm since both are part of the
    # controllable kinematic chain for the end-effector!
    robot.keep_still()
    # Since this demo aims to showcase how users can directly control the robot with IK, 
    # we will need to disable the built-in controllers in OmniGibson
    robot.control_enabled = False

    control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
    try:
        ik_solver = IKSolver(
            robot_description_path=robot.robot_arm_descriptor_yamls[robot.default_arm],
            robot_urdf_path=robot.urdf_path,
            reset_joint_pos=robot.get_joint_positions()[control_idx],
            eef_name=robot.eef_link_names[robot.default_arm],
        )
    except Exception as e:
        print(e)
        return 
        
    # Define a helper function for executing specific end-effector commands using the ik solver
    def execute_ik(pos, quat=None, max_iter=100):
        og.log.info("Querying joint configuration to current marker position")
        # Switch the target position so it is applied to the coordinate system of the robot
        inverse_ori = robot.get_orientation()
        inverse_ori[0] = -inverse_ori[0]
        inverse_ori[1] = -inverse_ori[1]
        inverse_ori[2] = -inverse_ori[2]
        robot_pos = Quaternion(inverse_ori[[3, 0, 1, 2]]).rotate(robot.get_position())
        target_pos = Quaternion(inverse_ori[[3, 0, 1, 2]]).rotate(pos)
        # Grab the joint positions in order to reach the desired pose target
        joint_pos = ik_solver.solve(
            target_pos=target_pos - robot_pos,
            target_quat=quat,
            tolerance_pos=0.002,
            max_iterations=max_iter,
            initial_joint_pos=robot.get_joint_positions()[control_idx],
        )
        if joint_pos is not None:
            og.log.info("Solution found. Setting new arm configuration.")
            robot.set_joint_positions(joint_pos, indices=control_idx, drive=True)
        else:
            og.log.info("EE position not reachable.")
        og.sim.step()

    # Execute it
    execute_ik(pos)
    # Make sure none of the joints are moving
    robot.keep_still()

def import_A_of_B(A, B=None):
    if B == None:
        obj_path = os.path.join(gm.DATASET_PATH, "objects", A)
        B = random.choice(os.listdir(obj_path))
    object = DatasetObject(
        name=A,
        category=A,
        model=B,
        fit_avg_dim_volume=True,
    )
    og.sim.import_object(object)
    
    for _ in range(1):
        og.sim.step()

    return object

def import_A_in_B(A, B):
    system = get_system(A)
    B.states[object_states.Filled].set_value(system, True)

    for _ in range(25):
        B.keep_still()
        og.sim.step()
    
    return system

def import_A_on_B(A, B):
    system = get_system(A)
    B.states[object_states.Covered].set_value(system, True)

    for _ in range(25):
        B.keep_still()
        og.sim.step()
    
    return system

def remove_A(A):
    # Stop the simulator and remove the object
    og.sim.stop()
    og.sim.remove_object(obj=A)

    for _ in range(1):
        og.sim.step()

def set_A_at_P(A, P):
    A.set_position_orientation(
         position=P,
         orientation=np.array([0, 0, 0, 1])
    )
    for _ in range(20):
        og.sim.step() 

def go_A(A, robot, trav_map, trav_map_size):
    # Get the start point and the target point of the path
    start = robot.get_position()[:-1]
    target = get_robot_position(A, trav_map, trav_map_size)[:-1]
    camera_rotate(robot, robot.get_eef_position())

    if SHOWING_ARM_MOVING:
        # Lift the arm
        move_arm_to_target(get_arm_position(robot), robot)
        for _ in range(20):
            # video_recorder.get_video(text=f"go {A.name}")
            og.sim.step()

    path = trav_map.get_shortest_path(
        floor=0, source_world=start, target_world=target, entire_path=True)[0]
    path = path_interpolation(path, 10)
    path.append(target)
    # Move the robot
    if SHOW_MOVING:
        for pos in path:
            robot_rotate(robot, pos)
            z = robot.get_position()[2]
            robot.set_position_orientation(position=[pos[0], pos[1], z])
            og.sim.step()
            video_recorder.get_video(text=f"move to {A.name}")
    else:
        robot_rotate(robot, path[-1])
        z = robot.get_position()[2]
        robot.set_position_orientation(position=[path[-1][0], path[-1][1], z])
        og.sim.step()
        video_recorder.get_video(text=f"move to {A.name}", path=path, movement=False)

    # Settle the object
    for _ in range(5):
        og.sim.step(np.array([]))

def go_A_with_B(A, B, robot, trav_map, trav_map_size):
    # Get the start point and the target point of the path
    start = robot.get_position()[:-1]
    target = get_robot_position(A, trav_map, trav_map_size)[:-1]
    # # Change the orientation of the robot so it towards the target
    # robot_rotate(robot, A, B)
    camera_rotate(robot, robot.get_eef_position())
    if SHOWING_ARM_MOVING:
    # Lift the arm
        move_arm_to_target(get_arm_position(robot), robot)
    for _ in range(10):
        video_recorder.get_video(text=f"move to {A.name} with {B.name}")
        B.set_position(robot.get_eef_position())
        B.keep_still()
        og.sim.step()
    
    # Get the shortest path from start to target
    path = trav_map.get_shortest_path(
        floor=0, source_world=start, target_world=target, entire_path=True)[0]
    path = path_interpolation(path, 10)
    path.append(target)
    # Move the robot
    if SHOW_MOVING: 
        for pos in path:
            robot_rotate(robot, pos, B)
            robot.set_position([pos[0], pos[1], robot.get_position()[2]])
            B.set_position(robot.get_eef_position())
            B.keep_still()
            # robot.release_grasp_immediately()
            video_recorder.get_video(text=f"move to {A.name} with {B.name}")
            og.sim.step()
    else:
        robot_rotate(robot, path[-1], B)
        robot.set_position([path[-1][0], path[-1][1], robot.get_position()[2]])
        B.set_position(robot.get_eef_position())
        B.keep_still()
        # robot.release_grasp_immediately()
        video_recorder.get_video(text=f"move to {A.name} with {B.name}", path=path, movement=False)
        og.sim.step()

    # Make sure none of the joints are moving
    robot.keep_still()

    # Keep things grasped not dropping
    B.keep_still()
        
def go_P_with_B(P, B, robot, trav_map, trav_map_size):
    # Get the start point and the target point of the path
    start = robot.get_position()[:-1]
    target = P[:-1]
    # # Change the orientation of the robot so it towards the target
    # robot_rotate(robot, P, B)
    camera_rotate(robot, robot.get_eef_position())
    if SHOWING_ARM_MOVING:
    # Lift the arm
        move_arm_to_target(get_arm_position(robot), robot)
    for _ in range(10):
        video_recorder.get_video(text=f"move with {B.name}")
        B.set_position(robot.get_eef_position())
        B.keep_still()
        og.sim.step()

    # Get the shortest path from start to target
    path = trav_map.get_shortest_path(
        floor=0, source_world=start, target_world=target, entire_path=True)[0]
    path = path_interpolation(path, 10)
    # Move the robot
    if SHOW_MOVING:
        for pos in path:
            B.set_position(robot.get_eef_position())
            B.keep_still()
            # robot.release_grasp_immediately()
            robot_rotate(robot, pos, B)
            robot.set_position([pos[0], pos[1], robot.get_position()[2]])
            B.set_position(robot.get_eef_position())
            video_recorder.get_video(text=f"move with {B.name}")
            og.sim.step()
    else:
        robot_rotate(robot, path[-1], B)
        robot.set_position([path[-1][0], path[-1][1], robot.get_position()[2]])
        B.set_position(robot.get_eef_position())
        B.keep_still()
        # robot.release_grasp_immediately()
        video_recorder.get_video(text=f"move with {B.name}", path=path, movement=False)
        og.sim.step()

    # Make sure none of the joints are moving
    robot.keep_still()

    # Keep things grasped not dropping
    B.keep_still()

    return [start, target, path]

def open_A(A, robot, B=None):
    # Set the orientation of robot so it towards to the target
    robot_rotate(robot, A, B)

    if SHOWING_ARM_MOVING:
        # Lift the arm
        move_arm_to_target(get_operation_position(A), robot)
        for _ in range(100):
            if isinstance(B, DatasetObject):
                B.set_position(robot.get_eef_position())
                B.keep_still()
            video_recorder.get_video(text=f"open {A.name}")
            og.sim.step()
    else:
        for _ in range(10):
            if isinstance(B, DatasetObject):
                B.set_position(robot.get_eef_position())
                B.keep_still()
            video_recorder.get_video(text=f"open {A.name}")
            og.sim.step()
    
    camera_rotate(robot, A, B)
    for _ in range(5):
        if isinstance(B, DatasetObject):
            B.set_position(robot.get_eef_position())
            B.keep_still()
        video_recorder.get_video(text=f"open {A.name}")
        og.sim.step()
    
    A.states[object_states.Open].set_value(new_value=True, fully=True)

    # Settle the door
    for _ in range(20):
        if isinstance(B, DatasetObject):
            B.set_position(robot.get_eef_position())
            B.keep_still()
        video_recorder.get_video(text=f"open {A.name}")
        og.sim.step()

def close_A(A, robot, B=None, way="pick"):
    # # Set the orientation of robot so it towards to the target
    # robot_rotate(robot, A)

    # # Lift the arm
    # arm_pos = get_arm_position(robot)
    # arm_pos[2] = A.get_position()[2]
    # move_arm_to_target(arm_pos, robot)
    for _ in range(30):
        video_recorder.get_video(text=f"close {A.name}")
        og.sim.step()

    A.states[object_states.Open].set_value(new_value=False, fully=True)

    # Settle the door
    for _ in range(20):
        if isinstance(B, DatasetObject):
            if way == "pick":
                B.set_position(robot.get_eef_position())
                B.keep_still()
            elif way == "place":
                B.set_position(A.get_position())
                B.keep_still()
        video_recorder.get_video(text=f"close {A.name}")
        og.sim.step()
    
def pick_A(A, robot):
    # Set the orientation of robot so it towards to the target
    robot_rotate(robot, A)

    # Move the arm to the target object
    if SHOWING_ARM_MOVING:
        move_arm_to_target(A.get_position(), robot)
        for _ in range(100):
            video_recorder.get_video(text=f"pick {A.name}")
            og.sim.step()
    else:
        for _ in range(10):
            video_recorder.get_video(text=f"pick {A.name}")
            og.sim.step()

    # Grasp the target object
    # robot.release_grasp_immediately()

    # # Lift the arm
    # move_arm_to_target(get_arm_position(robot), robot)
    camera_rotate(robot, robot.get_eef_position(), A)
    for _ in range(20):
        A.set_position_orientation(
            position=robot.get_eef_position(),
            orientation=[0, 0, 0, 1],
        )
        # Keep things grasped not dropping
        A.keep_still()
        video_recorder.get_video(text=f"pick {A.name}")
        og.sim.step()
        
def place_A_on_B(A, B, robot):
    # Set the orientation of robot so it towards to the target
    robot_rotate(robot, B, A)

    final_pos = B.get_position()
    final_pos[2] += 0.5 * A.native_bbox[2] + 0.5 * B.native_bbox[2] + 0.05
    
    if SHOWING_ARM_MOVING:
        arm_pos = final_pos
        # Move the arm to the target pos
        move_arm_to_target(arm_pos, robot)
        # Place the object on the receptacle
        for _ in range(100):
            A.set_position_orientation(
                position=robot.get_eef_position(),
                orientation=[0, 0, 0, 1],
            )
            # Keep things grasped not dropping
            A.keep_still()
            B.keep_still()
            video_recorder.get_video(text=f"place {A.name} on {B.name}")
            og.sim.step()
        # Lift the arm
        move_arm_to_target(arm_pos + np.array([0, 0, 0.2]), robot)
    else:
        for _ in range(10):
            A.set_position_orientation(
                position=robot.get_eef_position(),
                orientation=[0, 0, 0, 1],
            )
            # Keep things grasped not dropping
            A.keep_still()
            B.keep_still()
            video_recorder.get_video(text=f"place {A.name} on {B.name}")
            og.sim.step()
    
    camera_rotate(robot, final_pos + np.array([0, 0, -0.05]))
    for _ in range(40):
        A.set_position_orientation(
            position=final_pos + np.array([0, 0, -0.05]),
            orientation=[0, 0, 0, 1],
        )
        # Keep things grasped not dropping
        B.keep_still()
        video_recorder.get_video(text=f"place {A.name} on {B.name}")
        og.sim.step()

def place_A_in_B(A, B, robot):
    # Set the orientation of robot so it towards to the target
    robot_rotate(robot, B, A)

    # Move the arm to the target pos
    final_pos = B.get_position()
    final_pos[2] += 0.05

    if SHOWING_ARM_MOVING:
        arm_pos = final_pos
        move_arm_to_target(arm_pos, robot)
        # Place the object on the receptacle
        for _ in range(200):
            A.set_position_orientation(
                position=robot.get_eef_position(),
                orientation=[0, 0, 0, 1],
            )
            # Keep things grasped not dropping
            A.keep_still()  
            B.keep_still()
            video_recorder.get_video(text=f"place {A.name} in {B.name}")
            og.sim.step()
            
        # Lift the arm
        move_arm_to_target(arm_pos + np.array([0, 0, 0.2]), robot)
    else:
        for _ in range(10):
            A.set_position_orientation(
                position=robot.get_eef_position(),
                orientation=[0, 0, 0, 1],
            )
            # Keep things grasped not dropping
            A.keep_still()  
            B.keep_still()
            video_recorder.get_video(text=f"place {A.name} in {B.name}")
            og.sim.step()
    
    camera_rotate(robot,final_pos + np.array([0, 0, -0.05]))
    for _ in range(40):
        A.set_position_orientation(
            position=final_pos + np.array([0, 0, -0.05]),
            orientation=[0, 0, 0, 1],
        )
        # Keep things grasped not dropping
        A.keep_still()
        B.keep_still()
        video_recorder.get_video(text=f"place {A.name} in {B.name}")
        og.sim.step()

def place_A_at_P(A, P, robot, B=None):
    # Set the orientation of robot so it towards to the target
    robot_rotate(robot, P, A)

    # Move the arm to the target pos
    final_pos = P
    final_pos[2] += 0.05

    if SHOWING_ARM_MOVING:
        arm_pos = final_pos
        move_arm_to_target(arm_pos, robot)
        # Place the object on the receptacle
        for _ in range(100):
            A.set_position_orientation(
                position=robot.get_eef_position(),
                orientation=[0, 0, 0, 1],
            )
            # Keep things grasped not dropping
            A.keep_still()
            if isinstance(B, DatasetObject):
                B.keep_still()
            video_recorder.get_video(text=f"place {A.name}")
            og.sim.step()
            
        # Lift the arm
        move_arm_to_target(arm_pos + np.array([0, 0, 0.2]), robot)
    else:
        for _ in range(10):
            A.set_position_orientation(
                position=robot.get_eef_position(),
                orientation=[0, 0, 0, 1],
            )
            # Keep things grasped not dropping
            A.keep_still()
            if isinstance(B, DatasetObject):
                B.keep_still()
            video_recorder.get_video(text=f"place {A.name}")
            og.sim.step()
    
    camera_rotate(robot, final_pos + np.array([0, 0, -0.05]))
    for _ in range(40):
        A.set_position_orientation(
            position=final_pos + np.array([0, 0, -0.05]),
            orientation=[0, 0, 0, 1],
        )
        # Keep things grasped not dropping
        A.keep_still()
        if isinstance(B, DatasetObject):
            B.keep_still()
        video_recorder.get_video(text=f"place {A.name} at ")
        og.sim.step()

def cut_A_with_B(A, B, robot):
    # Set the orientation of robot so it towards to the target
    robot_rotate(robot, A, B)

    target_pos = A.get_position() + np.array([-0.15, 0.0, 0.1])
    if SHOWING_ARM_MOVING:
        move_arm_to_target(target_pos, robot)
        for _ in range(100):
            B.set_position_orientation(
                position=robot.get_eef_position(),
                orientation=T.euler2quat([-np.pi / 2, 0, 0]),
            )
            B.keep_still()
            video_recorder.get_video(text=f"cut {A.name} with {B.name}")
            og.sim.step()
    else:
        for _ in range(10):
            B.set_position_orientation(
                position=robot.get_eef_position(),
                orientation=T.euler2quat([-np.pi / 2, 0, 0]),
            )
            B.keep_still()
            video_recorder.get_video(text=f"cut {A.name} with {B.name}")
            og.sim.step()

    camera_rotate(robot, A, B)        
    B.keep_still()
    B.set_position_orientation(
        position=A.get_position() +  np.array([0.0, 0.0, A.native_bbox[2] * 0.5 + B.native_bbox[2] * 0.5]),
        orientation=T.euler2quat([-np.pi / 2, 0, 0]),
    )
    for _ in range(5):
        video_recorder.get_video(text=f"cut {A.name} with {B.name}")
        og.sim.step()
          
    B.keep_still()
    B.set_position_orientation(
        position=A.get_position() +  np.array([0.5, 0.0, 0.0]),
        orientation=T.euler2quat([0, 0, 0]),
    )
    for _ in range(10):
        video_recorder.get_video(text=f"cut {A.name} with {B.name}")
        og.sim.step()

def toggle_on_A(A, robot, B=None):
    # # Set the orientation of robot so it towards to the target
    # robot_rotate(robot, A, B)

    # # Lift the arm
    # move_arm_to_target(get_operation_position(A), robot)
    # for _ in range(100):
    #     video_recorder.get_video(text=f"toggle on {A.name}")
    #     og.sim.step()

    A.states[object_states.ToggledOn].set_value(True)

    # Settle the state
    for _ in range(20):
        if isinstance(B, DatasetObject):
            B.set_position(robot.get_eef_position())
            B.keep_still()
        video_recorder.get_video(text=f"toggle on {A.name}")
        og.sim.step()

def toggle_off_A(A, robot):
    # # Set the orientation of robot so it towards to the target
    # robot_rotate(robot, A)

    A.states[object_states.ToggledOn].set_value(False)

    # Settle the state
    for _ in range(20):
        video_recorder.get_video(text=f"toggle off {A.name}")
        og.sim.step()

def add_water_to_A_with_B(A, B, robot):
    # Set the orientation of robot so it towards to the target
    robot_rotate(robot, B, A)
    # Lift the arm
    if SHOWING_ARM_MOVING:
        move_arm_to_target(B.states[object_states.ToggledOn].link.get_position(), robot)
        for _ in range(200):
            A.set_position_orientation(
                position=B.states[object_states.ParticleSource].link.get_position() - np.array([0, 0, A.native_bbox[2] * 0.5 + 0.1]),
                orientation=[0, 0, 0, 1],
            )
            A.keep_still()
            video_recorder.get_video(text=f"add water to {A.name} with {B.name}")
            og.sim.step()
    else:
        for _ in range(10):
            A.set_position_orientation(
                position=B.states[object_states.ParticleSource].link.get_position() - np.array([0, 0, A.native_bbox[2] * 0.5 + 0.1]),
                orientation=[0, 0, 0, 1],
            )
            A.keep_still()
            video_recorder.get_video(text=f"add water to {A.name} with {B.name}")
            og.sim.step()

    camera_rotate(robot, B.states[object_states.ParticleSource].link.get_position())
    if SHOWING_ARM_MOVING:
    # Lift the arm
        arm_pos = get_arm_position(robot)
        arm_pos[2] = B.states[object_states.ParticleSource].link.get_position()[2] + 0.1
        move_arm_to_target(arm_pos, robot)

    B.states[object_states.ToggledOn].set_value(True)

    for _ in range(20):
        A.set_position_orientation(
            position=B.states[object_states.ParticleSource].link.get_position() - np.array([0, 0, A.native_bbox[2] * 0.5 + 0.1]),
            orientation=[0, 0, 0, 1],
        )
        A.keep_still()
        video_recorder.get_video(text=f"add water to {A.name} with {B.name}")
        og.sim.step()

    B.states[object_states.ToggledOn].set_value(False)

    for _ in range(40):
        A.keep_still()
        video_recorder.get_video(text=f"add water to {A.name} with {B.name}")
        og.sim.step()

def wait_N(N, robot):
    for _ in range(N):
        video_recorder.get_video(text=f"wait {N}")
        og.sim.step()

def cook_A(A, robot):
    # # Set the orientation of robot so it towards to the target
    # robot_rotate(robot, A)

    A.states[object_states.Cooked].set_value(True)

    # Settle the state
    for _ in range(20):
        video_recorder.get_video(text=f"cook {A.name}")
        og.sim.step()
            
def uncover_A_with_B(A, B, robot):
    # A = {"stain", "dust", "water", ...}
    # # Set the orientation of robot so it towards to the target
    # robot_rotate(robot, A, B)
    # move_arm_to_target(get_operation_position(A), robot)
    # for _ in range(50):
    #     video_recorder.get_video(text=f"uncover {A.name} with {B.name}")
    #     og.sim.step()

    A.states[object_states.Covered].set_value(B, False)

    # Settle the state
    for _ in range(20):
        A.keep_still()
        video_recorder.get_video(text=f"uncover {A.name} with {B.name}")
        og.sim.step()
            
def cover_A_with_B(A, B, robot):
    # A = {"stain", "dust", "water", ...}
    # Set the orientation of robot so it towards to the target
    robot_rotate(robot, A, B)
    camera_rotate(robot, A.get_position() + np.array([0, 0, A.native_bbox[2] * 0.5]))
    if SHOWING_ARM_MOVING:
        arm_pos = A.get_position()
        arm_pos[2] += 0.5 * A.native_bbox[2] + 0.1
        move_arm_to_target(arm_pos, robot)
        for _ in range(150):
            video_recorder.get_video(text=f"cover {A.name} with {B.name}")
            og.sim.step()
    else:
        for _ in range(10):
            video_recorder.get_video(text=f"cover {A.name} with {B.name}")
            og.sim.step()
    
    A.states[object_states.Covered].set_value(B, True)

    # Settle the state
    for _ in range(20):
        A.keep_still()
        video_recorder.get_video(text=f"cover {A.name} with {B.name}")
        og.sim.step()
            
def uncontain_A_with_B(A, B, robot):
    # B = {"water", ...}
    # Set the orientation of robot so it towards to the target
    robot_rotate(robot, A, B)
    camera_rotate(robot, A, B)
    if SHOWING_ARM_MOVING:
        move_arm_to_target(get_operation_position(A), robot)
        for _ in range(50):
            video_recorder.get_video(text=f"uncontain {A.name} with {B.name}")
            og.sim.step()
    else:
        for _ in range(10):
            video_recorder.get_video(text=f"uncontain {A.name} with {B.name}")
            og.sim.step()

    A.states[object_states.Contains].set_value(B, False)

    # Settle the state
    for _ in range(20):
        A.keep_still()
        video_recorder.get_video(text=f"uncontain {A.name} with {B.name}")
        og.sim.step()

def add_A_to_B_with_C(A, B, C, robot):
    robot_rotate(robot, B, C)
    # Lift the arm
    if SHOWING_ARM_MOVING:
        move_arm_to_target(B.get_position() +  np.array([0.0, 0.0, B.native_bbox[2] * 0.5 + C.native_bbox[2] * 0.5 + 0.1]), robot)
        for _ in range(200):
            C.set_position(robot.get_eef_position())
            C.keep_still()
            video_recorder.get_video(text=f"add {A.name} to {B.name}")
            og.sim.step()
    else:
        for _ in range(10):
            C.set_position(robot.get_eef_position())
            C.keep_still()
            video_recorder.get_video(text=f"add {A.name} to {B.name}")
            og.sim.step()

    camera_rotate(robot, B.get_position() + np.array([0, 0, B.native_bbox[2] * 0.5]))    
    # flip
    C.set_position_orientation(
        position=B.get_position() + np.array([0.0, 0.0, B.native_bbox[2] * 0.5 + C.native_bbox[2] * 0.5 + 0.1]),
        orientation=T.euler2quat([np.pi, 0, 0]),
    )
    B.states[object_states.Filled].set_value(A, True)
    for _ in range(25):
        B.keep_still()
        C.keep_still()
        video_recorder.get_video(text=f"add {A.name} to {B.name}")
        og.sim.step()
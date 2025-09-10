#!/usr/bin/env python3

import argparse
import os
import signal
import time
from functools import partial

# ALOHA imports
from aloha.constants import (
    DT,  # Changed from DT_DURATION to DT
    FOLLOWER_GRIPPER_JOINT_OPEN, 
    FOLLOWER_GRIPPER_JOINT_CLOSE,
    FOLLOWER_GRIPPER_JOINT_NORMALIZE_FN,
    IS_MOBILE,
    LEADER2FOLLOWER_JOINT_FN,
    LEADER_GRIPPER_CLOSE_THRESH,
    LEADER_GRIPPER_JOINT_MID,
    START_ARM_POSE,
    FOLLOWER_GRIPPER_JOINT_UNNORMALIZE_FN, 
    FOLLOWER_GRIPPER_POSITION_NORMALIZE_FN
)

from constants import (
    PUPPET_GRIPPER_POSITION_NORMALIZE_FN, 
    PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN, 
    PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN, 
    MASTER_GRIPPER_JOINT_NORMALIZE_FN
)

from aloha.robot_utils import (
    enable_gravity_compensation,
    disable_gravity_compensation,
    get_arm_gripper_positions,
    ImageRecorder,
    move_arms,
    move_grippers,
    torque_off,
    torque_on,
)

# Robot control imports
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
import numpy as np
import rclpy

# OpenPI policy client
import cv2
from openpi_client import websocket_client_policy


def opening_ceremony(
    follower_bot_left: InterbotixManipulatorXS,
    follower_bot_right: InterbotixManipulatorXS,
) -> None:
    """Move follower arms to a pose where it is easy to start demonstration."""
    # reboot gripper motors, and set operating modes for all motors
    follower_bot_left.core.robot_reboot_motors('single', 'gripper', True)
    follower_bot_left.core.robot_set_operating_modes('group', 'arm', 'position')
    follower_bot_left.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
    follower_bot_left.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 300)

    follower_bot_right.core.robot_reboot_motors('single', 'gripper', True)
    follower_bot_right.core.robot_set_operating_modes('group', 'arm', 'position')
    follower_bot_right.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
    follower_bot_right.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 300)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms(
        [follower_bot_left, follower_bot_right],
        [start_arm_qpos] * 2,
        moving_time=4.0,
    )
    # move grippers to starting position

    move_grippers(
        [follower_bot_left, follower_bot_right],
        [FOLLOWER_GRIPPER_JOINT_OPEN] * 2,
        moving_time=0.5
    )

    move_grippers(
        [follower_bot_left, follower_bot_right],
        [FOLLOWER_GRIPPER_JOINT_CLOSE] * 2,
        moving_time=0.5
    )


def get_robot_state(
    follower_bot_left: InterbotixManipulatorXS,
    follower_bot_right: InterbotixManipulatorXS
):
    """Get current robot state (joint positions)."""
    state = np.zeros(14)  # 6 joint + 1 gripper, for two arms
    # Arm positions
    state[:6] = follower_bot_left.core.joint_states.position[:6]
    state[7:7+6] = follower_bot_right.core.joint_states.position[:6]
    # Gripper positions - use FOLLOWER gripper normalization function
    state[6] = FOLLOWER_GRIPPER_JOINT_NORMALIZE_FN(follower_bot_left.core.joint_states.position[6])
    state[7+6] = FOLLOWER_GRIPPER_JOINT_NORMALIZE_FN(follower_bot_right.core.joint_states.position[6])
    return state


def apply_action(
    follower_bot_left: InterbotixManipulatorXS,
    follower_bot_right: InterbotixManipulatorXS,
    action: np.ndarray,
    gripper_command: JointSingleCommand, 
    frequency: float = 10.0
):
    """Apply action to the robots."""

    for i in range(np.shape(action)[0]):

    # Split action into left and right arm components
        left_arm_action = action[i][:6]
        left_gripper_action = action[i][6]
        right_arm_action = action[i][7:13]
        right_gripper_action = action[i][13]

        # print(f"Left arm action: {left_arm_action}")
        # print(f"Left gripper action: {left_gripper_action}")
        # print(f"Right arm action: {right_arm_action}")
        # print(f"Right gripper action: {right_gripper_action}")

        # Unnormalize gripper actions from SmolVLA's normalized output
        # to actual gripper joint values that the robot expects
        left_gripper_unnormalized = FOLLOWER_GRIPPER_POSITION_NORMALIZE_FN(left_gripper_action)
        right_gripper_unnormalized = FOLLOWER_GRIPPER_POSITION_NORMALIZE_FN(right_gripper_action)

        # print(f"Left gripper: {left_gripper_action}, Left gripper unnormalized: {left_gripper_unnormalized}")
        # print(f"Right gripper: {right_gripper_action}, Right gripper unnormalized: {right_gripper_unnormalized}")
        # Apply arm actions (position control)

        follower_bot_left.arm.set_joint_positions(left_arm_action, blocking=False)
        follower_bot_right.arm.set_joint_positions(right_arm_action, blocking=False)
        
        gripper_command.cmd = left_gripper_unnormalized
        follower_bot_left.gripper.core.pub_single.publish(gripper_command)

        gripper_command.cmd = right_gripper_unnormalized
        follower_bot_right.gripper.core.pub_single.publish(gripper_command)

        time.sleep(1.0 / frequency)



def prepare_openpi_observation(images: dict, robot_state: np.ndarray, task_description: str):
    """Prepare observation for OpenPI policy server."""
    observation = {}
    
    # Add robot state (14 DOF for ALOHA dual-arm setup)
    observation["state"] = robot_state.astype(np.float32)
    
    # Process camera images for OpenPI format
    # Expected format: (channels, height, width) with values in [0, 255] uint8
    observation["images"] = {}
    
    for cam_name, image in images.items():
        if image is not None:
            # Convert BGR to RGB 
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Resize to expected dimensions (224x224 for OpenPI)
                image_resized = cv2.resize(image_rgb, (224, 224))
                # Convert to channels-first format (C, H, W)
                image_chw = np.transpose(image_resized, (2, 0, 1))
                
                # Map camera names to OpenPI expected format
                cam_lower = cam_name.lower()
                if 'high' in cam_lower or 'top' in cam_lower or 'cam_high' in cam_lower:
                    observation["images"]["cam_high"] = image_chw.astype(np.uint8)
                elif 'left' in cam_lower and ('wrist' in cam_lower or 'hand' in cam_lower):
                    observation["images"]["cam_left_wrist"] = image_chw.astype(np.uint8)
                elif 'right' in cam_lower and ('wrist' in cam_lower or 'hand' in cam_lower):
                    observation["images"]["cam_right_wrist"] = image_chw.astype(np.uint8)
                else:
                    # If we can't match the name, use it as the first available camera
                    if "cam_high" not in observation["images"]:
                        observation["images"]["cam_high"] = image_chw.astype(np.uint8)
                    elif "cam_left_wrist" not in observation["images"]:
                        observation["images"]["cam_left_wrist"] = image_chw.astype(np.uint8)
                    elif "cam_right_wrist" not in observation["images"]:
                        observation["images"]["cam_right_wrist"] = image_chw.astype(np.uint8)
    
    # Ensure we have all required cameras (fill missing ones with zeros)
    required_cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    for cam in required_cameras:
        if cam not in observation["images"]:
            observation["images"][cam] = np.zeros((3, 224, 224), dtype=np.uint8)
    
    # Add task description (prompt)
    observation["prompt"] = task_description
    
    return observation


def main(args: dict) -> None:
    host = args.get('host', 'localhost')
    port = args.get('port', 8000)
    api_key = args.get('api_key', None)
    task_description = args.get('task', "Put the sponge in the pot")
    save_images = args.get('save_images', False)
    save_dir = args.get('save_dir', '/tmp/robot_controller_images')
    control_frequency = args.get('control_frequency', 10.0)  # Hz
    

    print(f"🔄 Control frequency: {control_frequency} Hz")

    # Connect to OpenPI policy server
    print(f"Connecting to OpenPI policy server at {host}:{port}")
    try:
        policy = websocket_client_policy.WebsocketClientPolicy(
            host=host,
            port=port,
            api_key=api_key,
        )
        
        metadata = policy.get_server_metadata()
        print(f"✅ Connected to OpenPI policy server!")
        print(f"   Server metadata: {metadata}")
        
    except Exception as e:
        print(f"❌ Error connecting to policy server: {e}")
        print("   Make sure the policy server is running with: python scripts/serve_policy.py")
        return

    # Initialize ROS node
    node = create_interbotix_global_node('aloha')
    
    # Initialize follower bots
    follower_bot_left = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='follower_left',
        node=node,
        iterative_update_fk=False,
    )
    follower_bot_right = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='follower_right',
        node=node,
        iterative_update_fk=False,
    )
    
    # Initialize camera recorder
    print("Initializing cameras...")
    image_recorder = ImageRecorder(node=node, is_mobile=IS_MOBILE)
    
    # Initialize gripper command for ROS publishing
    gripper_command = JointSingleCommand()
    gripper_command.name = "gripper"  # Specify which joint we're commanding
    
    # Give cameras time to initialize
    time.sleep(1.0)

    # Initialize robots
    robot_startup(node)
    
    # Set up follower bots for control
    print("Setting up robots...")

    opening_ceremony(follower_bot_left, follower_bot_right)
    
    print("🤖 Robot control with OpenPI started!")
    print(f"🎯 Task: {task_description}")
    print("📷 Reading camera data and controlling robots with OpenPI policy")
    
    
    # Signal handler for clean shutdown
    def signal_handler(signum, frame):
        print("\nShutting down robots...")
        robot_shutdown(node)
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    # Main control loop
    loop_count = 0
    loop_start_time = time.time()
    try:
        while rclpy.ok():
            
            # Get current robot state
            robot_state = get_robot_state(follower_bot_left, follower_bot_right)
            
            # Get camera images
            images = image_recorder.get_images()
            
            # Debug camera data before processing
            if loop_count % 20 == 0:  # Print debug info every 20 iterations
                elapsed_time = time.time() - loop_start_time
                actual_freq = loop_count / elapsed_time if elapsed_time > 0 else 0
                print(f"\n🔍 Debug Info - Step {loop_count} | Actual freq: {actual_freq:.1f} Hz")
                print(f"ImageRecorder camera_names: {image_recorder.camera_names}")
                print(f"Images dict keys: {list(images.keys())}")
                for cam_name, img in images.items():
                    if img is not None:
                        print(f"  {cam_name}: {img.shape} {img.dtype}")
                    else:
                        print(f"  {cam_name}: None (no data)")
            
            # Prepare observation for OpenPI policy server
            try:
                observation = prepare_openpi_observation(images, robot_state, task_description)
                # Get action prediction from OpenPI policy server
                result = policy.infer(observation)
                action = result["actions"]

                # print(f"Action shape: {action.shape}")
                # print(f"Action: {action}")
                
                # Handle action format - take first action from action chunk
                # if len(action.shape) == 2:  # If we get an action chunk [horizon, action_dim]
                #     action = action[0]  # Take the first action
                
                # Apply the predicted action to the robots
                apply_action(follower_bot_left, follower_bot_right, action, gripper_command, control_frequency)

                # print action in readable format, action shap is (50, 14)
                print(f"Action shape: {action.shape}")
                print(f"Action: {action}")

                loop_count += 1

                time.sleep(0.2)
                

                
            except Exception as e:
                print(f"❌ Error in control loop: {e}")
                # Continue loop but don't apply action
            

            
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    finally:
        print("Disabling robot torque...")
        # torque_off(follower_bot_left)
        # torque_off(follower_bot_right)
        robot_shutdown(node)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenPI Robot Controller - Control ALOHA robots using OpenPI policy server')
    parser.add_argument(
        '--host',
        type=str,
        # default='10.6.231.114',
        default='0.0.0.0',
        help='OpenPI policy server host (default: localhost)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='OpenPI policy server port (default: 8000)'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default=None,
        help='API key for the OpenPI policy server (optional)'
    )
    parser.add_argument(
        '--task',
        type=str,
        default="double-fold the shorts",
        help='Task description for the robot (default: "clean dish")'
    )
    parser.add_argument(
        '--save_images',
        action='store_true',
        help='Save camera images to disk periodically'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='/tmp/robot_controller_images',
        help='Directory to save camera images (default: /tmp/robot_controller_images)'
    )
    parser.add_argument(
        '--control_frequency',
        type=float,
        default=50.0,
        help='Desired control frequency in Hz (default: 10.0)'
    )
    
    args = vars(parser.parse_args())
    main(args)

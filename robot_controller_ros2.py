#!/usr/bin/env python3
"""
ROS 2 Robot Controller for OpenPI (Python 3.10 compatible)

This version separates robot control (ROS 2, Python 3.10) from 
model inference (OpenPI server, Python 3.11) using websocket communication.
"""

import argparse
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import dataclasses
from pathlib import Path
import json

import cv2
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, JointState, CompressedImage
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
import einops

# Lightweight OpenPI client imports (no heavy model dependencies)
try:
    import websocket
    import msgpack
    import msgpack_numpy
    HAS_WEBSOCKET = True
except ImportError:
    print("Warning: websocket/msgpack not available. Install with: pip install websocket-client msgpack msgpack-numpy")
    HAS_WEBSOCKET = False


@dataclasses.dataclass
class RobotConfig:
    """Configuration for robot controller."""
    
    # Server connection
    host: str = "localhost"
    port: int = 8000
    
    # Control parameters
    max_publish_steps: int = 10000
    publish_rate: int = 25
    control_frequency: int = 25
    action_horizon: int = 64
    
    # Robot topics
    img_front_topic: str = '/cam_high/camera/color/image_rect_raw/compressed'
    img_left_topic: str = '/cam_left_wrist/camera/color/image_rect_raw/compressed'
    img_right_topic: str = '/cam_right_wrist/camera/color/image_rect_raw/compressed'
    
    puppet_arm_left_topic: str = '/follower_left/joint_states'
    puppet_arm_right_topic: str = '/follower_right/joint_states'
    puppet_arm_left_cmd_topic: str = '/follower_left/commands/joint_group'
    puppet_arm_right_cmd_topic: str = '/follower_right/commands/joint_group'
    puppet_arm_left_gripper_cmd_topic: str = '/follower_left/commands/joint_single'
    puppet_arm_right_gripper_cmd_topic: str = '/follower_right/commands/joint_single'
    
    robot_base_topic: str = '/odom_raw'
    robot_base_cmd_topic: str = '/cmd_vel'
    
    # Sync thresholds
    rgb_sync_threshold: float = 0.1
    
    # Safety settings
    disable_puppet_arm: bool = False
    use_robot_base: bool = False
    
    # Task settings
    task_prompt: str = "pick up the red block"
    
    # Image processing
    render_height: int = 224
    render_width: int = 224
    
    # Arm movement limits
    arm_steps_length: List[float] = dataclasses.field(
        default_factory=lambda: [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    )
    
    # Interpolation settings
    use_actions_interpolation: bool = True
    
    # Home positions
    home_left: List[float] = dataclasses.field(
        default_factory=lambda: [0.0, -1.822, 1.554, -0.008, -1.569, -0.008, 0.69]
    )
    home_right: List[float] = dataclasses.field(
        default_factory=lambda: [0.0, -1.822, 1.554, -0.008, -1.569, -0.008, 0.69]
    )
    
    # Gripper mapping
    gripper_input_min: float = 0.0
    gripper_input_max: float = 1.0
    gripper_output_min: float = 0.6197
    gripper_output_max: float = 1.6214


class SimplePolicyClient:
    """Lightweight OpenPI policy client for Python 3.10."""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.ws = None
        self.connected = False
        
        if not HAS_WEBSOCKET:
            raise ImportError("websocket-client and msgpack packages required")
        
        self._connect()
    
    def _connect(self):
        """Connect to OpenPI policy server."""
        try:
            import websocket
            self.ws = websocket.create_connection(f"ws://{self.host}:{self.port}")
            
            # Receive server metadata
            metadata_data = self.ws.recv()
            metadata = msgpack.unpackb(metadata_data, raw=False)
            print(f"Connected to OpenPI server. Metadata: {metadata}")
            
            self.connected = True
            
        except Exception as e:
            print(f"Failed to connect to OpenPI server: {e}")
            raise
    
    def infer(self, observation: Dict) -> Dict:
        """Get action from OpenPI policy server."""
        if not self.connected:
            raise RuntimeError("Not connected to policy server")
        
        try:
            # Send observation
            obs_data = msgpack.packb(observation, use_bin_type=True)
            self.ws.send(obs_data, websocket.ABNF.OPCODE_BINARY)
            
            # Receive response
            response_data = self.ws.recv()
            if isinstance(response_data, str):
                raise RuntimeError(f"Server error: {response_data}")
            
            response = msgpack.unpackb(response_data, raw=False)
            return response
            
        except Exception as e:
            print(f"Error during inference: {e}")
            raise
    
    def close(self):
        """Close connection."""
        if self.ws:
            self.ws.close()
            self.connected = False


class LogManager:
    """Centralized logging manager for robot operations."""
    
    def __init__(self, log_dir: str = "logs"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, self.timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup different loggers
        self.system_logger = self._setup_logger("system", "system.log")
        self.action_logger = self._setup_logger("action", "actions.log")
        self.error_logger = self._setup_logger("error", "errors.log")
        
    def _setup_logger(self, name: str, filename: str) -> logging.Logger:
        """Setup a logger with file and console handlers."""
        logger = logging.getLogger(f"robot_controller.{name}")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        
        # File handler
        fh = logging.FileHandler(os.path.join(self.log_dir, filename))
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def log_system(self, message: str):
        self.system_logger.info(message)
    
    def log_action(self, action_type: str, data: Any):
        self.action_logger.info(f"{action_type}: {data}")
    
    def log_error(self, error_msg: str, exception: Optional[Exception] = None):
        if exception:
            self.error_logger.error(f"{error_msg}: {str(exception)}", exc_info=True)
        else:
            self.error_logger.error(error_msg)


class SensorDataManager:
    """Manages synchronized sensor data collection."""
    
    def __init__(self, config: RobotConfig, logger: LogManager):
        self.config = config
        self.logger = logger
        self.bridge = CvBridge()
        
        # Data storage
        self.synced_data = deque(maxlen=10)
        
    def compressed_imgmsg_to_cv2(self, compressed_msg: CompressedImage) -> np.ndarray:
        """Convert compressed image message to CV2 format."""
        try:
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.logger.log_error("Error converting compressed image", e)
            raise
    
    def process_rgb_arm_data(self, msg_front, msg_left, msg_right, msg_left_arm, msg_right_arm):
        """Process synchronized RGB and arm data."""
        try:
            # Convert images
            cv_front = self.compressed_imgmsg_to_cv2(msg_front)
            cv_left = self.compressed_imgmsg_to_cv2(msg_left)
            cv_right = self.compressed_imgmsg_to_cv2(msg_right)
            
            # Store synchronized data
            synced_data = {
                'timestamp': msg_front.header.stamp,
                'rgb_front': cv_front,
                'rgb_left': cv_left,
                'rgb_right': cv_right,
                'left_arm': msg_left_arm,
                'right_arm': msg_right_arm,
            }
            
            self.synced_data.append(synced_data)
            
        except Exception as e:
            self.logger.log_error("Error in RGB-ARM sync callback", e)
    
    def get_latest_observation(self) -> Optional[Dict]:
        """Get the latest synchronized observation."""
        if len(self.synced_data) == 0:
            return None
            
        latest_data = self.synced_data[-1]
        
        # Process joint positions
        left_pos = np.array(latest_data['left_arm'].position[:6])
        right_pos = np.array(latest_data['right_arm'].position[:6])
        
        # Add gripper positions (7th element)
        left_gripper = latest_data['left_arm'].position[6] if len(latest_data['left_arm'].position) > 6 else 0.7
        right_gripper = latest_data['right_arm'].position[6] if len(latest_data['right_arm'].position) > 6 else 0.7
        
        left_pos = np.append(left_pos, left_gripper)
        right_pos = np.append(right_pos, right_gripper)
        
        # Concatenate to get 14 values total
        qpos = np.concatenate((left_pos, right_pos), axis=0)
        
        # Process images for OpenPI (convert to channels-first format)
        images = {}
        for cam_name, img in [
            ('cam_high', latest_data['rgb_front']),
            ('cam_left_wrist', latest_data['rgb_left']),
            ('cam_right_wrist', latest_data['rgb_right'])
        ]:
            # Resize to expected size
            img_resized = cv2.resize(img, (self.config.render_width, self.config.render_height))
            # Convert to channels-first format (C, H, W)
            img_chw = np.transpose(img_resized, (2, 0, 1))
            images[cam_name] = img_chw.astype(np.uint8)
        
        return {
            'state': qpos.astype(np.float32),
            'images': images,
            'prompt': self.config.task_prompt
        }


class RobotActuator:
    """Handles robot actuation and movement."""
    
    def __init__(self, config: RobotConfig, logger: LogManager, node: Node):
        self.config = config
        self.logger = logger
        self.node = node
        
        # Initialize publishers
        self._init_publishers()
        
        # Movement tracking
        self.last_action = np.zeros(14)
        
    def _init_publishers(self):
        """Initialize ROS publishers."""
        self.left_arm_publisher = self.node.create_publisher(
            JointGroupCommand, self.config.puppet_arm_left_cmd_topic, 10
        )
        self.right_arm_publisher = self.node.create_publisher(
            JointGroupCommand, self.config.puppet_arm_right_cmd_topic, 10
        )
        self.left_gripper_publisher = self.node.create_publisher(
            JointSingleCommand, self.config.puppet_arm_left_gripper_cmd_topic, 10
        )
        self.right_gripper_publisher = self.node.create_publisher(
            JointSingleCommand, self.config.puppet_arm_right_gripper_cmd_topic, 10
        )
    
    def execute_action(self, action: np.ndarray):
        """Execute a single action on the robot."""
        try:
            if self.config.disable_puppet_arm:
                self.logger.log_action("DISABLED", action.tolist())
                return
            
            # Split action into left and right arms
            left_action = action[:7]
            right_action = action[7:14]
            
            # Extract joint positions and gripper values
            left_joints = left_action[:6]
            left_gripper = left_action[6]
            right_joints = right_action[:6]
            right_gripper = right_action[6]
            
            # Map gripper values
            left_gripper_mapped = self._map_gripper_range(abs(left_gripper))
            right_gripper_mapped = self._map_gripper_range(abs(right_gripper))
            
            # Publish arm commands
            left_msg = JointGroupCommand()
            left_msg.name = "arm"
            left_msg.cmd = left_joints.tolist()
            self.left_arm_publisher.publish(left_msg)
            
            right_msg = JointGroupCommand()
            right_msg.name = "arm"
            right_msg.cmd = right_joints.tolist()
            self.right_arm_publisher.publish(right_msg)
            
            # Publish gripper commands
            left_gripper_msg = JointSingleCommand()
            left_gripper_msg.name = "gripper"
            left_gripper_msg.cmd = float(left_gripper_mapped)
            self.left_gripper_publisher.publish(left_gripper_msg)
            
            right_gripper_msg = JointSingleCommand()
            right_gripper_msg.name = "gripper"
            right_gripper_msg.cmd = float(right_gripper_mapped)
            self.right_gripper_publisher.publish(right_gripper_msg)
            
            # Update last action
            self.last_action = action.copy()
            
        except Exception as e:
            self.logger.log_error("Error executing action", e)
            raise
    
    def _map_gripper_range(self, x: float) -> float:
        """Map gripper value from one range to another."""
        return (x - self.config.gripper_input_min) * (self.config.gripper_output_max - self.config.gripper_output_min) / (self.config.gripper_input_max - self.config.gripper_input_min) + self.config.gripper_output_min


class RobotController(Node):
    """Main robot controller class."""
    
    def __init__(self, config: RobotConfig):
        super().__init__('openpi_robot_controller')
        
        self.config = config
        self.logger = LogManager()
        
        # Initialize components
        self.sensor_manager = SensorDataManager(config, self.logger)
        self.actuator = RobotActuator(config, self.logger, self)
        
        # Initialize policy client
        try:
            self.policy_client = SimplePolicyClient(config.host, config.port)
            self.logger.log_system("Connected to OpenPI policy server")
        except Exception as e:
            self.logger.log_error("Failed to connect to policy server", e)
            raise
        
        # Control state
        self.running = False
        self.step_count = 0
        
        # Initialize ROS components
        self._init_subscribers()
        
        # Start ROS spin thread
        self.spin_thread = threading.Thread(target=self._spin_thread)
        self.spin_thread.daemon = True
        self.spin_thread.start()
        
        self.logger.log_system("Robot controller initialized successfully")
    
    def _init_subscribers(self):
        """Initialize ROS subscribers."""
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=30
        )
        
        # Create synchronized subscribers
        subs_rgb_arm = [
            message_filters.Subscriber(self, CompressedImage, self.config.img_front_topic, qos_profile=sensor_qos),
            message_filters.Subscriber(self, CompressedImage, self.config.img_left_topic, qos_profile=sensor_qos),
            message_filters.Subscriber(self, CompressedImage, self.config.img_right_topic, qos_profile=sensor_qos),
            message_filters.Subscriber(self, JointState, self.config.puppet_arm_left_topic, qos_profile=sensor_qos),
            message_filters.Subscriber(self, JointState, self.config.puppet_arm_right_topic, qos_profile=sensor_qos)
        ]
        
        # Create synchronizer
        self.synchronizer = ApproximateTimeSynchronizer(
            subs_rgb_arm,
            queue_size=30,
            slop=self.config.rgb_sync_threshold
        )
        self.synchronizer.registerCallback(self.sensor_manager.process_rgb_arm_data)
    
    def _spin_thread(self):
        """ROS spin thread."""
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
    
    def run_control_loop(self):
        """Main control loop."""
        try:
            self.logger.log_system("Starting robot control loop")
            
            # Wait for initial data
            self.logger.log_system("Waiting for sensor data...")
            while len(self.sensor_manager.synced_data) == 0:
                time.sleep(0.1)
            
            # Create rate limiter
            rate = self.create_rate(self.config.publish_rate)
            
            self.running = True
            self.step_count = 0
            
            while self.running and rclpy.ok() and self.step_count < self.config.max_publish_steps:
                try:
                    # Get latest observation
                    observation = self.sensor_manager.get_latest_observation()
                    if observation is None:
                        rate.sleep()
                        continue
                    
                    # Get action from policy
                    result = self.policy_client.infer(observation)
                    action = result["actions"]
                    
                    # Handle action format
                    if len(action.shape) == 2:
                        action = action[0]  # Take first action from chunk
                    
                    # Execute action
                    self.actuator.execute_action(action)
                    
                    self.step_count += 1
                    
                    if self.step_count % 50 == 0:
                        self.logger.log_system(f"Completed {self.step_count} control steps")
                    
                    rate.sleep()
                    
                except KeyboardInterrupt:
                    self.logger.log_system("Keyboard interrupt received")
                    break
                except Exception as e:
                    self.logger.log_error("Error in control loop", e)
                    time.sleep(0.1)
            
            self.logger.log_system(f"Control loop completed after {self.step_count} steps")
            
        except Exception as e:
            self.logger.log_error("Fatal error in control loop", e)
            raise
        finally:
            self.running = False
            if hasattr(self, 'policy_client'):
                self.policy_client.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ROS 2 Robot Controller for OpenPI")
    parser.add_argument('--host', type=str, default='localhost', help='OpenPI server host')
    parser.add_argument('--port', type=int, default=8000, help='OpenPI server port')
    parser.add_argument('--task_prompt', type=str, default='pick up the red block', help='Task description')
    parser.add_argument('--disable_puppet_arm', action='store_true', help='Disable arm movement for testing')
    parser.add_argument('--publish_rate', type=int, default=25, help='Control loop frequency (Hz)')
    
    args = parser.parse_args()
    
    # Initialize ROS
    rclpy.init()
    
    try:
        # Create config
        config = RobotConfig(
            host=args.host,
            port=args.port,
            task_prompt=args.task_prompt,
            disable_puppet_arm=args.disable_puppet_arm,
            publish_rate=args.publish_rate
        )
        
        # Create and run robot controller
        controller = RobotController(config)
        controller.run_control_loop()
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        raise
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

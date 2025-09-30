#!/usr/bin/env python3
import rclpy
import requests
import base64
import cv2
import numpy as np
import time
import csv
import os
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge
from std_srvs.srv import SetBool
from multi_mode_control_msgs.msg import CartesianImpedanceGoal
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
from vla_interfaces.srv import SetPrompt
from vla_interfaces.srv import SetRequest
from rclpy.action import ActionClient
from franka_msgs.action import Move
from std_srvs.srv import Trigger



class VLABridgeNode(Node):
    def __init__(self):
        super().__init__('vla_bridge_node')
        self.get_logger().info('vla_bridge_node has been started.')

        # Topic parameterisation
        self.declare_parameter('input_image_topic', '/camera/realsense2_camera/color/image_raw')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('output_pose_topic', '/panda/panda_cartesian_impedance_controller/desired_pose')
        self.declare_parameter('gripper_action', '/panda_gripper/move')
        self.declare_parameter('gripper_stop', '/panda_gripper/stop')
        self.declare_parameter('tf_base_frame', 'panda_link0')
        self.declare_parameter('tf_ee_frame', 'panda_hand_tcp')
        self.declare_parameter('record_timings', False)
        self.declare_parameter('timings_csv_path', '/tmp/vla_timings.csv')

        self.input_image_topic  = self.get_parameter('input_image_topic').value
        self.joint_states_topic = self.get_parameter('joint_states_topic').value
        self.output_pose_topic  = self.get_parameter('output_pose_topic').value
        self.gripper_action     = self.get_parameter('gripper_action').value
        self.gripper_stop       = self.get_parameter('gripper_stop').value
        self.tf_base_frame      = self.get_parameter('tf_base_frame').value
        self.tf_ee_frame        = self.get_parameter('tf_ee_frame').value
        self.record_timings     = bool(self.get_parameter('record_timings').value)
        self.timings_csv_path   = str(self.get_parameter('timings_csv_path').value)

        # Defaults
        self.prompt             = "go up"
        self.backend_url        = "http://localhost:8000/predict"
        self.request_interval   = 0.2  # seconds, also impacts scaling of deltas
        self.model              = "openvla/openvla-7b"
        self.image_width        = 224
        self.image_height       = 224
        self.grip_speed         = 0.1
        self.active             = False  # Initially stopped

        # Dummys
        self.latest_image = self.generate_dummy_image()
        self.joint_states = {}
        self.latest_joint_angles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # TODO migrate arm logic to joint_states
        self.cv_bridge = CvBridge()

        # Services
        self.create_service(SetBool, '/toggle_active', self.handle_toggle)
        self.create_service(SetPrompt, '/set_prompt', self.set_prompt_callback)
        self.create_service(SetRequest, '/set_request', self.set_request_callback)

        # Subscribers
        self.create_subscription(Image, self.input_image_topic, self.image_callback, 10)
        self.create_subscription(JointState, self.joint_states_topic, self.joint_state_callback, 10)

        # Publishers
        self.pose_pub = self.create_publisher(CartesianImpedanceGoal, self.output_pose_topic, 10)

        # Actions
        self.gripper_move_ac = ActionClient(self, Move, self.gripper_action)

        # Client
        self.gripper_stop = self.create_client(Trigger, self.gripper_stop)

        # Timer for periodic requests
        self.timer = self.create_timer(self.request_interval, self.send_request)

        # TF listener setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Create csv file if record_timings is set
        self._ensure_csv()

    def _ensure_csv(self):
        if not self.record_timings:
            return
        exists = os.path.exists(self.timings_csv_path)
        if not exists:
            with open(self.timings_csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(["t_client_start", "t_publish", "prompt"])
    
    def generate_dummy_image(self):
        # dummy image and encode once (size is arbitrary here)
        dummy_image = np.zeros((240, 320, 3), dtype=np.uint8)
        return self.encode_png(dummy_image)
    
    def encode_png(self, img_bgr):
        # Encode original BGR image as PNG → base64
        try:
            ok, buffer = cv2.imencode('.png', img_bgr)
            if not ok:
                raise RuntimeError("cv2.imencode('.png') failed")
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            self.get_logger().error(f"Preprocess/encode failed: {e}")
            return None

    def image_callback(self, msg):
        try:
            cv_image_bgr = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = self.encode_png(cv_image_bgr)
            if self.latest_image is None:
                self.get_logger().warn("Latest image not set due to encode failure.")
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def set_prompt_callback(self, request, response):
        self.prompt = request.prompt
        response.success = True
        response.message = f"Prompt updated to: {self.prompt}"
        self.get_logger().info(response.message)
        return response
    
    def joint_state_callback(self, msg):
        self.latest_joint_angles = list(msg.position) # TODO migrate arm logic to joint_states
        self.joint_states = {n: p for n, p in zip(msg.name, msg.position or [])}

    def handle_toggle(self, request, response):
        self.active = request.data  # True to start, False to stop
        state_str = "started" if self.active else "stopped"
        self.get_logger().info(f"VLA Requester has been {state_str} by service call.")
        response.success = True
        response.message = f"Requester {state_str}."
        return response
    
    def set_request_callback(self, request, response):
        # Update configuration fields
        self.prompt = request.prompt
        self.backend_url = request.backend_url
        self.request_interval = request.request_interval
        self.active = request.active
        self.model = request.model

        # Reset timer with new interval
        try:
            self.timer.cancel()
            self.timer = self.create_timer(self.request_interval, self.send_request)
        except Exception as e:
            self.get_logger().error(f"Failed to reset timer: {e}")
            response.success = False
            response.message = f"Update failed: {e}"
            return response

        response.success = True
        response.message = (
            f"Updated request: model={self.model}, "
            f"url={self.backend_url}, "
            f"interval={self.request_interval}, "
            f"prompt='{self.prompt}', active={self.active}"
        )
        self.get_logger().info(response.message)
        return response

    def send_request(self):
        if not self.active:
            return  # Skip if inactive
        
        if self.record_timings:
            t_client_start = time.perf_counter_ns()
    
        if self.latest_image is None or not self.latest_joint_angles:
            self.get_logger().warn("Waiting for image and joint states...")
            return

        payload = {
            "prompt": self.prompt,
            "joint_angles": self.latest_joint_angles,
            "image": self.latest_image,
            "model": self.model
        }

        try:
            response = requests.post(self.backend_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                result = self.saturate_delta(result, 0.1, 0.1)
                self.get_logger().info(
                    f"Request successful"
                    f": x={result.get('delta_x', 0.0):.3f}, "
                    f"y={result.get('delta_y', 0.0):.3f}, "
                    f"z={result.get('delta_z', 0.0):.3f}, "
                    f"g={result.get('delta_gripper', 0.0):.3f}"
                ) # ad {result} when needed

                current_pose = self.get_current_pose()
                if current_pose:
                    absolute_pose = self.compute_absolute_pose(current_pose, result)
                    self.publish_pose(absolute_pose)
                else:
                    self.get_logger().warn("Skipping publish because current pose is unavailable.")
                
                current_width = self.get_current_width()
                if current_width is not None and "delta_gripper" in result:
                    step = self.grip_speed * self.request_interval * 0.5 # choosing a acheevable stepsize
                    absolute_width = self.compute_absolute_width(current_width, result, step)
                    if abs(current_width - absolute_width) > 0.001:
                        self.get_logger().info(f"grip cw={current_width} dg={result.get('delta_gripper')} -> aw={absolute_width}")
                        self.publish_gripper_width(absolute_width)
                    else:
                        self.get_logger().info("width difference < 0.001")
                else:
                    self.get_logger().warn("Skipping gripper publish: current width unavailable or no delta_gripper.")
                if self.record_timings:
                    t_publish = time.perf_counter_ns()
                    with open(self.timings_csv_path, 'a', newline='') as f:
                        csv.writer(f).writerow([t_client_start, t_publish, str(self.prompt)])
            else:
                self.get_logger().warn(f"Backend response: {response.status_code}")
        except Exception as e:
            self.get_logger().error(f"Failed to send request: {e}")

    def publish_gripper_width(self, width):
        # publish a Move goal with target width [m].
        if width is None or not self.active:
            return
        self.gripper_stop.call_async(Trigger.Request())
        goal = Move.Goal()
        goal.width = float(width)
        goal.speed = float(self.grip_speed)
        self.gripper_move_ac.send_goal_async(goal)

    def publish_pose(self, result):
        goal = CartesianImpedanceGoal()
        
        goal.pose.position.x = result["x"]
        goal.pose.position.y = result["y"]
        goal.pose.position.z = result["z"]
        
        goal.pose.orientation.x = result.get("qx", 0.0)
        goal.pose.orientation.y = result.get("qy", 0.0)
        goal.pose.orientation.z = result.get("qz", 0.0)
        goal.pose.orientation.w = result.get("qw", 1.0)
        
        goal.q_n = [0.0] * 7  # Just a placeholder; adjust if needed
        
        self.pose_pub.publish(goal)
        # self.get_logger().info(f"Published desired pose: {goal}")

    def get_current_width(self):
        # Compute current gripper width [m] from latest /joint_states.
        try:
            j1 = self.joint_states.get('panda_finger_joint1')
            j2 = self.joint_states.get('panda_finger_joint2')
            if j1 is None or j2 is None:
                self.get_logger().warn("Finger joints not yet in JointState; width unavailable.")
                return None
            width = float(j1 + j2)
            return max(0.0, min(0.08, width))
        except Exception as e:
            self.get_logger().error(f"get_current_width error: {e}")
            return None

    def get_current_pose(self):
        # Fetch the current pose of tf_ee_frame (end-effector) in tf_base_frame (robot base) frame.
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                self.tf_base_frame,
                self.tf_ee_frame,
                now,
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            pose = PoseStamped()
            pose.header.frame_id = 'tf_ee_frame'
            pose.pose.position.x = trans.transform.translation.x
            pose.pose.position.y = trans.transform.translation.y
            pose.pose.position.z = trans.transform.translation.z
            pose.pose.orientation = trans.transform.rotation

            return pose
        except Exception as e:
            self.get_logger().error(f"Failed to get current pose: {e}")
            return None
    
    @staticmethod
    def compute_absolute_width(current_width, delta, step=0.02, min_w=0.0, max_w=0.08):
        # Map delta_gripper to an absolute target width considering limits.
        dg = float(delta.get("delta_gripper", 0.0))
        base = current_width if current_width is not None else min_w
        target = base + step * dg # adding? deltas
        if target < min_w:
            return min_w
        if target > max_w:
            return max_w
        return target

    @staticmethod
    def compute_absolute_pose(current_pose: PoseStamped, delta: dict):
        # Compute absolute pose from current pose and delta values.
        x = current_pose.pose.position.x + delta.get("delta_x", 0.0)
        y = current_pose.pose.position.y + delta.get("delta_y", 0.0)
        z = current_pose.pose.position.z + delta.get("delta_z", 0.0)

        # Get current orientation as quaternion [x, y, z, w]
        current_q = [
            current_pose.pose.orientation.x,
            current_pose.pose.orientation.y,
            current_pose.pose.orientation.z,
            current_pose.pose.orientation.w
        ]

        # Convert delta roll/pitch/yaw to Rotation object
        delta_roll = delta.get("delta_roll", 0.0)
        delta_pitch = delta.get("delta_pitch", 0.0)
        delta_yaw = delta.get("delta_yaw", 0.0)

        delta_r = R.from_euler('xyz', [delta_roll, delta_pitch, delta_yaw])

        # Compose rotations: target = delta * current
        current_r = R.from_quat(current_q)
        target_r = delta_r * current_r

        # Get quaternion as [x, y, z, w]
        target_q = target_r.as_quat()

        # Return dictionary in expected format
        return {
            "x": x,
            "y": y,
            "z": z,
            "qx": target_q[0],
            "qy": target_q[1],
            "qz": target_q[2],
            "qw": target_q[3]
        }
    
    @staticmethod
    def saturate_delta(delta: dict, max_trans: float = 0.1, max_rot: float = 0.1) -> dict:
        # Scale translation and rotation deltas so they do not exceed limits. max_trans in meters, max_rot in radians.
        dpos = np.array([
            delta.get("delta_x", 0.0),
            delta.get("delta_y", 0.0),
            delta.get("delta_z", 0.0)
        ])
        dist = np.linalg.norm(dpos)
        if dist > max_trans and dist > 0:
            dpos = dpos * (max_trans / dist)
        delta["delta_x"], delta["delta_y"], delta["delta_z"] = dpos.tolist()

        # --- rotation ---
        drot = R.from_euler(
            'xyz',
            [
                delta.get("delta_roll", 0.0),
                delta.get("delta_pitch", 0.0),
                delta.get("delta_yaw", 0.0)
            ]
        )
        angle = drot.magnitude()
        if angle > max_rot and angle > 0:
            axis = drot.as_rotvec() / angle
            drot = R.from_rotvec(axis * max_rot)
            delta["delta_roll"], delta["delta_pitch"], delta["delta_yaw"] = drot.as_euler('xyz').tolist()

        return delta


def main(args=None):
    rclpy.init(args=args)
    node = VLABridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
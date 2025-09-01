#!/usr/bin/env python3
import rclpy
import requests
import base64
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from cv_bridge import CvBridge
from std_srvs.srv import SetBool
from multi_mode_control_msgs.msg import CartesianImpedanceGoal
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
from vla_interfaces.srv import SetPrompt
from vla_interfaces.srv import SetRequest



class VLABridgeNode(Node):
    def __init__(self):
        super().__init__('vla_bridge_node')
        self.get_logger().info('vla_bridge_node has been started.')

        self.create_service(SetBool, '/toggle_active', self.handle_toggle)
        self.create_service(SetPrompt, '/set_prompt', self.set_prompt_callback)
        self.create_service(SetRequest, '/set_request', self.set_request_callback)

        # Configuration
        self.prompt = "go up"
        self.backend_url = "http://localhost:8000/predict"
        self.request_interval = 1.0  # seconds
        self.model = "openvla/openvla-7b"
        self.image_width = 224
        self.image_height = 224
        self.active = False  # Initially stopped

        # Internal state
        self.latest_image = self.generate_dummy_image()
        self.latest_joint_angles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.bridge = CvBridge()

        # Subscribers
        self.create_subscription(Image, '/camera/realsense2_camera/color/image_raw', self.image_callback, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        # Publishers
        self.pose_pub = self.create_publisher(
            CartesianImpedanceGoal,
            '/panda/panda_cartesian_impedance_controller/desired_pose',
            10
        )

        # Timer for periodic requests
        self.timer = self.create_timer(self.request_interval, self.send_request)

        # TF listener setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def generate_dummy_image(self):
        # Create black dummy image and encode
        import numpy as np
        dummy_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        return self.encode_and_resize(dummy_image)
    
    def encode_and_resize(self, img_bgr):
        # BGR to RGB, resize, JPEG to base64
        try:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
            ok, buffer = cv2.imencode('.jpg', resized)
            if not ok:
                raise RuntimeError("cv2.imencode failed")
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            self.get_logger().error(f"Preprocess/encode failed: {e}")
            return None

    def image_callback(self, msg):
        try:
            cv_image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = self.encode_and_resize(cv_image_bgr)
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
        self.latest_joint_angles = list(msg.position)

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
                self.get_logger().info(f"Request successful: {result}")
                result = self.saturate_delta(result, 0.1, 0.1)

                current_pose = self.get_current_pose()
                current_widt = self.get_current_width() # TODO publish width
                if current_pose:
                    absolute_pose = self.compute_absolute_pose(current_pose, result)
                    self.publish_pose(absolute_pose)
                else:
                    self.get_logger().warn("Skipping publish because current pose is unavailable.")
            else:
                self.get_logger().warn(f"Backend response: {response.status_code}")
        except Exception as e:
            self.get_logger().error(f"Failed to send request: {e}")

    def get_current_width(self):
        """Compute current gripper width [m] from latest /joint_states."""
        try:
            j1 = self._joint_pos.get('panda_finger_joint1')
            j2 = self._joint_pos.get('panda_finger_joint2')
            if j1 is None or j2 is None:
                self.get_logger().warn("Finger joints not yet in JointState; width unavailable.")
                return None
            width = float(j1 + j2)
            return max(0.0, min(0.08, width))
        except Exception as e:
            self.get_logger().error(f"get_current_width error: {e}")
            return None

    @staticmethod
    def compute_absolute_width(current_width, delta, step=0.02, min_w=0.0, max_w=0.08):
        """Map delta_gripper to an absolute target width [m]."""
        dg = float(delta.get("delta_gripper", 0.0))
        base = current_width if current_width is not None else min_w
        target = base + step * dg
        if target < min_w:
            return min_w
        if target > max_w:
            return max_w
        return target

    def publish_pose(self, result):
        goal = CartesianImpedanceGoal()
        
        # For simplicity assume result provides absolute target pose:
        goal.pose.position.x = result["x"]
        goal.pose.position.y = result["y"]
        goal.pose.position.z = result["z"]
        
        goal.pose.orientation.x = result.get("qx", 0.0)
        goal.pose.orientation.y = result.get("qy", 0.0)
        goal.pose.orientation.z = result.get("qz", 0.0)
        goal.pose.orientation.w = result.get("qw", 1.0)
        
        goal.q_n = [0.0] * 7  # Just a placeholder; adjust if needed
        
        self.pose_pub.publish(goal)
        self.get_logger().info(f"Published desired pose: {goal}")

    def get_current_pose(self):
        # Fetch the current pose of panda_hand_tcp in panda_link0 frame.
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                'panda_link0',
                'panda_hand_tcp',
                now,
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            pose = PoseStamped()
            pose.header.frame_id = 'panda_link0'
            pose.pose.position.x = trans.transform.translation.x
            pose.pose.position.y = trans.transform.translation.y
            pose.pose.position.z = trans.transform.translation.z
            pose.pose.orientation = trans.transform.rotation

            return pose
        except Exception as e:
            self.get_logger().error(f"Failed to get current pose: {e}")
            return None

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
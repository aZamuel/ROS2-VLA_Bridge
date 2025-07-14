import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import requests
import base64
import cv2
from cv_bridge import CvBridge


class VLARequester(Node):
    def __init__(self):
        super().__init__('vla_requester')
        self.get_logger().info('VLA Requester node has been started.')

        # Configuration
        self.prompt = "Default prompt"
        self.backend_url = "http://localhost:5000/api/vla_request"
        self.request_interval = 1.0  # seconds

        # Internal state
        self.latest_image = None
        self.latest_joint_angles = []
        self.bridge = CvBridge()

        # Subscribers
        self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        # Timer for periodic requests
        self.timer = self.create_timer(self.request_interval, self.send_request_loop)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            _, buffer = cv2.imencode('.jpg', cv_image)
            self.latest_image = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def joint_state_callback(self, msg):
        self.latest_joint_angles = list(msg.position)

    def send_request_loop(self):
        if self.latest_image is None or not self.latest_joint_angles:
            self.get_logger().warn("Waiting for image and joint states...")
            return

        payload = {
            "prompt": self.prompt,
            "joint_angles": self.latest_joint_angles,
            "image": self.latest_image
        }

        try:
            response = requests.post(self.backend_url, json=payload)
            if response.status_code == 200:
                self.get_logger().info("Request successfully sent to backend.")
            else:
                self.get_logger().warn(f"Backend response: {response.status_code}")
        except Exception as e:
            self.get_logger().error(f"Failed to send request: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = VLARequester()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
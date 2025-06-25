import rclpy
from rclpy.node import Node


class VLARequester(Node):
    def __init__(self):
        super().__init__('vla_requester')
        self.get_logger().info('VLA Requester node has been started.')


def main(args=None):
    rclpy.init(args=args)
    node = VLARequester()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

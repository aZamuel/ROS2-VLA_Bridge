from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetLaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    robot_ip = LaunchConfiguration("robot_ip")
    backend  = LaunchConfiguration("backend_url")

    vla_client = Node(
        package="vla_client",
        executable="vla_bridge_node",
        name="vla_bridge_node",
        output="screen",
        parameters=[{
            "backend_url": backend,
            "request_interval": 0.2,
        }]
    )

    realsense = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="realsense2_camera",
        output="screen"
    )

    # Dockerfile copies into franka_bringup/launch/real/, not launch/
    franka = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("franka_bringup"), "launch", "real", "multimode_franka.launch.py"])
        ),
        launch_arguments={"robot_ip": robot_ip, "robot_ip_1": robot_ip}.items()
    )

    gripper = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("franka_gripper"), "launch", "gripper.launch.py"])
        ),
        launch_arguments={"robot_ip": robot_ip, "robot_ip_1": robot_ip}.items()
    )

    return LaunchDescription([
        DeclareLaunchArgument("robot_ip",   default_value="172.16.0.2"),
        DeclareLaunchArgument("backend_url",default_value="http://localhost:8000/predict"),
        SetLaunchConfiguration("robot_ip", robot_ip),
        SetLaunchConfiguration("robot_ip_1", robot_ip),
        franka,
        gripper,
        realsense,
        vla_client,
    ])
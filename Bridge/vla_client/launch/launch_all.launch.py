from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    robot_ip = LaunchConfiguration("robot_ip")
    prompt   = LaunchConfiguration("prompt")
    backend  = LaunchConfiguration("backend_url")
    autostart= LaunchConfiguration("autostart")
    req_hz   = LaunchConfiguration("request_hz")

    vla_client = Node(
        package="vla_client",
        executable="vla_bridge_node",   # installing this as a script via CMake
        name="vla_bridge_node",
        output="screen",
        parameters=[{
            "prompt": prompt,
            "backend_url": backend,
            "request_interval": 1.0,   # pass as string if node expects str to keep simple for now
            "active": autostart
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
        launch_arguments={"robot_ip_1": robot_ip}.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument("robot_ip",   default_value="172.16.0.2"),
        DeclareLaunchArgument("prompt",     default_value="Default prompt"),
        DeclareLaunchArgument("backend_url",default_value="http://localhost:8000/predict"),
        DeclareLaunchArgument("request_hz", default_value="1.0"),
        DeclareLaunchArgument("autostart",  default_value="false"),
        franka,
        realsense,
        vla_client,
    ])
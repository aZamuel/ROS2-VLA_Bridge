from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetLaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PythonExpression
from launch.conditions import IfCondition
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    robot_ip            = LaunchConfiguration("robot_ip")
    backend             = LaunchConfiguration("backend_url")
    record              = LaunchConfiguration('record')
    records_csv_path    = LaunchConfiguration('records_csv_path')

    profile = LaunchConfiguration("profile")
    profile_yaml = PathJoinSubstitution([
        FindPackageShare("vla_client"),
        "config",
        PythonExpression(["'", profile, "'", " + '.yaml'"])
    ])

    vla_client = Node(
        package="vla_client",
        executable="vla_bridge_node",
        name="vla_bridge_node",
        output="screen",
        parameters=[
            profile_yaml,
            {
                "backend_url": backend,
                "request_interval": 0.2,
                "record": ParameterValue(record, value_type=bool),
                "records_csv_path": ParameterValue(records_csv_path, value_type=str)
            }
        ]
    )

    realsense = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="realsense2_camera",
        output="screen",
        condition=IfCondition(PythonExpression(["'", profile, "'", " == 'real'"]))
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
        DeclareLaunchArgument("robot_ip",           default_value="172.16.0.2"),
        DeclareLaunchArgument("robot_ip_1",         default_value="172.16.0.2"),
        DeclareLaunchArgument("backend_url",        default_value="http://localhost:8000/predict"),
        DeclareLaunchArgument("profile",            default_value="real"),
        DeclareLaunchArgument('record',             default_value='false'),
        DeclareLaunchArgument('records_csv_path',   default_value='/logs/vla_records.csv'),
        SetLaunchConfiguration("robot_ip", robot_ip),
        SetLaunchConfiguration("robot_ip_1", robot_ip),
        franka,
        gripper,
        realsense,
        vla_client,
    ])
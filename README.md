## ROS2-VLA_Bridge
This is the Project to the bachelor thesis of Samuel Rochlitzer in Computer Science at the Eberhardt Karls Universität Tübingen. It aims to implement an Interface to an external VLA in the ROS2 Framework. It should ultimately work and be adaptable to different VLAs.

---

### How to Start the Requester

* As a starting point I used the ros2_jazzy Dockerfile from the RobotReplicationFiles provided by David Ott. Using the same commands one can start the docker by running these lines in the base repository:  
docker build -t ros2_jazzy_vla_bridge .  
docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --net=host --privileged ros2_jazzy_vla_bridge  

* To pass on any display correctly one should also run  
xhost +local:docker  
on the host system.

* In the container  
source /opt/ros/jazzy/setup.bash  
source /root/ws_moveit/install/setup.bash

* Now you can start th Node  
ros2 run ros2_vla_bridge_requester vla_requester_node

---


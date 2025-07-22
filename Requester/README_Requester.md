## ROS2-VLA_Bridge Requester
This is the ROS2 side of the Bridge. It allows to set a prompt, start a request loop and controll the Panda-robot.

---

### Summary Functions:

**1. ROS (panda3gpu):**  
1.1 Access point, set prompt and parameters, start feedback loop  
1.2 Get image data  
1.3 Send request to VLA  
1.4 Get results from VLA  
1.5 Pass instructions on to controller
---

### How to Start the requester node

* As a starting point I used the ros2_multipanda Dockerfile from the RobotReplicationFiles provided by David Ott. Using the same commands one can start the docker by running these lines in the base repository:  
docker build -t ros2_vla_bridge .  
docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --net=host --privileged --device=/dev/bus/usb ros2_vla_bridge  

* To pass on any display correctly one should also run  
xhost +local:docker  
on the host system.

* In the container  
source /opt/ros/humble/setup.bash  
source /root/humble_ws/install/setup.bash

* Now you can start the node  
ros2 run ros2_vla_bridge_requester vla_requester_node


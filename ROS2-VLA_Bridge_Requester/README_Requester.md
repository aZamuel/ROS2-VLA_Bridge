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

### How to Start the requester

* As a starting point I used the ros2_jazzy Dockerfile from the RobotReplicationFiles provided by David Ott. Using the same commands one can start the docker by running these lines in the base repository:  
docker build -t ros2_jazzy_vla_bridge ROS2-VLA_Bridge_Requester/  
docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --net=host --privileged ros2_jazzy_vla_bridge  

* To pass on any display correctly one should also run  
xhost +local:docker  
on the host system.


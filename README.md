## ROS2-VLA_Bridge
This is the Project to the bachelor thesis of Samuel Rochlitzer in Computer Science at the Eberhardt Karls Universität Tübingen. It aims to implement an Interface to an external VLA in the ROS2 Framework.

---

### Summary Functions:

**1. ROS (panda3gpu):**  
1.1 Access point, set prompt and parameters, start feedback loop  
1.2 Get image data  
1.3 Send request to VLA  
1.4 Get results from VLA  
1.5 Pass instructions on to controller

**2. VLA (Avalon):**  
2.1 Access point, starting services (loop)  
2.2 Receiving data  
2.3 Passing data on to VLA  
2.4 Getting results from VLA  
2.5 Sending results back  

---

### How to Start the requester

* As a starting point I used the ros2_jazzy Dockerfile from the RobotReplicationFiles provided by David Ott. Using the same commands one can start the docker by running these lines in the base repository:  
docker build -t ros2_jazzy_vla_bridge ROS2-VLA_Bridge_Requester/  
docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --net=host --privileged ros2_jazzy_vla_bridge  

* To pass on any display correctly one should also run  
xhost +local:docker  
on the host system.


# ROS2-VLA_Bridge

This is the Project to the bachelor thesis of Samuel Rochlitzer in Computer Science at the Eberhardt Karls Universität Tübingen. It aims to implement an Interface to an external VLA in the ROS2 Framework. It should ultimately work and be adaptable to different VLAs.

## How to ...
---

### ... start the requester node

* As a starting point I used the ros2_multipanda Dockerfile from the RobotReplicationFiles provided by David Ott. Using the same commands one can start the docker by running these lines in the base repository:  
docker build -t ros2_vla_bridge .  
docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --net=host --privileged --device=/dev/bus/usb ros2_vla_bridge  

* To pass on any display correctly one should also run  
xhost +local:docker  
on the host system.

* In the container  
source /opt/ros/humble/setup.bash  
source /root/humble_ws/install/setup.bash

* Now you can start nodes (each In a new Terminal):  
ros2 run ros2_vla_bridge_requester vla_requester_node  
ros2 run realsense2_camera realsense2_camera_node

### ... open new Terminal in docker

* Open new terminal (Str+Alt+T)  

* docker exec -it <container (Tab)> "bin/bash"

### ... use the requester node

* The vla_requester_node is startet inactive and with dummy values:  
    self.prompt = "Default prompt"  
    self.backend_url = "http://localhost:8000/predict"              # Flask route for VLA (still needs to be started)  
    self.request_interval = 1.0                                     # seconds  
    self.active = False                                             # Initially stopped  
    self.latest_image = self.generate_dummy_image()  
    self.latest_joint_angles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  
    self.bridge = CvBridge()  

* To start the request loop:  
    ros2 service call /vla_requester/toggle std_srvs/srv/SetBool "{data: true}"  

### ... start the VLA Wrapper

* For now the Flask app can be started in the container or an the host system.  

* In new Terminal:  
On Container ~/humble_ws:  
python3 VLA_Wrapper/app.py  
Or in the Repo:  
python3 Wrapper/app.py

---

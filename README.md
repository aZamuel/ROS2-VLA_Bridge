# ROS2-VLA_Bridge

This is the Project to the bachelor thesis of Samuel Rochlitzer in Computer Science at the Eberhardt Karls Universität Tübingen. It aims to implement an Interface to an external VLA in the ROS2 Framework. It should ultimately work and be adaptable to different VLAs.

## How to ...
---

### ... start the client node

* As a starting point I used the ros2_multipanda Dockerfile from the RobotReplicationFiles provided by my Tutor David Ott. Using the similar commands one can start the docker by running these lines in the base repository:  
docker build -t ros2_vla_bridge .  
docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --net=host --privileged --device=/dev/bus/usb ros2_vla_bridge  

* To pass on any display correctly one should also run  
xhost +local:docker  
on the host system.

* Now you can start all nodes (when robot is available!):  
ros2 launch vla_client launch_all.launch.py robot_ip:=172.16.0.2  

* Manually start nodes with (each In a new Terminal):  
ros2 run vla_client vla_bridge_node  
ros2 run realsense2_camera realsense2_camera_node  
ros2 launch franka_bringup multimode_franka.launch.py robot_ip_1:=172.16.0.2  
ros2 launch franka_gripper gripper.launch.py robot_ip:=172.16.0.2  

### ... open new Terminal in docker

* Open new terminal (Str+Alt+T)  

* docker exec -it <container (Tab)> "/bin/bash"

### ... use the client node

* The vla_bridge_node is startet inactive and with dummy values:  
    self.prompt = "Default prompt"  
    self.backend_url = "http://localhost:8000/predict"              # Flask route for VLA (still needs to be started)  
    self.request_interval = 1.0                                     # seconds  
    self.active = False                                             # Initially stopped  
    self.latest_image = self.generate_dummy_image()  
    self.latest_joint_angles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  
    self.bridge = CvBridge()  

* To call upon the services manually:  
ros2 service call /toggle_active std_srvs/srv/SetBool "{data: true}"  
ros2 service call /set_prompt vla_interfaces/srv/SetPrompt "{prompt: 'Pick up the red cube'}"  
ros2 service call /set_request vla_interfaces/srv/SetRequest "{model: 'openvla/openvla-7b', backend_url: 'http://localhost:8000/predict', request_interval: 0.5, prompt: 'put the duplo brick into the box', active: true}"  

### ... start the Backend

* For now the Flask server and VLA wrapper app should be started on the host system for now. Run the following on a Terminal in the base of the Repo.  

* To create the conda environment for the Backend. (ones on a new system):  
conda env create -f environment.yml (ones on a new system)  

* To activate the environment (with cpu fallback):  
conda activate openvla  

* To start the Backend:  
python3 Backend/vla_server.py  


* To get backend debugging image:  
curl -s http://localhost:8000/debug/last_image.jpg -o last.jpg  
and open in browser: http://localhost:8000/debug/view  

---

# ### Build with:
# 
#     docker build -t ros2-vla-bridge .
#
# ### Run with:
#
#    docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --net=host --privileged --device=/dev/bus/usb ros2_vla_bridge
#
# ### To test whether everything works: 
# # Graphical pass through
#
#    ros2 run rviz2 rviz2
#    ros2 run vla_client vla_bridge_node  
#    ros2 run realsense2_camera realsense2_camera_node
#    ros2 launch franka_bringup multimode_franka.launch.py robot_ip_1:=172.16.0.2
#
# # Mujoco
#
#   ~/Libraries/mujoco/bin/simulate
#
# # RT kernel and robot connection
#
#   ~/Libraries/libfranka/bin/communication_test 172.16.0.2
#
# ### Publish to the cartesian controller:
# ros2 topic pub /panda/panda_cartesian_impedance_controller/desired_pose multi_mode_control_msgs/msg/CartesianImpedanceGoal "{
#   pose: {
#     position: {
#       x: 0.682,
#       y: -0.015,
#       z: 0.390
#     },
#     orientation: {
#       x: 0.999,
#       y: -0.014,
#       z: 0.041, 
#       w: -0.001
#     }
#   },
#   q_n: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# }" -1

# # # NOTE:
#     - that the target pose may only have a maximum distance of 0.1m from the current pose
#     - the target frame is panda_hand_tcp
#     - current pose can be found with 
#         - ros2 run tf2_ros tf2_echo panda_link0 panda_hand_tcp
#     - the q_n is used as the desired joint_configuration to solve null-space redundancies


# ## Simulated
# 
#    ros2 launch franka_bringup franka_sim.launch.py
#


FROM ros:humble


ENV HOMEDIR=/root
ENV COLCON_WS=/root/humble_ws/

RUN apt-get update && apt-get upgrade -y
RUN apt-get install curl -y

RUN mkdir ${HOMEDIR}/Libraries
RUN mkdir ${HOMEDIR}/source_code 
RUN cd ${HOMEDIR}/source_code \
    && curl https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz --output eigen.tar.gz \
    && tar -xvzf eigen.tar.gz

RUN cd ~/source_code/eigen-3.3.9 \
    && mkdir build \ 
    && cd build \
    && cmake .. \
    && make \
    && make install \
    && cd ~/

# Install the package dependencies
# RUN apt-get update -y && apt-get install -y --allow-unauthenticated \
RUN apt-get update -y && apt-get install -y \
    software-properties-common \
    clang-14 \
    clang-format-14 \
    clang-tidy-14 \
    python3-pip \
    libpoco-dev \
    ros-humble-control-msgs \
    ros-humble-xacro \
    ros-humble-ament-cmake-clang-format \
    ros-humble-ament-clang-format \
    ros-humble-ament-flake8 \
    ros-humble-ament-cmake-clang-tidy \
    ros-humble-angles \
    ros-humble-ros2-control \
    ros-humble-realtime-tools \
    ros-humble-control-toolbox \    
    ros-humble-controller-manager \
    ros-humble-hardware-interface \
    ros-humble-hardware-interface-testing \
    ros-humble-launch-testing \
    ros-humble-generate-parameter-library \
    ros-humble-controller-interface \
    ros-humble-ros2-control-test-assets \
    ros-humble-controller-manager \
    ros-humble-moveit \
    ros-humble-nav-msgs \
    && rm -rf /var/lib/apt/lists/*


RUN python3 -m pip install -U \
    argcomplete \
    flake8-blind-except \
    flake8-builtins \
    flake8-class-newline \
    flake8-comprehensions \ 
    flake8-deprecated \
    flake8-docstrings \
    flake8-import-order \
    flake8-quotes \
    pytest-repeat \
    pytest-rerunfailures \
    pytest


# Build libfranka
RUN cd ${HOMEDIR}/source_code && git clone https://github.com/frankaemika/libfranka.git \
    && mkdir ${HOMEDIR}/Libraries/libfranka \
    && cd libfranka \
    && git checkout 0.9.2 \
    # && git checkout 0.8.0 \
    && git submodule init \
    && git submodule update \
    && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=${HOMEDIR}/Libraries/libfranka .. \
    && cmake --build . \
    && cmake --install .

# Install MuJoCo dependencies first
RUN apt-get update -y && apt-get install -y \
    libglfw3 \
    libglfw3-dev \
    libgl1-mesa-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxrandr-dev \
    libxi-dev \
    ninja-build \
    zlib1g-dev \
    clang-12

# Install dqrobotics
RUN add-apt-repository ppa:dqrobotics-dev/release && apt-get update && apt-get install libdqrobotics

# Install MuJoCo from scratch
RUN cd ~/source_code && git clone https://github.com/google-deepmind/mujoco.git \
    && cd mujoco \
    # Checkout the required version 3.2.0 of mujoco
    && git checkout 3.2.0 \
    && mkdir ~/source_code/mujoco/build \
    && mkdir ${HOMEDIR}/Libraries/mujoco \
    && cd ~/source_code/mujoco/build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=${HOMEDIR}/Libraries/mujoco \
    && cmake --build . \
    && cmake --install .


# Now copy the contents of the repository into a new workspace
RUN mkdir -p ${HOMEDIR}/humble_ws/src && cd ${HOMEDIR}/humble_ws/src \
    && git clone https://github.com/tenfoldpaper/multipanda_ros2


# Set up the environment variables
RUN echo 'source /opt/ros/humble/setup.bash' >> ${HOMEDIR}/.bashrc
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOMEDIR}/Libraries/libfranka/lib:${HOMEDIR}/Libraries/mujoco/lib' >> ${HOMEDIR}/.bashrc 
RUN echo 'export CMAKE_PREFIX_PATH=~/Libraries/libfranka/lib/cmake:~/Libraries/mujoco/lib/cmake' >> ${HOMEDIR}/.bashrc


# TODO for some reason these are not entirely correct, fix this!
# Additional Environment Variables for mujoco_ros_pkgs
RUN echo 'export MUJOCO_DIR=${HOMEDIR}/Libraries/mujoco' >> ${HOMEDIR}/.bashrc
# ENV MUJOCO_DIR=${HOMEDIR}/Libraries/mujoco
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO_DIR/lib' >> ${HOMEDIR}/.bashrc
RUN echo 'export LIBRARY_PATH=$LIBRARY_PATH:$MUJOCO_DIR/lib' >> ${HOMEDIR}/.bashrc
RUN echo 'export Franka_DIR=${HOMEDIR}/Libraries/libfranka' >> ${HOMEDIR}/.bashrc
RUN echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ${HOMEDIR}/.bashrc
# To make libspacemouse available

# Install mujoco_ros_pkgs
RUN cd ${HOMEDIR}/humble_ws/src \
    && git clone https://github.com/tenfoldpaper/mujoco_ros_pkgs

WORKDIR /root

SHELL ["/bin/bash", "-c"]

RUN source ~/.bashrc \
    && . /opt/ros/humble/setup.sh \
    && cd ~/humble_ws && rosdep update \
    && cd ~/humble_ws && rosdep install -i --from-path src --rosdistro humble -y
# Suppresss the XDG errors when running GUI apps like RVIZ

# Installing dependencies for space mouse
# RUN sudo apt-get update
# RUN sudo apt-get install libbluetooth-dev libcwiid-dev -y 
# RUN sudo apt install libx11-dev libxi-dev libxtst-dev libgl1-mesa-dev -y 
# RUN sudo apt-get install ros-${ROS_DISTRO}-diagnostic-updater -y

RUN apt-get update && apt-get install -y ros-humble-realsense2-camera

# --- Spacemouse driver & libspnav (disabled) ---
# RUN cd ~/source_code  \
#     && git clone https://github.com/FreeSpacenav/spacenavd \
#     && cd spacenavd \
#     && git checkout tags/v1.3.1 \
#     && ./configure \
#     && make \
#     && sudo make install \
#     && sudo ./setup_init \
#     && cd .. \
#     && git clone https://github.com/FreeSpacenav/libspnav \
#     && cd libspnav \
#     && git checkout tags/v1.2 \
#     && ./configure \
#     && make \
#     && sudo make install

# --- ROS joystick_drivers (disabled) ---
# RUN cd ~/humble_ws/src/ && \
#     git clone https://github.com/ros-drivers/joystick_drivers

# TODO copy small changes that are required to launch
COPY Bridge/vla_client/config/single_multimode.yaml /root/humble_ws/src/multipanda_ros2/franka_bringup/config/real/single_multimode.yaml
COPY Bridge/vla_client/launch/franka.launch.py /root/humble_ws/src/multipanda_ros2/franka_bringup/launch/real/franka.launch.py
COPY Bridge/vla_client/launch/multimode_franka.launch.py /root/humble_ws/src/multipanda_ros2/franka_bringup/launch/real/multimode_franka.launch.py

# ENV XDG_RUNTIME_DIR=/tmp/${UID}
ENV CMAKE_PREFIX_PATH=~/Libraries/libfranka/lib/cmake:~/Libraries/mujoco/lib/cmake
RUN cd ~/humble_ws \
    && source ~/.bashrc \
    && . /opt/ros/humble/setup.sh \
    && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

RUN echo 'source ${HOMEDIR}/humble_ws/install/setup.bash' >> ${HOMEDIR}/.bashrc
# RUN echo 'spacenavd' >> ${HOMEDIR}/.bashrc
# Autostart spacenavd (disabled)

# --- ROS2-VLA_Bridge specific ---

WORKDIR /root/humble_ws

# Setup VLA Wrapper for dev in container
COPY ./Backend ./VLA_Wrapper/
RUN apt-get update && apt-get install -y python3-pip
RUN python3 -m pip install -r VLA_Wrapper/requirements.txt
EXPOSE 8000
#CMD ["ros2", "run", "vla_client", "vla_bridge_node"]

# copy VLA Requester package into the workspace
COPY ./Bridge/vla_client/ ./src/vla_client/
RUN chmod +x ./src/vla_client/nodes/vla_bridge_node.py
COPY ./Bridge/vla_interfaces/ ./src/vla_interfaces/

# Install cv_bridge (ROS) and ensure OpenCV compatibility
RUN apt-get update && apt-get install -y \
     ros-${ROS_DISTRO}-cv-bridge

# Remove conflicting system OpenCV packages
RUN apt-get remove -y python3-opencv libopencv-dev libopencv-* opencv-data || true

# Reinstall NumPy and OpenCV via pip to ensure cv_bridge compatibility
RUN pip install --force-reinstall "numpy<2" opencv-python requests

RUN apt-get update && apt-get install -y ros-humble-realsense2-camera

RUN pip install scipy

# --- Spacemouse fallback install (disabled) ---
# RUN DEBIAN_FRONTEND=noninteractive \
#     apt-get install -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" \
#     spacenavd libspnav-dev libsdl2-dev

# install package dependencies and build
RUN source /opt/ros/$ROS_DISTRO/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y --rosdistro $ROS_DISTRO && \
    apt-get install -y ros-$ROS_DISTRO-cv-bridge && \
    colcon build --packages-up-to vla_client

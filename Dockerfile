
# ### Build with:
# 
#    docker build -t ros2_jazzy docker_ros2_jazzy/
#     
# OLD: docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg USERNAME=$(whoami) -t ros2_jazzy docker_ros2_jazzy/
#
# ### Run with:
#
#    docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --net=host --privileged ros2_jazzy
#
# ### For remote development use it without --rm
#
#    docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --net=host --privileged ros2_jazzy
#
# ### To test if everything works correctly, one can run the following and try if movement planning and execution with MoveIt works
# ### TODO this does not work yet
#
#    ros2 launch franka_moveit_config moveit.launch.py robot_ip:=172.16.0.2
# 


FROM moveit/moveit2:jazzy-source

# ARG USER_ID=101385
# ARG GROUP_ID=5011
# ARG USERNAME=ott


# ENV LIBRARIES_DIR=/home/$USERNAME/Libraries
# ENV COLCON_WS=/home/$USERNAME/ws_moveit/
# ENV SOURCE_CODE_DIR=/home/$USERNAME/source_code
# ENV HOMEDIR=/home/$USERNAME

ENV LIBRARIES_DIR=/root/Libraries
ENV COLCON_WS=/root/ws_moveit/
ENV SOURCE_CODE_DIR=/root/source_code
ENV HOMEDIR=/root


RUN apt update && apt install -y sudo

# # Create a new group and user with the same UID and GID as the host
# RUN groupadd --gid $GROUP_ID $USERNAME \
#     && useradd --uid $USER_ID --gid $GROUP_ID -m -s /bin/bash $USERNAME \
#     && echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# # Set working directory and change ownership
# RUN chown -R $USERNAME:$USERNAME /home/$USERNAME

# RUN rm -rf build && \
#     sudo mv /root/ws_moveit /home/$USERNAME/ && \
#     chown -R $USER_ID:$GROUP_ID ${COLCON_WS} 



# overwrite the entrypoint, which would already source workspaces, etc. 
ENTRYPOINT ["/bin/bash"]
SHELL ["/bin/bash", "-c"]

# USER $USERNAME

WORKDIR $HOMEDIR




RUN mkdir -p $SOURCE_CODE_DIR



RUN sudo apt update && sudo apt upgrade -y 
RUN sudo apt install -y \
    ninja-build \
    git \
    nano \
    wget \
    curl \
    unzip \
    libeigen3-dev \
    libpoco-dev


RUN sudo apt install ros-$ROS_DISTRO-rmw-cyclonedds-cpp -y
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# RUN colcon mixin add default https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml
RUN colcon mixin update default


# RUN sudo apt install -y \
#     build-essential \
#     cmake \
#     git \
#     python3-colcon-common-extensions \
#     python3-flake8 \
#     python3-rosdep \
#     python3-setuptools \
#     python3-vcstool \
#     wget


RUN sudo apt update
RUN sudo apt dist-upgrade -y
RUN rosdep update



### TODO libfranka 0.8.0 does not compile with gcc13 for some reasons (probably changes in the standard)
### I tested all versions from gcc9 upwards, and gcc9 is the only one that works
### -> replace gcc with gcc9
### TODO in principle, after building libfranka we could switch back to gcc13
RUN sudo apt-get update -y && sudo apt install gcc-9 g++-9 -y && \
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 10 && \
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 10


### TODO change libfranka to 0.8.0! (changed from 0.9.2)
# Build libfranka 
RUN cd $SOURCE_CODE_DIR \ 
    && git clone https://github.com/frankaemika/libfranka.git \
    && mkdir -p $LIBRARIES_DIR/libfranka \
    && cd libfranka \
    && git checkout 0.8.0 \ 
    && git submodule init \
    && git submodule update \
    && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=$LIBRARIES_DIR/libfranka .. \
    && cmake --build . \
    && cmake --install .

RUN sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 20 && \
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 20


### bypass git authentication for vcs import by setting empty username/pw 
RUN git config --global credential.helper 'cache --timeout=3600' && \
    git config --global credential.helper 'store --file ~/.git-credentials' && \
    touch ~/.git-credentials && \
    chmod 600 ~/.git-credentials
# Main Branch supports Jazzy
RUN cd $COLCON_WS/src && \
    git clone https://github.com/moveit/moveit2_tutorials
# HACKY bypass the github authentication and add || true to prevent docker from stopping from the errors 
RUN cd $COLCON_WS/src && \
    GIT_ASKPASS=/bin/true vcs import --recursive < moveit2_tutorials/moveit2_tutorials.repos || true




### TODO official franka_ros2 is not supporting the emika panda anymore (https://github.com/frankaemika/franka_ros)
### we use this fork: https://github.com/mul-cps/franka_ros2 (this is a new version of https://github.com/boschresearch/franka_ros2)
### but these promise to deliver the same: https://github.com/LCAS/franka_arm_ros2 and https://github.com/tenfoldpaper/multipanda_ros2
RUN cd $COLCON_WS/src && \
    git clone --recursive https://github.com/mul-cps/franka_ros2 # TODO at least with this one it builds ### TODO might only work with humble



RUN cd $COLCON_WS/src && \
    rosdep install -r --from-paths . --ignore-src --rosdistro $ROS_DISTRO -y


# Fix include to allow franka_ros2 to compile
# Not required with https://github.com/mul-cps/franka_ros2 anymore
# RUN sed -i 's/#include <hardware_interface\/visibility_control.h>/\/\/ #include <hardware_interface\/visibility_control.h>/g' $COLCON_WS/src/franka_ros2/franka_hardware/include/franka_hardware/franka_hardware_interface.hpp


RUN source /opt/ros/$ROS_DISTRO/setup.bash && \
    cd $COLCON_WS && \
    export Franka_DIR=$SOURCE_CODE_DIR/libfranka/build && \
    rm build -rf && \
    colcon build --mixin release --mixin ninja --cmake-args -DCMAKE_BUILD_TYPE=Release




# ## Librealsense installation from precompiled packages (required to detect multiple connected cameras correctly)
# ## the Beta debian packages are the easiest to install and support ubuntu 24.04 and ROS2 Jazzy
RUN cd $SOURCE_CODE_DIR && \
    mkdir librealsense2 && \
    cd librealsense2 && \
    wget https://github.com/IntelRealSense/librealsense/releases/download/v2.56.3/librealsense2_noble_x86_debians_beta.zip && \
    unzip librealsense2_noble_x86_debians_beta.zip && \
    sudo apt install ./librealsense2_2.56.3-0~realsense.14840_amd64.deb -y && \
    sudo apt install ./librealsense2-dev_2.56.3-0~realsense.14840_amd64.deb -y && \
    sudo apt install ./librealsense2-dbg_2.56.3-0~realsense.14840_amd64.deb -y && \
    sudo apt install ./librealsense2-gl_2.56.3-0~realsense.14840_amd64.deb -y && \
    sudo apt install ./librealsense2-gl-dev_2.56.3-0~realsense.14840_amd64.deb -y && \
    sudo apt install ./librealsense2-gl-dbg_2.56.3-0~realsense.14840_amd64.deb -y && \
    sudo apt install ./librealsense2-utils_2.56.3-0~realsense.14840_amd64.deb -y && \
    sudo apt install ./librealsense2-udev-rules_2.56.3-0~realsense.14840_amd64.deb -y


RUN sudo apt install ros-$ROS_DISTRO-realsense2-* -y
RUN sudo apt install ros-$ROS_DISTRO-image-view -y 
RUN sudo apt install ros-$ROS_DISTRO-rqt-graph -y 




# Set up the environment variables
RUN echo 'source /opt/ros/$ROS_DISTRO/setup.bash' >> ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBRARIES_DIR/libfranka/lib' >> ~/.bashrc 
RUN echo "source /root/ws_moveit/install/setup.bash" >> ~/.bashrc
RUN echo 'export Franka_DIR=$SOURCE_CODE_DIR/libfranka/build' >> ~/.bashrc

WORKDIR $COLCON_WS

# copy VLA Requester package into the workspace
#RUN mkdir -p src/ros2_vla_bridge_requester
COPY Requester/ ./src/ros2_vla_bridge_requester/

# install package dependencies and build
RUN source /opt/ros/$ROS_DISTRO/setup.bash && \
    rosdep install --from-paths src/ros2_vla_bridge_requester -i -y --rosdistro $ROS_DISTRO && \
    colcon build --packages-select ros2_vla_bridge_requester

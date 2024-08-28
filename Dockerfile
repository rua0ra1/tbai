FROM ubuntu:20.04

# Install ROS Noetic
RUN apt-get update && \
    apt-get install -y lsb-release curl && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt-get update && \
    apt-get install -y ros-noetic-desktop-full

# Set up environment
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"

# Install dependencies and setup workspace
RUN apt-get install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
RUN rosdep init && rosdep update


# Create a catkin workspace
RUN mkdir -p ~/catkin_ws/src
WORKDIR /root/catkin_ws/

# Copy your ROS package into the workspace
COPY . src/

# Install package dependencies
RUN rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Source the setup file
RUN echo "source /root/catkin_ws/devel/setup.bash" >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"


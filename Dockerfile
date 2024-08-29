# Use the official Ubuntu 20.04 base image
FROM ubuntu:20.04

# Set environment variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary packages
RUN apt-get update && \
    apt-get install -y \
    lsb-release \
    curl \
    gnupg2 \
    software-properties-common \
    build-essential \
    python3-rosdep \
    python3-pip

# Add ROS repository and keys
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# Install ROS Noetic Desktop Full

# Initialize rosdep
#RUN rosdep init && rosdep update

# Source ROS environment
#RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Create and initialize a catkin workspace
RUN mkdir -p /root/catkin_ws/src

# Set the working directory
WORKDIR /root/catkin_ws

# Source ROS setup file explicityly saying run this command using bash
#RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Set the default command to be executed in the container
#CMD ["/bin/bash"]



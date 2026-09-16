# Registered at https://hub.docker.com/repository/docker/chicheng/orb_slam3/general
FROM ubuntu:22.04

# copy files over
ADD . /ORB_SLAM3

# use non-interactive
ENV TZ=America/Los_Angeles
ENV DEBIAN_FRONTEND=noninteractive 

# install pangolin prereq
RUN apt-get update && apt install -y sudo
RUN echo "Y" | bash /ORB_SLAM3/Thirdparty/Pangolin/scripts/install_prerequisites.sh

# install orb slam prereq
RUN apt install -y build-essential libopencv-dev libboost-dev libboost-serialization-dev libssl-dev

# build
RUN cd /ORB_SLAM3 && bash build.sh

## RGBD-Inertial Trajectory Estimation and Mapping for Small Ground Rescue Robot
Based one open source SLAM framework [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono).

The approach contains
+ Depth-integrated visual-inertial initialization process.
+ Visual-inertial odometry by utilizing depth information while avoiding the limitation is working for 3D pose estimation.
+ Noise elimination map which is suitable for path planning and navigation.

However, the proposed approach can also be applied to other application like handheld and wheeled robot.

This dataset is part of the dataset collection of the [STAR Center](https://star-center.shanghaitech.edu.cn/), [ShanghaiTech University](http://www.shanghaitech.edu.cn/eng): https://star-datasets.github.io/

A video showing the data is available here: https://robotics.shanghaitech.edu.cn/datasets/VINS-RGBD

## Paper
Shan, Zeyong, Ruijian Li, and Sören Schwertfeger. "RGBD-inertial trajectory estimation and mapping for ground robots." Sensors 19.10 (2019): 2251.


    @article{shan2019rgbd,
      title={RGBD-inertial trajectory estimation and mapping for ground robots},
      author={Shan, Zeyong and Li, Ruijian and Schwertfeger, S{\"o}ren},
      journal={Sensors},
      volume={19},
      number={10},
      pages={2251},
      year={2019},
      publisher={Multidisciplinary Digital Publishing Institute}
    }


## 1. Prerequisites
1.1. **Ubuntu** 16.04 or 18.04.

1.2. **ROS** version Kinetic or Melodic fully installation

1.3. **Ceres Solver**
Follow [Ceres Installation](http://ceres-solver.org/installation.html)

1.4. **Sophus**
```
  git clone http://github.com/strasdat/Sophus.git
  git checkout a621ff
```


## 2. Datasets
Recording by RealSense D435i. Contain 9 bags in three different applicaions:
+ [Handheld](https://star-center.shanghaitech.edu.cn/seafile/d/0ea45d1878914077ade5/)
+ [Wheeled robot](https://star-center.shanghaitech.edu.cn/seafile/d/78c0375114854774b521/) ([Jackal](https://www.clearpathrobotics.com/jackal-small-unmanned-ground-vehicle/))
+ [Tracked robot](https://star-center.shanghaitech.edu.cn/seafile/d/f611fc44df0c4b3d936d/)

Note the rosbags are in compressed format. Use "rosbag decompress" to decompress.

Topics:
+ depth topic: /camera/aligned_depth_to_color/image_raw
+ color topic: /camera/color/image_raw
+ imu topic: /camera/imu


## 3. Quick Start(Run with Docker)
### 3.1. Build Docker Image
make Dockerfile like below.
```c
FROM ros:melodic-ros-core-bionic

# apt-get update
RUN apt-get update

# install essentials
RUN apt install -y gcc
RUN apt install -y g++
RUN apt-get install -y cmake
RUN apt-get install -y wget
RUN apt install -y git

# install ceres
WORKDIR /home
RUN apt-get install -y libgoogle-glog-dev libgflags-dev
RUN apt-get install -y libatlas-base-dev
RUN apt-get install -y libeigen3-dev
RUN apt-get install -y libsuitesparse-dev
RUN wget http://ceres-solver.org/ceres-solver-2.1.0.tar.gz
RUN tar zxf ceres-solver-2.1.0.tar.gz
WORKDIR /home/ceres-solver-2.1.0
RUN mkdir build
WORKDIR /home/ceres-solver-2.1.0/build
RUN cmake ..
RUN make
RUN make install

# install sophus
WORKDIR /home
RUN git clone https://github.com/demul/Sophus.git
WORKDIR /home/Sophus
RUN git checkout fix/unit_complex_eror
RUN mkdir build
WORKDIR /home/Sophus/build
RUN cmake ..
RUN make
RUN make install

# install ros dependencies
WORKDIR /home
RUN mkdir ros_ws
WORKDIR /home/ros_ws
RUN apt-get -y install ros-melodic-cv-bridge
RUN apt-get -y install ros-melodic-nodelet
RUN apt-get -y install ros-melodic-tf
RUN apt-get -y install ros-melodic-image-transport
RUN apt-get -y install ros-melodic-rviz
RUN apt-get -y install ros-melodic-pcl-ros

# build vins-rgbd
RUN mkdir src
WORKDIR /home/ros_ws/src
RUN git clone https://github.com/STAR-Center/VINS-RGBD
WORKDIR /home/ros_ws
RUN /bin/bash -c ". /opt/ros/melodic/setup.bash; cd /home/ros_ws; catkin_make"
RUN echo "source /home/ros_ws/devel/setup.bash" >> ~/.bashrc
```
docker build --tag vins_rgbd:1.0 .

### 3.2. Run Docker Container with X11
docker run -it --name vins_rgbd -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -p 9999:9999 -p 9999:9999 vins_rgbd:1.0

### 3.3. Launch VINS-RGBD inside Container
#### 3.3.1. Terminal-1
roslaunch vins_estimator realsense_color.launch 
#### 3.3.2. Terminal-2
roslaunch vins_estimator vins_rviz.launch 
#### 3.3.2. Terminal-3
rosbag play some_demo_data

## 4. Licence
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.

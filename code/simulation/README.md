# README

In the following instructions, I assume you have already installed ROS Melodic (http://wiki.ros.org/melodic/Installation/Ubuntu) in Ubuntu 18.04 (https://www.ubuntu.com/download/desktop).

## Create a workspace

First, you will need to create a ROS workspace. To do that, follow the instructions at http://wiki.ros.org/catkin/Tutorials/create_a_workspace.

## Catkin Tools

In the following sections, I will assume that you are using Catkin Tools, which can be installed by following the instructions at https://catkin-tools.readthedocs.io/en/latest/installing.html.

Then rename the folders that are created after a build of the workspace, so that they do not clash with the folders created by `catkin_make`. Inside your workspace, issue the following command

    catkin config --init --space-suffix _ct

## Required catkin packages

To execute the code under this `simulation` folder, you will need the following ROS packages in your workspace

1. `pioneer3at_simulation`
2. `thymioid`
3. `thymio_msgs`
4. `asebaros_msgs`
5. `thymio_controller`

where `thymio_msgs` and `asebaros_msgs` will be installed from [`ros-aseba`](https://github.com/jeguzzi/ros-aseba). The `thymio_controller` contains the ad-hoc controller for the Pioneer3at robot, which is used to move the robot, while recording the bag files, which will be used to create the dataset for training and testing the neural network.

Issue the following commands inside the `src` folder of your workspace.

### Install `pioneer3at_simulation`

    git clone https://github.com/nbro/pioneer3at_simulation.git

### Install `thymioid`

    git clone -b client https://github.com/jeguzzi/thymioid.git

### Install `asebaros_msgs` and `thymio_msgs`

    git clone -b client https://github.com/jeguzzi/ros-aseba


## Build the workspace

First, `cd` the root folder of your workspace, then issue the following command

    catkin build


## Source the `setup.bash` file

After building the workspace, source the `setup.bash` (under the `devel` folder of your workspace)

    source devel/setup.bash

Or add the following to the `~/.bashrc` (so that you don't have to source the `setup.bash` file every time)

    source ~/catkin_ws/devel_ct/setup.bash

## Launch the simulator

Now, you should lunch the Gazebo simulator, which will be used to simulate a robot that roams a simple environment (a colored plane)

    roslaunch pioneer3at_simulation gazebo.launch world_name:=color_plane

## How to run the Pioneer controller?

After having built the workspace and lunched the simulator, start the controller that will make the robot roam the simple colored environment

    rosrun thymio_controller prand_walk.py 
    
where `prand_walk.py` is the Python script that contains the controller.

## Record the bag file

We will now record a bag file, which will be used to create the training and test datasets, while the robot roams the environment. 

We are only interested in the topics: `/pioneer3at/odom`, `/pioneer3at/camera_one/image_raw/compressed` and `/pioneer3at/camera_down/image_raw/compressed`. So, issue the following command from the terminal

    rosbag record --output-name=<name-of-bag-file.bag> --duration=10m /pioneer3at/odom /pioneer3at/camera_one/image_raw/compressed /pioneer3at/camera_down/image_raw/compressed

The recording of the messages will stop after 10 minutes. At this point, you should see a `.bag` file under the folder you had started recording. This `.bag` file will be converted to a `.h5` file, which will represent the training and test datasets.

## Stop the controller

To stop the controller, you just press <kbd>control + C</kbd> (and the robot should stop). 

## Create the dataset

To create the training and test datasets (to train a neural network), you will need to install several dependencies. First, you should have `pip` (for Python 2.7) installed. Then you should install `virtualenv`, create a virtual environment (with Python 2.7) and source it (e.g. `source /venv/bin/activate`, where `venv` is the name of your virtual environment). Then issue the following command to install all the required dependencies both to create the dataset and to train and test the get_model

    pip install -r requirements.txt

You _might_ need to add `export PYTHONPATH=$PYTHONPATH:/usr/lib/python2.7/dist-packages` to the `~/.bashrc` file, in order to make `rospkg` visible.

Then issue the following command

    python preprocess.py
    
However, the Python script might get killed by the system. See https://answers.ros.org/question/317590/python-is-killed-while-reading-a-bag-file-using-read_messages/. In that case, for example, record a smaller bag file or allow the VM to use more RAM.

## Train the get_model

    python train.py -d "<h5-dataset-file-name>.h5"
# VM and ROS installation

The following instructions will allow USI students to install ROS in an Ubuntu virtual machine (with the VMWare Fusion hypervisor).

## Installation

- Download VMWare Fusion 11 for free from http://vmap.usi.ch/
- Download Ubuntu 18.04.2 LTS from https://www.ubuntu.com/download/desktop
- Create a new virtual machine (takes about 5 min and will result in a 15GB machine) from the image
    - example of configuration (depends on your host resources):
        - RAM 4096MB
        - 4 processors
        - 2048Mb VRAM (found in sub-menu)
        - 80 GB Hard Disk space
- When defining the network configuration, choose "bridged network" 
- Boot the created virtual machine a first time (this will run some first setup scripts), and reboot it.

- By default, VMWare Fusion should install automatically (or prompt an installation window) VMWare Tools 
    - See: https://kb.vmware.com/s/article/1022525

- Reboot the virtual machine 
- Run the created virtual machine and in a terminal execute:
    1. `sudo add-apt-repository ppa:xorg-edgers/ppa -y`
    2. `sudo apt upgrade`
    3. `sudo reboot`
    4. Install ROS melodic full desktop (which comprises gazebo 9): http://wiki.ros.org/melodic/Installation/Ubuntu

- Add `source /opt/ros/melodic/setup.bash` to  `~/.bashrc`
- Add `export SVGA_VGPU10=0` to `~/.bashrc`

## Testing

- Before continuing, log out from the system and log back in, so the changes in `.bashrc` have effect.
- Type the following commands (one by one) in a terminal and verify the result (indicated between `[ ]`).

- `roscore` [it should display that roscore is running (you can stop it by pressing ctrl + c)]
- While `roscore` is running in one terminal, in another terminal execute:
    - `rqt` [It will open a window]
    - `rviz` [It will open the rviz visualiser ]
    - `gazebo` [It should open the gui of gazebo simulator]

## Post installation

Install catkin_tools (see "Installing on Ubuntu with apt-get" https://catkin-tools.readthedocs.io/en/latest/installing.html)

## Possible issues

- Error in VM fusion "Cannot find a valid peer process to connect to ": see https://superuser.com/a/1258547
- "Mware Could not open /dev/vmmon"
    - See this thread: https://communities.vmware.com/thread/600496
        - It might contain the solution to this problem
        - In my case, I did the following things
            1. System Preferences > Security & Privacy > Privacy (tab) > Full Disk Access > Manually add VMWare Fusion (I also did the same thing for Accessibility as opposed to Full Disk Access)
            2. Reboot the OS
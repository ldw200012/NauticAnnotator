#!/usr/bin/python3

import os
import os.path as osp
import rospy
import rospkg
import subprocess

if __name__ == '__main__':

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('pillars_reid3d')
    bash_path = os.path.join(pkg_path, "scripts/train.sh")

    METHOD = os.path.join(pkg_path, "config")
    WORK_DIR = os.path.join(pkg_path, "logs")

    result = subprocess.run([bash_path, METHOD, WORK_DIR])
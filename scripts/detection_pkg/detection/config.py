import os
from os.path import join as pjoin

project_path = os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath(__file__))))

CLASSES = [
    'tray',
    'dosing_nozzle',
    'button_pad',
    'part1',
    'part2',
    'part3'
    ]

CLASSES_GRASP = [
    'tray',
    'dosing_nozzle',
    'button_pad',
    'part1',
    'part2',
    'part3',
    'grasping_tag',
    'grasping_cylinder',
    'grasping_cuboid']


if __name__ == '__main__':
    pass
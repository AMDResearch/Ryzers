from setuptools import setup
import os
from glob import glob

package_name = 'vlm_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'requests'],
    zip_safe=True,
    description='ROS2 node for Vision Language Model integration',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vlm_node = vlm_ros.vlm_node:main',
        ],
    },
)

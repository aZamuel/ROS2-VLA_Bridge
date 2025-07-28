from setuptools import setup, find_packages
from glob import glob

package_name = 'vla_client'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(where='.', include=['vla_client', 'vla_client.*']),
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.py')),
        (f'share/{package_name}/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='VLA ROS2 bridge client',
    license='MIT',
    entry_points={
        'console_scripts': [
            'vla_bridge_node = vla_client.nodes.vla_bridge_node:main',
        ],
    },
)
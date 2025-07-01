from setuptools import setup

package_name = 'ros2_vla_bridge_requester'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README_Requester.md']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='UNKNOWN',
    maintainer_email='unknown@example.com',
    description='Empty VLA requester node package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vla_requester_node = ros2_vla_bridge_requester.vla_requester_node:main'
        ],
    },
)

#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import time, os
from utils import LineTrajectory
from tf.transformations import euler_from_quaternion

class PathPlan(object):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic")

        # Map handling
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.on_map_change)
        self.map_info = None
        self.map_transform = None
        self.map_transform_inverse = None
        self.is_map_valid = False

        # Trajectory handling
        self.trajectory = LineTrajectory("/planned_trajectory")
        self.traj_pub = rospy.Publisher("/trajectory/current", PoseArray, queue_size=10)

        # Map handling
        self.grid = None
        self.grid_height = None
        self.grid_width = None

        # Search handling
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.on_goal_change, queue_size=10)
        self.goal_location = None
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.on_get_odometry)
        self.start_location = None


    def on_map_change(self, msg):
        # Gather map transformation
        self.map_info = msg.info
        rotation_quaternion = self.map_info.orientation
        rotation = euler_from_quaternion(rotation_quaternion.x, rotation_quaternion.y, rotation_quaternion.z, rotation_quaternion.w).z
        translation = self.map_info.origin.position
        self.map_transform = np.array([[np.cos(rotation), -np.sin(rotation), translation.x], [np.sin(rotation), np.cos(rotation), translation.y], [0, 0, 1]])
        self.map_transform_inverse = np.linalg.inv(self.map_transform)

        # Build map grid
        self.grid_height = msg.info.height
        self.grid_width = msg.info.width
        self.grid = np.reshape(np.array(msg.data), (self.grid_height, self.grid_width))

        self.is_map_valid = True
        rospy.loginfo("Map Initialized")

    def on_get_odometry(self, msg):
        position = msg.pose.pose.position
        self.start_location = (position.x, position.y)

    def on_goal_change(self, msg):
        position = msg.pose.position
        self.goal_location = (position.x, position.y)

        if self.is_map_valid:
            self.plan_path_search_based(self.start_location, self.goal_location, self.grid)

    def plan_path_search_based(self, start_point, end_point, map):
        ## CODE FOR PATH PLANNING ##

        # publish trajectory
        self.traj_pub.publish(self.trajectory.toPoseArray())

        # visualize trajectory Markers
        self.trajectory.publish_viz()

    def pixel_to_real(self, pixel_coords):
        pixel_vector = np.array([[pixel_coords[0]*self.map_info.resolution],[pixel_coords[1]*self.map_info.resolution],[1]])
        real_vector = self.map_transform * pixel_vector
        return (real_vector[0][0], real_vector[1][0])

    def real_to_pixel(self, real_coords):
        real_vector = np.array([[real_coords[0]],[real_coords[1]],[1]])
        pixel_vector = self.map_transform_inverse * real_vector
        return (pixel_vector[0][0] / self.map_info.resolution, pixel_vector[1][0] / self.map_info.resolution)

if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()

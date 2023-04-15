#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import time, os
from utils import LineTrajectory

class PathPlan(object):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic")
        
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.on_map_change)
        self.trajectory = LineTrajectory("/planned_trajectory")
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.on_goal_change, queue_size=10)
        self.traj_pub = rospy.Publisher("/trajectory/current", PoseArray, queue_size=10)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.on_get_odometry)


    def on_map_change(self, msg):
        pass ## REMOVE AND FILL IN ##

    def on_get_odometry(self, msg):
        pass ## REMOVE AND FILL IN ##


    def on_goal_change(self, msg):
        pass ## REMOVE AND FILL IN ##

    def plan_path(self, start_point, end_point, map):
        ## CODE FOR PATH PLANNING ##

        # publish trajectory
        self.traj_pub.publish(self.trajectory.toPoseArray())

        # visualize trajectory Markers
        self.trajectory.publish_viz()


if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()

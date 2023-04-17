#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import time, os
from utils import LineTrajectory
from tf.transformations import euler_from_quaternion
import math
import heapq

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
        rotation_quaternion = self.map_info.origin.orientation
        rotation = euler_from_quaternion((rotation_quaternion.x, rotation_quaternion.y, rotation_quaternion.z, rotation_quaternion.w))[2]
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
        if self.is_map_valid:
            position = msg.pose.pose.position
            self.start_location = self.real_to_pixel((position.y, position.x))

    def on_goal_change(self, msg):
        position = msg.pose.position
        self.goal_location = self.real_to_pixel((position.y, position.x))

        if self.is_map_valid and self.start_location != None:
            self.plan_path_search_based(self.start_location, self.goal_location, self.grid)

    def plan_path_search_based(self, start_point, end_point, map):
        # A* graph search as introduced in MIT 6.009

        agenda = [(0,0,self.start_location)] # Priority queue of nodes to visit
        seen = set() # Nodes that have already been visited
        parents = {} # Maps nodes to where they came from

        while agenda:
            # Take next unfinished task
            total_cost, distance, cell = heapq.heappop(agenda)
            if cell in seen:
                continue
            seen.add(cell)


            # If reached goal, terminate search
            if cell == self.goal_location:
                # Build best path
                parent = self.goal_location
                path = [parent]
                while parent != None:
                    parent = parents[parent]
                    path.append(parent)
                path = path[::-1]

                # Create Trajectory 
                #TODO: Improve with Dubian Curve
                self.trajectory = LineTrajectory("/planned_trajectory")
                for point in path:
                    point_y = self.pixel_to_real(point[0])
                    point_x = self.pixel_to_real(point[1])
                    self.trajectory.addPoint(Point(point_x, point_y, 0))

                # Publish trajectory
                self.traj_pub.publish(self.trajectory.toPoseArray())

                # Visualize trajectory Markers
                self.trajectory.publish_viz()

                rospy.loginfo("Path found and trajectory plotted")
                break

            # Gather neighbors
            neighbors = []
            #print(cell, (max(0, cell[0]-1), min(cell[0]+2, self.grid_height)), (max(0, cell[1]-1), min(cell[1]+2, self.grid_width)), self.grid_height, self.grid_width)
            for y in range(max(0, cell[0]-1), min(cell[0]+2, self.grid_height)):
                for x in range(max(0, cell[1]-1), min(cell[1]+2, self.grid_width)):
                    neighbor = (y, x)
                    if neighbor not in seen and self.grid[y, x] == 0 : #Ensures no collision with obstacles
                        neighbors.append(neighbor)

            # Add neighbors to agenda
            for neighbor in neighbors:
                # Compute new costs
                # Done through optimized Euclidean Distance function 
                # https://stackoverflow.com/questions/37794849/efficient-and-precise-calculation-of-the-euclidean-distance
                next_distance = distance + math.sqrt(sum([(a - b)**2 for a, b in zip(cell, neighbor)]))   
                heuristic = math.sqrt(sum([(a - b)**2 for a, b in zip(neighbor, self.goal_location)]))    

                # Add to agenda
                parents[neighbor] = cell
                heapq.heappush(agenda, (next_distance+heuristic, next_distance, neighbor))

        rospy.loginfo("Path Planning finished")



    def pixel_to_real(self, pixel_coords):
        pixel_vector = np.array([[pixel_coords[1]*self.map_info.resolution],[pixel_coords[0]*self.map_info.resolution],[1]])
        real_vector = np.matmul(self.map_transform,pixel_vector)
        return (real_vector[1][0], real_vector[0][0])

    def real_to_pixel(self, real_coords):
        real_vector = np.array([[real_coords[1]],[real_coords[0]],[1]])
        pixel_vector = np.matmul(self.map_transform_inverse,real_vector)
        print(real_coords, (pixel_vector[1][0], pixel_vector[0][0]))
        return (int(pixel_vector[1][0] / self.map_info.resolution), int(pixel_vector[0][0] / self.map_info.resolution))

if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()

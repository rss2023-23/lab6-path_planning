#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray, Point
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import time, os
from utils import LineTrajectory
from tf.transformations import euler_from_quaternion
import math
import heapq
from scipy import ndimage

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
        self.path_resolution = 2

    def init_RRT_vars(self):
        self.V_adj = {self.start_location: set()}
        self.parents = {self.start_location: self.start_location}
        self.costs = {self.start_location: 0}
        # self.p_best_in_target = None
        # self.cur_total_cost = None
        self.exploration_bias = 0.1
        self.min_vertices = 1000
        self.total_size = (self.grid_height + self.grid_width + 1) // 2
        self.step_size = int(math.sqrt(self.total_size))  # eta
        self.NUM_DIM = 2  # d
        self.MU_X_FREE = self.total_size ** self.NUM_DIM # Lebesgue measure of X_free space, upper bounded by X_sample
        self.ZETA_DIM = math.pi * 1 * 1 #area of ball in d dimensions
        self.GAMMA_RRT_STAR = 2 * ((1 + 1/self.NUM_DIM) * self.MU_X_FREE / self.ZETA_DIM)**(1/self.NUM_DIM)

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

        # Erode the map
        self.grid = ndimage.binary_dilation(self.grid, iterations=14)

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
            self.plan_path_sample_based()

    def plan_path_sample_based(self):
        # RRT* graph search as introduced in https://arxiv.org/pdf/1105.1186.pdf
        self.init_RRT_vars()
        num_vertices = 0
        while self.goal_location not in self.V_adj or num_vertices < self.min_vertices:
            self.add_vertex_RRT_Star()
            num_vertices += 1
        self.publish_path_to_goal()

    def is_within_map(self, point):
        y, x = point
        return self.grid[y, x] == 0
    
    def distance(self, p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

    def cost(self, p1, p2):
        return self.distance(p1, p2)

    def get_nearest(self, goal_point):   # O(num vertices in V)
        min_dist = self.grid_height + self.grid_width
        min_v = (-1,-1)
        for v in self.V_adj:
            d = self.distance(v, goal_point)
            if d < min_dist:
                min_dist = d
                min_v = v
        return min_v

    def get_near(self, goal_point, dist):   # O(num vertices in V)
        return [v for v in self.V_adj if self.distance(v, goal_point) < dist]

    def point_along_at(self, p1, p2, proportion):
        p_new = tuple()
        for i in [0,1]:
            p_new += (int(p1[i] + proportion * (p2[i] - p1[i])),)
        return p_new
    
    def move_towards(self, p_i, p_f):  # O(1)
        line_length = self.distance(p_i, p_f)
        if line_length <= self.step_size:
            return p_f
        proportion = self.step_size / line_length
        return self.point_along_at(p_i, p_f, proportion)
    
    def line_is_within_map(self, p1, p2):
        # Compute the indices and coordinates along the line
        y1, x1 = p1
        y2, x2 = p2
        num_points = max(abs(x2-x1), abs(y2-y1)) + 1
        idx = np.linspace(0, 1, num_points)[1:-1]
        x = np.round(x1 + idx * (x2-x1)).astype(int)
        y = np.round(y1 + idx * (y2-y1)).astype(int)

        # Check if all the cells along the line are unoccupied
        for i in range(len(x)):
            if self.grid[y[i], x[i]] != 0:
                return False

        return True
    
    def update_costs(self, v):
        self.costs[v] = self.costs[self.parents[v]] + self.cost(self.parents[v], v)
        for neighbor in self.V_adj[v]:
            if neighbor != self.parents[v]:
                self.update_costs(neighbor)

    def add_vertex_RRT_Star(self):   # O(num nodes)
        # rospy.loginfo("Add")
        # Choose random point within the map bounds
        while True:
            p_random = (np.random.randint(0, self.grid_height),
                        np.random.randint(0, self.grid_width))
            if self.is_within_map(p_random):
                break
        if np.random.uniform() < self.exploration_bias:
            p_random = self.goal_location

        # Create new point by moving towards the chosen random point
        p_nearest = self.get_nearest(p_random)
        p_new = self.move_towards(p_nearest, p_random)
        if p_new not in self.V_adj and self.line_is_within_map(p_nearest, p_new):
            RRT_Star = min(self.GAMMA_RRT_STAR * (math.log(len(self.V_adj)) / len(self.V_adj))**(1/self.NUM_DIM), 2.5*self.step_size)
            P_near = self.get_near(p_new, RRT_Star)

            # Find minimum cost to reach p_new
            p_min, c_min = p_nearest, self.costs[p_nearest] + self.cost(p_nearest, p_new)
            for p_near in P_near:
                if self.line_is_within_map(p_near, p_new) and self.costs[p_near] + self.cost(p_near, p_new) < c_min:
                    c_min = self.costs[p_near] + self.cost(p_near, p_new)
                    p_min = p_near
            self.V_adj.setdefault(p_min, set()).add(p_new)
            self.V_adj.setdefault(p_new, set()).add(p_min)
            self.parents[p_new] = p_min
            self.costs[p_new] = c_min

            # Rewire the tree with updated minimum costs through p_new
            for p_near in P_near:
                if self.line_is_within_map(p_new, p_near) and self.costs[p_new] + self.cost(p_new, p_near) < self.costs[p_near]:
                    self.V_adj[p_near].remove(self.parents[p_near])
                    self.V_adj[self.parents[p_near]].remove(p_near)
                    self.parents[p_near] = p_new
                    self.update_costs(p_near)
                    self.V_adj.setdefault(p_near, set()).add(p_new)
                    self.V_adj.setdefault(p_new, set()).add(p_near)
    
    # def closest_to_target(self):
    #     # Update goal path and cost text
    #     p_closest_to_target = None
    #     self.cur_total_cost = None
    #     for node in self.V_adj:
    #         if self.in_target(node) and (self.p_best_in_target is None or self.costs[node] <= self.costs[self.p_best_in_target]):
    #             self.p_best_in_target = node
    #     if self.p_best_in_target:
    #         self.cur_total_cost = self.costs[self.p_best_in_target]
    #         self.path_to_goal(self.p_best_in_target)
    
    def publish_path_to_goal(self):
        rospy.loginfo("Goal")
        rospy.loginfo("{} {}".format(self.grid_height, self.grid_width))
        # Construct path in reverse using parents dict
        # Precondition: Goal location must have been found using RRT*
        path = []
        cur = self.goal_location
        while cur != self.start_location:
            path.append(cur)
            cur = self.parents[cur]
            rospy.loginfo(cur)
        path.append(self.start_location)
        path.reverse()
        rospy.loginfo("Here")

        self.trajectory = LineTrajectory("/planned_trajectory")
        for path_point in path:
            point = self.pixel_to_real(path_point)
            self.trajectory.addPoint(Point(point[1], point[0], 0))
        
        # Publish trajectory
        self.traj_pub.publish(self.trajectory.toPoseArray())

        # Visualize trajectory Markers
        self.trajectory.publish_viz()

        rospy.loginfo("Path found and trajectory plotted")

    def pixel_to_real(self, pixel_coords):
        pixel_vector = np.array([[pixel_coords[1]*self.map_info.resolution],[pixel_coords[0]*self.map_info.resolution],[1]])
        real_vector = np.matmul(self.map_transform,pixel_vector)
        return (real_vector[1][0], real_vector[0][0])

    def real_to_pixel(self, real_coords):
        real_vector = np.array([[real_coords[1]],[real_coords[0]],[1]])
        pixel_vector = np.matmul(self.map_transform_inverse,real_vector)
        #print(real_coords, (pixel_vector[1][0], pixel_vector[0][0]))
        return (int(pixel_vector[1][0] / self.map_info.resolution), int(pixel_vector[0][0] / self.map_info.resolution))

if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()

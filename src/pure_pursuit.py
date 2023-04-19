#!/usr/bin/env python

import rospy
import numpy as np
import time
import utils
import tf
import math
import tf2_ros

from geometry_msgs.msg import PoseArray, PoseStamped, PointStamped
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class PurePursuit(object):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """
    def __init__(self):
        self.odom_topic       = rospy.get_param("~odom_topic")
        self.lookahead        = rospy.get_param('lookahead',1.0)
        self.speed            = rospy.get_param('VELOCITY', 1.0)
        self.wheelbase_length = 0.325
        self.trajectory  = utils.LineTrajectory("/followed_trajectory")
        self.pose = None

        # odometry estimates
        self.x = 0
        self.y = 0
        self.theta = 0
        self.odometry_initialized = False
        # intersect intilization
        self.segment_index = 0
        self.tdex = 0
        self.steering_angle = None
        self.num_segments = None

        # points initialization
        self.points = None

        # initialize subscribers
        self.traj_sub = rospy.Subscriber("/trajectory/current", PoseArray, self.trajectory_callback, queue_size=10)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odometry_callback, queue_size=10)
        # self.transform_sub = rospy.Subscriber('/map_coordinates', PointStamped, self.transform_callback, queue_size=1)

        self.tf_listener = tf.TransformListener()

        # initialize publishers
        #self.drive_pub = rospy.Publisher(rospy.get_param("~drive_topic", "/drive"), AckermannDriveStamped, queue_size=1)
        self.drive_pub = rospy.Publisher(rospy.get_param("~drive_topic", "/vesc/ackermann_cmd_mux/input/navigation"), AckermannDriveStamped, queue_size=1)


    def map_to_robot_frame(self, map_vector):
        '''
        Transform vector [x, y] from map_coords to robot_coords
        '''

        # Create a PointStamped message with the pose in the "map" frame
        map_pose = PointStamped()
        map_pose.header.frame_id = "/map"
        map_pose.point.x = map_vector[0]
        map_pose.point.y = map_vector[1]
        map_pose.point.z = 0.0

        # Transform the pose from "map" frame to "robot" frame
        robot_pose = self.tf_listener.transformPoint("/base_link", map_pose)

        # Access the transformed pose in robot coordinates
        robot_x = robot_pose.point.x
        robot_y = robot_pose.point.y
        robot_z = robot_pose.point.z

        

        # Print the transformed pose in robot coordinates
        # print("Point Position in Robot Coordinates:")
        # print("X: {}".format(robot_x))
        # print("Y: {}".format(robot_y))
        # print("Z: {}".format(robot_z))

        return [robot_x, robot_y]


    def trajectory_callback(self, msg):
        ''' Clears the currently followed trajectory, and loads the new one from the message
        '''
        rospy.loginfo("Receiving new trajectory: " + str(len(msg.poses)) + " points")
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        if self.odometry_initialized == False:
            rospy.loginfo("I haven't estimated my odometry yet")

        # print('\n')
        # for point in np.array(self.trajectory.points):
        #     print(point)
        # print('\n')

        self.points = np.array(self.trajectory.points)
        self.num_segments = len(self.points) - 1
        
        while self.tdex < len(msg.poses) - 1.01: # stop when lookahead point is 99% of the last segment.
            self.pursuit_algorithm()
        self.drive(0,self.steering_angle)
        self.tdex = 0

    def pursuit_algorithm(self):
        '''
        Pursuit algorithm
        
        '''
        self.search_segment_index()

        # use in case of suspicious segment chosen
        # print("P1: " + str(G1) + " P2: " + str(G2) + " Odometry: " + str(Q))
        for index in range(self.segment_index, self.num_segments):
            (G1, G2, Q) = self.obtain_segment(index)
            (success_marker, lookahead_vector) = self.find_lookahead(G1, G2, Q)
            if success_marker:
                # goal_vector is in map coordinates. Need to translate to robot coordinates.
                # print("Lookahead Point" + str(lookahead_vector))
                goal_vector_robot_coords = self.map_to_robot_frame(lookahead_vector)
                self.pure_pursuit(goal_vector_robot_coords)
                break

    def search_segment_index(self):
        '''
        Faster (hopefully) implimentation of finding nearest trajectory to car.
        Returns three points, G1, G2, Q in [x, y]. 
        G1 and G2 are points defining the linesegment closest to the robot.
        Q is the current position of the robot in [x,y]
        '''
        # for testing
        # num_points = len(self.trajectory.points)
        # points = np.array(self.trajectory.points)
        # print(np.shape(points))

        P1 = self.points[0:-1, :]
        p1x = P1[:, 0]
        p1y = P1[:, 1]

        P2 = self.points[1:, :]
        p2x = P2[:, 0]
        p2y = P2[:, 1]
        
        C = np.subtract(p2x, p1x)
        D = np.subtract(p2y, p1y)

        A = -p1x + self.x
        B = -p1y + self.y

        norm = np.multiply(C, C) + np.multiply(D, D)

        u = np.clip(np.divide((np.multiply(A,C) + np.multiply(B,D)), norm), 0, 1)

        nearest_x_on_segment = p1x+ np.multiply(u, C)
        nearest_y_on_segment = p1y+ np.multiply(u, D)

        dx = nearest_x_on_segment - self.x
        dy = nearest_y_on_segment - self.y

        distances_from_curr_pos = np.square(dx) + np.square(dy) # no need to sqrt because care about realitive size

        self.segment_index = np.argmin(distances_from_curr_pos)

    def obtain_segment(self, index):
        P1 = self.points[0:-1, :]
        p1x = P1[:, 0]
        p1y = P1[:, 1]

        P2 = self.points[1:, :]
        p2x = P2[:, 0]
        p2y = P2[:, 1]

        # goal points defining closest line segment
        g1x, g1y = p1x[index], p1y[index]
        g2x, g2y = p2x[index], p2y[index]

        G1 = np.array([g1x, g1y])
        G2 = np.array([g2x, g2y])
        Q = np.array([self.x, self.y])
        return (G1, G2, Q)
    

    def pure_pursuit(self, goal_vector_robot_coords):
        """
        Given goal point [x,y] in robot frame, steers
        the robot along curvature determined by lookahead distance
        """
        relative_x = goal_vector_robot_coords[0]
        relative_y = goal_vector_robot_coords[1]
        L = 0.325 # Wheel base
        eta = math.atan2(relative_y, relative_x)
        
        self.steering_angle = math.atan(2*L*math.sin(eta)/self.lookahead)
        
        self.drive(self.speed, self.steering_angle)


    def find_lookahead(self,G1, G2, Q):
        '''
        Find's point that is one lookahead distance away from car (i.e)
        lies on a circle that is radius = lookahead 
        and intersects nearest line segment.
        
        Returns this goal point as an [x,y] vector in /map frame
        '''
        r = self.lookahead
        V = G2-G1

        a = np.dot(V, V)
        b = 2*np.dot(V, G1-Q)
        c = np.dot(G1, G1) + np.dot(Q, Q) - 2* np.dot(G1, Q) - r**2

        # print('a: ' + str(a) +' b: ' + str(b) +' c: ' + str(c))

        disc = b**2 - 4 * a * c
        if disc < 0:
            print("Discrimenant is negative: Line misses the circle entirely")
            return False, None
        
        sqrt_disc = math.sqrt(disc)
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)

        intersect1 = G1 + t1*V
        intersect2 = G1 + t2*V

        self.tdex = t1 + self.segment_index
        # tdex has to grow in order to progress along path
        # if tdex is smaller than last tdex, command last steering angle
        #print("Current Tdex: " + str(self.tdex) +  ' t: ' + str(t1) + 'index: ' + str(self.segment_index))
        

        if (0 <= t1 <= 1):
            print('Returning t1')
            return True, intersect1
        # elif (0 <= t2 <= 1):
        #     print('Returning t2')
        #     return True, intersect2
        else:
            #print("Line segment misses the circle entirely: But would hit if extended")
            return False, None

    def odometry_callback(self, msg):
        '''
        Updates the estimated odometry of the robot
        '''
        if self.odometry_initialized==False:
            self.odometry_initialized = True

        self.pose = msg.pose.pose
        orient = self.pose.orientation
        self.x = self.pose.position.x
        self.y = self.pose.position.y
        
        quat_tuple = (orient.x, orient.y, orient.z, orient.w)
        roll, pitch, yaw = euler_from_quaternion(quat_tuple)
        self.theta = yaw

    def drive(self, speed = 0, steering_angle = 0):
        """
        Publishes AckermannDriveStamped msg with speed, steering_angle, and steering_angle_velocity
        """
        print("issued drive")
        # create drive object
        ack_drive = AckermannDrive()
        ack_drive.speed = speed
        ack_drive.steering_angle = steering_angle

        #create AckermannDriveStamped object
        ack_stamp = AckermannDriveStamped()
        ack_stamp.drive = ack_drive
        self.drive_pub.publish(ack_stamp)

if __name__=="__main__":
    rospy.init_node("pure_pursuit")
    pf = PurePursuit()
    rospy.spin()

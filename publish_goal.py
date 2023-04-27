#!/usr/bin/env python2

import sys
import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler

if __name__ == "__main__":
    rospy.init_node("publish_goal")
    x = float(sys.argv[1])
    y = float(sys.argv[2])
    theta = float(sys.argv[3])

    pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1, latch=True)
    msg = PoseStamped()
    msg.pose.position.x = x
    msg.pose.position.y = y
    o = quaternion_from_euler(0,0,theta)
    msg.pose.orientation.x = o[0]
    msg.pose.orientation.y = o[1]
    msg.pose.orientation.z = o[2]
    msg.pose.orientation.w = o[3]

    pub.publish(msg)
    rospy.spin()

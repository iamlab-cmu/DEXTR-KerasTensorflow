#!/usr/bin/env python

import cv2
import numpy as np
from matplotlib import pyplot as plt

import rospy

from networks.dextr import DEXTR
from mypath import Path
from helpers import helpers as helpers

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from dextr_msgs.srv import DEXTRRequest,DEXTRRequestResponse
from dextr_msgs.msg import Point2D


def dextr_client():
    rospy.init_node('dextr_client')
    rospy.wait_for_service('dextr')
    print("DEXTR Server Ready.")

    bridge = CvBridge()

    dextr_client = rospy.ServiceProxy('dextr', DEXTRRequest)


    # Read image and click the points
    image = np.array(cv2.imread('/home/klz/unlabeled_food_images/tomato_8_4_ending_grasp_image.png'))
    plt.ion()
    plt.axis('off')
    plt.imshow(image)
    plt.title('Click the four extreme points of the objects\nHit enter when done (do not close the window)')

    results = []

    while 1:
        extreme_points_ori = np.array(plt.ginput(4, timeout=0)).astype(np.int)

        points = []
        for point in extreme_points_ori:
            points.append(Point2D(point[0],point[1]))

        sensor_image = bridge.cv2_to_imgmsg(image, "bgr8")

        resp = dextr_client(sensor_image, points)
        results.append(np.array(bridge.imgmsg_to_cv2(resp.mask) > 0))
        plt.imshow(helpers.overlay_masks(image / 255, results))
        plt.plot(extreme_points_ori[:, 0], extreme_points_ori[:, 1], 'gx')


if __name__ == "__main__":
    dextr_client()
#!/usr/bin/env python

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K

import rospy
import tensorflow as tf

from networks.dextr import DEXTR
from mypath import Path
from helpers import helpers as helpers

from cv_bridge import CvBridge

from dextr_msgs.srv import DEXTRRequest,DEXTRRequestResponse

modelName = 'dextr_pascal-sbd'
pad = 50
thres = 0.8
gpu_id = 0

# Handle input and output args
sess = tf.compat.v1.Session()
K.set_session(sess)

bridge = CvBridge()

with sess.as_default():
    net = DEXTR(nb_classes=1, resnet_layers=101, input_shape=(512, 512), weights=modelName,
                num_input_channels=4, classifier='psp', sigmoid=True)

    def handle_request(req):
        image = req.image
        points = req.points
        cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')

        extreme_points_ori = np.zeros((4,2), dtype=int)
        for i in range(4):
            extreme_points_ori[i,0] = points[i].x
            extreme_points_ori[i,1] = points[i].y

        #  Crop image to the bounding box from the extreme points and resize
        bbox = helpers.get_bbox(cv_image, points=extreme_points_ori, pad=pad, zero_pad=True)
        crop_image = helpers.crop_from_bbox(cv_image, bbox, zero_pad=True)
        resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.float32)

        #  Generate extreme point heat map normalized to image values
        extreme_points = extreme_points_ori - [np.min(extreme_points_ori[:, 0]), np.min(extreme_points_ori[:, 1])] + [pad,
                                                                                                                      pad]
        extreme_points = (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(np.int)
        extreme_heatmap = helpers.make_gt(resize_image, extreme_points, sigma=10)
        extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)

        #  Concatenate inputs and convert to tensor
        input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)

        # Run a forward pass
        pred = net.model.predict(input_dextr[np.newaxis, ...])[0, :, :, 0]
        result = helpers.crop2fullmask(pred, bbox, im_size=cv_image.shape[:2], zero_pad=True, relax=pad) > thres
        
        result = result.astype(np.uint8)  #convert to an unsigned byte

        return DEXTRRequestResponse(bridge.cv2_to_imgmsg(result*255, encoding="passthrough"))

    def dextr_server():
        rospy.init_node('dextr_server')
        s = rospy.Service('dextr', DEXTRRequest, handle_request)
        print("DEXTR Server Ready.")
        rospy.spin()

if __name__ == "__main__":
    dextr_server()
import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException
from rclpy.duration import Duration
from rcl_interfaces.msg import SetParametersResult
from rcl_interfaces.msg import Parameter, ParameterType
import rclpy.qos as qos
import sys
import os
import cv2
import threading
from queue import Queue
#メッセージ型
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from yolact_ros2_msgs.msg import Detections
from yolact_ros2_msgs.msg import Detection
from yolact_ros2_msgs.msg import Box
from yolact_ros2_msgs.msg import Mask
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from collections import defaultdict
import cv2 as cv
from math import atan2, cos, sin, sqrt, pi ,degrees


class Pose_detect(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.qos_profile = qos.QoSProfile(depth=1, reliability=qos.QoSReliabilityPolicy.BEST_EFFORT)
        #subscriber
        self.create_subscription(Detections, '/yolact_ros2/detections', self.detect_callback,qos_profile=self.qos_profile)

    def detect_callback(self, msg):
        try :
            image_height=480
            image_width=640
            image_array = np.zeros((image_height, image_width), dtype=np.uint8)
            for i in range(len(msg.detections)):
                mask_msg=msg.detections[i].mask
                box=msg.detections[i].box
                unpacked_mask = np.unpackbits(mask_msg.mask, count=mask_msg.height*mask_msg.width)
                unpacked_mask = unpacked_mask.reshape((mask_msg.height, mask_msg.width))
                mask=image_array[box.y1:box.y2,box.x1:box.x2]+unpacked_mask

                _, bw = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY) 
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                dst = cv2.erode(bw, kernel)
                ma = np.where(dst == 255,1,0) #unpacked_mask #np.where(unpacked_mask[:,:] != 0)
               
                image_array[box.y1:box.y2,box.x1:box.x2] = ma#sk
               
            _, bin_img = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY)       
            fg_roi=self.pose(bin_img)   
            cv2.imshow("mask",fg_roi)   
            cv2.waitKey(1)
        except :
            pass

    def paint_mask(img_numpy, mask, color):
        h, w, _ = img_numpy.shape
        img_numpy = img_numpy.copy()

        mask = np.tile(mask.reshape(h, w, 1), (1, 1, 3))
        color_np = np.array(color[:3]).reshape(1, 1, 3)
        color_np = np.tile(color_np, (h, w, 1))
        mask_color = mask * color_np

        mask_alpha = 0.3

        # Blend image and mask
        image_crop = img_numpy * mask
        img_numpy *= (1-mask)
        img_numpy += image_crop * (1-mask_alpha) + mask_color * mask_alpha

        return img_numpy
    
    def drawAxis(self,dst, p_, q_, colour, scale):
        p = list(p_)
        q = list(q_)
        angle = atan2(p[1] - q[1], p[0] - q[0]) 
        hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
        q[0] = p[0] - scale * hypotenuse * cos(angle)
        q[1] = p[1] - scale * hypotenuse * sin(angle)
        cv.line(dst, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 5, cv.LINE_AA)
        p[0] = q[0] + 9 * cos(angle + pi / 4)
        p[1] = q[1] + 9 * sin(angle + pi / 4)
        cv.line(dst, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 5, cv.LINE_AA)
        p[0] = q[0] + 9 * cos(angle - pi / 4)
        p[1] = q[1] + 9 * sin(angle - pi / 4)
        cv.line(dst, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 5, cv.LINE_AA)
        
    def getOrientation(self,pts,dst):
        sz = len(pts)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for i in range(data_pts.shape[0]):
            data_pts[i,0] = pts[i,0,0]
            data_pts[i,1] = pts[i,0,1]
        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
        cntr = (int(mean[0,0]), int(mean[0,1]))
        cv.circle(dst, cntr, 3, (255, 0, 255), 2)
        p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
        p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
        #self.drawAxis(dst, cntr, p1, (50, 50, 30), 3)
        self.drawAxis(dst, cntr, p2, (255, 0, 0), 4)
        angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) 
        return angle


    def  pose(self,bw):
        contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        color_fg = (60, 20, 220)
        color_bg = (0, 0, 0)  
        dst = np.where(bw[..., np.newaxis] == 255, color_fg, color_bg).astype(np.uint8)
        for i, c in enumerate(contours):
            area = cv.contourArea(c)
            if area < 1e2 or 1e5 < area:
                continue
            cv.drawContours(bw,contours, i, (0, 0, 255), 2)
            angle=self.getOrientation(c,dst)
            #print(degrees(angle))
        return dst



def main():
    rclpy.init()
    pose_node = Pose_detect('pose_detect_node')

    try:
        rclpy.spin(pose_node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()

if __name__ == '__main__':
    main()
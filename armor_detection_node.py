
import sys
#import KCF
import time
import rospy
import multiprocessing
from ctypes import c_bool
from multiprocessing import Process, Value, Array
from detector import *
import ctypes
import os
import random
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision
import gxipy as gx


class ArmorDetectionNode():
    '''Main Process ROS Node
    '''
    def __init__(self):
        
        rospy.init_node('armor_detection_node')
        
        self._ctrlinfo_pub = rospy.Publisher('/cmd_gimbal_angle', 
                                             GimbalAngle, queue_size=1, 
                                             tcp_nodelay=True)

        self._fricwhl_client = rospy.ServiceProxy("/cmd_fric_wheel",FricWhl)
        self._shoot_client = rospy.ServiceProxy("/cmd_shoot",ShootCmd)
        self._can_ctrl = True
        undet_count = 40
        #num_bullets = 100
        self.camera_matrix = np.array(([1750, 0, 356.3],
                                       [0, 1756, 375.9],
                                       [0, 0, 1.0]), dtype=np.double)
        self.dist_coefs = np.array([0, 0, 0, 0, 0], dtype=np.double)
        object_3d_points = np.array(([-72, -32, 0],                 #xmin ymin
                                     [-58, 32, 0],                  #xmin ymax
                                     [58, -32, 0],                  #xmax ymax 
                                     [72, 32, 0]), dtype=np.double) #xmax ymin

        # main process loop
        while not rospy.is_shutdown():
            angle = self.calcu_angle(boundingbox)
            if self._can_ctrl:
                if angle is not None:
                    self._set_fricwhl(True)
                    self._ctrlinfo_pub.publish(angle[0], angle[1])
                    self._shoot(1,1)
                    rospy.loginfo('pitch '+str(angle[0])+' yaw '+str(angle[1]))
                elif undet_count != 0:
                    self._set_fricwhl(False)
                    self._shoot(0,0)
                    undet_count -= 1
                    self._ctrlinfo_pub.publish(angle[0],angle[1])
                    rospy.loginfo('pitch '+str(angle[0])+' yaw '+str(angle[1]))
                else:
                    self._set_fricwhl(False)
                    self._shoot(0,0)
                    #TODO: define searching mode
                    #searching_mode()
                    rospy.loginfo('searching')
            else:
                rospy.loginfo('decision node needs to control the gimbal')
                
            #rospy.sleep(0.005) # 200Hz frequency


    def _update_ctrlpower(self, ctrlpower_msg):
        '''decision node callback
        '''
        self.can_ctrl = ctrlpower_msg.data

    def _set_fricwhl(can_start):
        '''fricwheel service client
        '''
        rospy.wait_for_service("cmd_fric_wheel")
        try:
            resp = fricwhl_client.call(can_start)
            #rospy.loginfo("Message From fricwheelserver:%s"%resp.received)
        except rospy.ServiceException:
            rospy.logwarn("Service call failed")

    def _shoot(shoot_mode, shoot_number):
        '''shoot service client
        '''
        rospy.wait_for_service("cmd_fric_wheel")
        try:
            resp = shoot_client.call(shoot_mode, shoot_number)
            #rospy.loginfo("Message From shootserver:%s"%resp.received)
        except rospy.ServiceException:
            rospy.logwarn("Service call failed")

    #TODO:High accuracy but slow speed


    def calcu_angle(self, bbox):
        if bbox[2] == 0:
            return None
        else:
            # [ymin xmin ymax xmax]
            box = [bbox[1], bbox[0], bbox[1]+bbox[3], bbox[0]+bbox[2]] 
            object_2d_point = np.array(([box[1],box[0]],[box[1],box[2]],
                                        [box[3],box[2]],[box[3],box[0]]),
                                        dtype=np.double)
            _, _, tvec = cv2.solvePnP(self.object_3d_points, object_2d_point, 
                                      self.camera_matrix, self.dist_coefs, 
                                      cv2.SOLVEPNP_EPNP)
            pitch = float(np.arctan2(tvec[1][0], tvec[2][0])) 
            yaw   = float(np.arctan2(tvec[0][0], tvec[2][0]))
            return [pitch, yaw]




def select_target(box_list, cls_list, score_list, ENEMY_COLOR):
        '''select enemy bbox and get enemy direction
        '''
    for box, cls, score in zip(box_list, cls_list, score_list):
        tmp_armor_score = 0
        if cls == ENEMY_COLOR and score > tmp_armor_score:
            mp_armor_score = score
            armor_box = box
    for box, cls, score in zip(box_list, cls_list, score_list):
        tmp_direc_score = 0
        if cls >=3 and score > tmp_direc_score:
            if box[0] < armor_box[0] and box[2] > armor_box[2]:
                direction = [box, cls]
    return armor_box, direction



if __name__=='__main__':
    #1-blue 2-red
    enemy_color=1
    yolov5_wrapper = YoLov5TRT("build/yolov5s.engine")
    detecting   = Value(c_bool, True)
    initracker  = Value(c_bool, False)
    tracking    = Value(c_bool, False)
    flag        = Value('I', 0)  # num of tracking frames
    direction   = Value('I', 7)  # default direction is tracking
    # ArmorInfo varibles shared by all process
    # xmin,ymin,width,height
    boundingbox = Array('I', [0, 0, 0, 0]) # unsigned int bbox


    device_manager = gx.DeviceManager() 
    dev_num, dev_info_list = device_manager.update_device_list()
    if dev_num == 0:
        sys.exit(1)
    str_sn = dev_info_list[0].get("sn")
    cam = device_manager.open_device_by_sn(str_sn)
    
    cam.stream_on()
    while(1):
        cam.data_stream[0].flush_queue()
        t1 = time.clock()
        raw1_image = cam.data_stream[0].get_image(timeout=10000000)
        rgb1_image = raw1_image.convert("RGB")
        numpy1_image = rgb1_image.get_numpy_array()
        image_raw = cv2.cvtColor(numpy1_image, cv2.COLOR_RGB2BGR)

    #for input_image_path in input_image_paths:
        # create a new thread to do inference
        #thread1 = myThread(yolov5_wrapper.infer, [image_raw])
        #thread1.start()
        #thread1.join()
        box_list, cls_list, score_list=yolov5_wrapper.infer(image_raw)
		box, direc = select_target(box_list, cls_list, score_list, enemy_color)
        boundingbox[:] = [box[1], box[0], box[3]-box[1], box[2]-box[0]] 


        t2 = time.clock()
        print('Done. (%.3fs)' % (t2 - t1))
        if cv2.waitKey(1) == ord('q'): 
            cv2.destroyAllWindows()
            cam.stream_off()
            cam.close_device()
            break
    cam.stream_off()
    # destroy the instance
    yolov5_wrapper.destroy()
    #cam.stream_off()
    #cam.close_device()

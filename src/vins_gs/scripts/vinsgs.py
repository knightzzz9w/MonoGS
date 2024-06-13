#!/home/wkx123/anaconda3/envs/splatam/bin/python
# -*- coding: utf-8 -*-
import rospy
import torch
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud , Image
from visualization_msgs.msg import Marker, MarkerArray
from queue import Queue
from threading import Lock, Condition
from argparse import ArgumentParser
import yaml
from estimator import Estimator
import torch.multiprocessing as mp
import os


last_imu_t = 0.0

image_buf = Queue()
imu_odom_buf = Queue()
point_cloud_buf = Queue()
margin_point_cloud_buf = Queue()
campose_buf = Queue()
keypose_buf = Queue()   #camera的keyposes   
key_point_cloud_buf = Queue()
key_point_cloud_incre_buf = Queue()



m_buf = Lock()
con = Condition(m_buf)

def image_callback(data):
    with m_buf:
        image_buf.put(data)
        con.notify()  

def imu_odom_callback(data):
    with m_buf:
        imu_odom_buf.put(data)
        con.notify()  

def point_cloud_callback(data):
    with m_buf:
        point_cloud_buf.put(data)
        con.notify()  

def margin_cloud_callback(data):
    with m_buf:
        margin_point_cloud_buf.put(data)
        con.notify()  

def campose_callback(data):
    with m_buf:
        campose_buf.put(data)
        con.notify()  

def keypose_callback(data):
    with m_buf:
        keypose_buf.put(data)
        con.notify()
        
def key_point_cloud_callback(data):
    with m_buf:
        key_point_cloud_buf.put(data)
        con.notify()

def key_point_cloud_incre_callback(data):
    with m_buf:
        key_point_cloud_incre_buf.put(data)
        con.notify()

def process(estimator : Estimator) :

    while not rospy.is_shutdown():
        with m_buf:
            measurements = estimator.get_measurements(image_buf , keypose_buf , key_point_cloud_buf)
            
        estimator.project_and_get_colors(measurements)
            

if __name__ == '__main__':
    
    # parser = ArgumentParser(description="Training script parameters")
    # parser.add_argument("--config", type=str)

    # args = parser.parse_args()
    
    with open("/home/wkx/vins_ws/src/vins_gs/config/euroc/euroc.yaml", "r") as yml:
        config = yaml.safe_load(yml)
    
    mp.set_start_method("spawn")
    gs_estimator = Estimator(config)
    

    
    rospy.init_node('gsestimator', anonymous=True)

    #rospy.Subscriber("/vins_estimator/odometry", Odometry, imu_odom_callback)
    #rospy.Subscriber("/vins_estimator/point_cloud", PointCloud, point_cloud_callback)
    #rospy.Subscriber("/vins_estimator/history_cloud", PointCloud, margin_cloud_callback)   #这两个先不用了 我们只处理关键帧的信息
    #rospy.Subscriber("/cam0/image_raw", Image, image_callback , queue_size=200)
    # rospy.Subscriber("/vins_estimator/camera_pose", Odometry, campose_callback , queue_size=100)
    # rospy.Subscriber("/vins_estimator/keyframe_pose", Odometry, keypose_callback , queue_size=100)
    # rospy.Subscriber("/vins_estimator/keyframe_point_incre", PointCloud, key_point_cloud_callback , queue_size=100)
    
    
    rospy.Subscriber("/cam0/image_raw", Image, image_callback , queue_size=30)    #real time rendering
    rospy.Subscriber("/vins_estimator/camera_pose", Odometry, campose_callback , queue_size=10)
    rospy.Subscriber("/vins_estimator/keyframe_pose", Odometry, keypose_callback , queue_size=10)
    rospy.Subscriber("/vins_estimator/keyframe_point_incre", PointCloud, key_point_cloud_callback , queue_size=10)

    # rospy.Subscriber("/cam0/image_raw", Image, image_callback )   #no need for real time rendering
    # rospy.Subscriber("/vins_estimator/camera_pose", Odometry, campose_callback )
    # rospy.Subscriber("/vins_estimator/keyframe_pose", Odometry, keypose_callback )
    # rospy.Subscriber("/vins_estimator/keyframe_point", PointCloud, key_point_cloud_callback )
    # #rospy.Subscriber("/vins_estimator/keyframe_point_incre", PointCloud, key_point_cloud_incre_callback )

    rate = rospy.Rate(10)  # 10Hzs
    process(gs_estimator)
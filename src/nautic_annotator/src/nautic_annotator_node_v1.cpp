// Step 1: Cluster PointClouds
// Step 2: If any N points (default: 10) within the cluster E Restricted FoV ==> Result

#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "std_msgs/Float32MultiArray.h"

#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

using namespace std;

ros::Subscriber sub_raw_pointcloud;


ros::Subscriber sub_cam_fov;
float arr_cam_fov[2] = { 0 };

ros::Publisher pub_vessel_pointcloud;
bool cloud_cb_lock = true;
pcl::PointCloud<pcl::PointXYZI>::Ptr raw_pcl(new pcl::PointCloud<pcl::PointXYZI>);
Eigen::MatrixXf mat;

Eigen::MatrixXf pcl_to_eigen(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& pcl){
    Eigen::MatrixXf eigen;

    int numPoints = pcl->points.size();
    eigen.setZero(numPoints, 4);

    for(int i = 0; i < numPoints; i++) {
        eigen(i, 0) = pcl->points[i].x;
        eigen(i, 1) = pcl->points[i].y;
        eigen(i, 2) = pcl->points[i].z;
        eigen(i, 3) = pcl->points[i].intensity;
    }

    return eigen;
}

void cloud_cb(const sensor_msgs::PointCloud2::ConstPtr& msg){
    if (!cloud_cb_lock){
        pcl::fromROSMsg (*msg, *raw_pcl);
        mat = pcl_to_eigen(raw_pcl);

        pcl::PointCloud<pcl::PointXYZI> vessel_pcl;
        int row_num = mat.rows();
        for (int i=0; i<row_num; i++){
            Eigen::VectorXf rowVector = mat.row(i);

            float x = rowVector(0);
            float y = rowVector(1);
            float z = rowVector(2);
            float I = rowVector(3);

            float theta = atan2(y,x);

            if ((theta < arr_cam_fov[0]) && (theta > arr_cam_fov[1])){
                pcl::PointXYZI point;
                point.x = x;
                point.y = y;
                point.z = z;
                point.intensity = I;
                vessel_pcl.points.push_back(point);
            }
        }

        sensor_msgs::PointCloud2 vessel_msg;
        pcl::toROSMsg(vessel_pcl, vessel_msg);
        vessel_msg.header.frame_id = "os_sensor";
        pub_vessel_pointcloud.publish(vessel_msg);

        cloud_cb_lock = true;
    }
    // else cloud_cb does not run
}

void fov_cb(const std_msgs::Float32MultiArray::ConstPtr& msg){
    if (cloud_cb_lock){
        arr_cam_fov[0] = msg->data[0];
        arr_cam_fov[1] = msg->data[1];
        cloud_cb_lock = false;
    }
    // else fov_cb does not run
}

int main(int argc, char **argv){
    ros::init(argc, argv, "nautic_annotator_node");
    ros::NodeHandle n;

    sub_raw_pointcloud = n.subscribe("/ouster1/points", 1, cloud_cb);
    sub_cam_fov = n.subscribe("/nautic_annotator_node/cam_fov", 1, fov_cb);

    pub_vessel_pointcloud = n.advertise<sensor_msgs::PointCloud2>("/nautic_annotator_node/points", 1);

    ros::spin();
    ros::waitForShutdown();

    return 0;
}
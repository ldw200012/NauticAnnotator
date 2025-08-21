// nautic_annotator_node_v2.cpp
// Full version with saving pts_raw.bin, pts_xyz.bin, and img_det.png to indexed subfolders in [package_path]/data

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int32MultiArray.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <ros/package.h>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <limits>

using PointT  = pcl::PointXYZI;
using CloudT  = pcl::PointCloud<PointT>;
using CloudTP = CloudT::Ptr;

static const char* kOutFrameDefault = "os_sensor";

struct Params {
  float z_min = -std::numeric_limits<float>::infinity();
  float r_min = 0.0f;
  float tail_theta_deg = 0.0f;
  float tail_length = 0.0f;
  bool  voxelize = true;
  float voxel_leaf_x = 0.5f, voxel_leaf_y = 0.5f, voxel_leaf_z = 0.16f;
  bool  denoise = true;
  int   denoise_mean_k = 50;
  float denoise_std_dev_mult = 1.0f;
  float cluster_tolerance = 1.0f;
  int cluster_size_min = 5, cluster_size_max = 25000;
  int n_points_fov_min = 30;
  float closest_cluster_size_min_meters = 2.0f;
  float filter_x_min = -5.0f, filter_x_max = 1.0f;
  float filter_y_min = -1.0f, filter_y_max = 7.0f;
  std::string out_frame = kOutFrameDefault;
} params;

ros::Subscriber sub_cloud, sub_fov, sub_img, sub_bbox, sub_original_img;
ros::Publisher pub_vessel;
sensor_msgs::Image::ConstPtr latest_img_msg;
sensor_msgs::Image::ConstPtr latest_original_img_msg;
std_msgs::Int32MultiArray::ConstPtr latest_bbox_msg;
float arr_cam_fov[2] = {0.0f, 0.0f};
bool cloud_cb_lock = true;
ros::Time process_start, process_end;
int save_index = 0;

inline bool inFoV(float theta, float left, float right) {
  return (theta < left) && (theta > right);
}

inline float planarRange(float x, float y) {
  return std::sqrt(x * x + y * y);
}

// Prefer node-private params (pnh) and fall back to global (nh)
template <typename T>
void loadParam(const ros::NodeHandle& pnh, const ros::NodeHandle& nh,
               const std::string& key, T& out_value) {
  if (!pnh.getParam(key, out_value)) {
    nh.param(key, out_value, out_value);
  }
}

float calculateClusterGeometricSize(const pcl::PointIndices& cluster, const CloudTP& cloud) {
  if (cluster.indices.empty()) return 0.0f;
  
  // Find min and max coordinates
  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float min_z = std::numeric_limits<float>::max();
  float max_x = -std::numeric_limits<float>::max();
  float max_y = -std::numeric_limits<float>::max();
  float max_z = -std::numeric_limits<float>::max();
  
  for (int idx : cluster.indices) {
    const auto& pt = cloud->points[idx];
    min_x = std::min(min_x, pt.x);
    min_y = std::min(min_y, pt.y);
    min_z = std::min(min_z, pt.z);
    max_x = std::max(max_x, pt.x);
    max_y = std::max(max_y, pt.y);
    max_z = std::max(max_z, pt.z);
  }
  
  // Calculate the diagonal length of the bounding box
  float dx = max_x - min_x;
  float dy = max_y - min_y;
  float dz = max_z - min_z;
  
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

inline bool passPointByPointFilters(float x, float y, float z,
  float z_min, float r_min, float tail_len, float tail_half_angle_rad) {
  // Z-axis filter
  if (z < z_min) return false;
  
  // Bounded box filter: erase points within the rectangular region
  if (x >= params.filter_x_min && x <= params.filter_x_max && 
      y >= params.filter_y_min && y <= params.filter_y_max) return false;
  
  // Radial distance filter
  const float r = planarRange(x, y);
  if (r < r_min) return false;
  
  // Tail angle filter
  if (r < tail_len) {
    float th = std::atan2(y, x);
    float lo = -tail_half_angle_rad - static_cast<float>(M_PI) * 0.5f;
    float hi =  tail_half_angle_rad - static_cast<float>(M_PI) * 0.5f;
    if (th > lo && th < hi) return false;
  }
  
  return true;
}

void img_cb(const sensor_msgs::Image::ConstPtr& msg) {
  latest_img_msg = msg;
}

void bbox_cb(const std_msgs::Int32MultiArray::ConstPtr& msg) {
  latest_bbox_msg = msg;
}

void original_img_cb(const sensor_msgs::Image::ConstPtr& msg) {
  latest_original_img_msg = msg;
}

void savePointCloudBin(const CloudTP& cloud, const std::string& filepath) {
  std::ofstream ofs(filepath, std::ios::binary);
  for (const auto& p : cloud->points) {
    ofs.write(reinterpret_cast<const char*>(&p.x), sizeof(float));
    ofs.write(reinterpret_cast<const char*>(&p.y), sizeof(float));
    ofs.write(reinterpret_cast<const char*>(&p.z), sizeof(float));
    ofs.write(reinterpret_cast<const char*>(&p.intensity), sizeof(float));
  }
}

void fov_cb(const std_msgs::Float32MultiArray::ConstPtr& msg) {
  if (msg->data.size() >= 2) {
    arr_cam_fov[0] = static_cast<float>(msg->data[0]);
    arr_cam_fov[1] = static_cast<float>(msg->data[1]);
    cloud_cb_lock  = false;
  }
}

void cloud_cb(const sensor_msgs::PointCloud2::ConstPtr& msg) {
  if (cloud_cb_lock) return;
  process_start = ros::Time::now();

  CloudTP raw(new CloudT);
  pcl::fromROSMsg(*msg, *raw);

  CloudTP filtered(new CloudT);
  filtered->points.reserve(raw->points.size());
  float tail_half_angle_rad = params.tail_theta_deg * static_cast<float>(M_PI) / 180.0f;
  for (const auto& p : raw->points) {
    if (passPointByPointFilters(p.x, p.y, p.z, params.z_min, params.r_min, params.tail_length, tail_half_angle_rad)) {
      filtered->points.push_back(p);
    }
  }
  filtered->width = filtered->points.size();
  filtered->height = 1;
  filtered->is_dense = false;

  CloudTP cloud_ds = params.voxelize ? CloudTP(new CloudT) : filtered;
  if (params.voxelize) {
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(filtered);
    vg.setLeafSize(params.voxel_leaf_x, params.voxel_leaf_y, params.voxel_leaf_z);
    vg.filter(*cloud_ds);
  }

  // Apply denoise filter if enabled
  if (params.denoise && !cloud_ds->empty()) {
    CloudTP denoised(new CloudT);
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(cloud_ds);
    sor.setMeanK(params.denoise_mean_k);
    sor.setStddevMulThresh(params.denoise_std_dev_mult);
    sor.filter(*denoised);
    cloud_ds = denoised;
  }

  if (cloud_ds->empty()) {
    cloud_cb_lock = true;
    return;
  }

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
  tree->setInputCloud(cloud_ds);
  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance(params.cluster_tolerance);
  ec.setMinClusterSize(params.cluster_size_min);
  ec.setMaxClusterSize(params.cluster_size_max);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud_ds);
  ec.extract(cluster_indices);

  float closest_range = std::numeric_limits<float>::infinity();
  const pcl::PointIndices* best_cluster = nullptr;
  for (const auto& idxs : cluster_indices) {
    int n_in_fov = 0;
    float min_range = std::numeric_limits<float>::infinity();
    for (int id : idxs.indices) {
      const auto& pt = cloud_ds->points[id];
      float theta = std::atan2(pt.y, pt.x);
      if (inFoV(theta, arr_cam_fov[0], arr_cam_fov[1])) {
        ++n_in_fov;
        float r = planarRange(pt.x, pt.y);
        if (r < min_range) min_range = r;
      }
    }
    float cluster_size = calculateClusterGeometricSize(idxs, cloud_ds);
    if (n_in_fov >= params.n_points_fov_min && min_range < closest_range && 
        cluster_size >= params.closest_cluster_size_min_meters) {
      closest_range = min_range;
      best_cluster = &idxs;
    }
  }

  CloudTP out_cloud(new CloudT);
  if (best_cluster) {
    for (int id : best_cluster->indices) {
      out_cloud->points.push_back(cloud_ds->points[id]);
    }
    out_cloud->width = out_cloud->points.size();
    out_cloud->height = 1;
    out_cloud->is_dense = false;
  }

  sensor_msgs::PointCloud2 out_msg;
  pcl::toROSMsg(*out_cloud, out_msg);
  out_msg.header.frame_id = params.out_frame.empty() ? msg->header.frame_id : params.out_frame;
  out_msg.header.stamp = msg->header.stamp;
  pub_vessel.publish(out_msg);

  // Saving logic
  if (best_cluster) {
    std::string base_path = ros::package::getPath("nautic_annotator") + "/data/object/" + std::to_string(save_index++);
    std::filesystem::create_directories(base_path);
    savePointCloudBin(raw, base_path + "/pts_raw.bin");
    savePointCloudBin(out_cloud, base_path + "/pts_xyz.bin");
    if (latest_img_msg) {
      try {
        cv::Mat img = cv_bridge::toCvShare(latest_img_msg, "bgr8")->image;
        cv::imwrite(base_path + "/img_det.png", img);
        
        // Save cropped image if bounding box is available
        if (latest_bbox_msg && latest_bbox_msg->data.size() >= 4 && latest_original_img_msg) {
          int x1 = latest_bbox_msg->data[0];
          int y1 = latest_bbox_msg->data[1];
          int x2 = latest_bbox_msg->data[2];
          int y2 = latest_bbox_msg->data[3];
          
          try {
            cv::Mat original_img = cv_bridge::toCvShare(latest_original_img_msg, "bgr8")->image;
            
            // Ensure coordinates are within image bounds
            x1 = std::max(0, std::min(x1, original_img.cols - 1));
            y1 = std::max(0, std::min(y1, original_img.rows - 1));
            x2 = std::max(0, std::min(x2, original_img.cols - 1));
            y2 = std::max(0, std::min(y2, original_img.rows - 1));
            
            if (x2 > x1 && y2 > y1) {
              cv::Mat cropped_img = original_img(cv::Rect(x1, y1, x2 - x1, y2 - y1));
              cv::imwrite(base_path + "/img_cropped.png", cropped_img);
            }
          } catch (cv_bridge::Exception& e) {
            ROS_WARN("cv_bridge exception for original image: %s", e.what());
          }
        }
      } catch (cv_bridge::Exception& e) {
        ROS_WARN("cv_bridge exception: %s", e.what());
      }
    }
  }

  ros::param::set("node2_processtime", std::round((ros::Time::now() - process_start).toSec() * 100000.0) / 100.0);
  cloud_cb_lock = true;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "nautic_annotator_node");
  ros::NodeHandle nh;

  nh.param("point_by_point_z",            params.z_min,             params.z_min);
  nh.param("point_by_point_r",            params.r_min,             params.r_min);
  nh.param("point_by_point_tail_theta",   params.tail_theta_deg,    params.tail_theta_deg);
  nh.param("point_by_point_tail_length",  params.tail_length,       params.tail_length);
  nh.param("voxelize",                    params.voxelize,          params.voxelize);
  nh.param("voxel_leaf_x",                params.voxel_leaf_x,      params.voxel_leaf_x);
  nh.param("voxel_leaf_y",                params.voxel_leaf_y,      params.voxel_leaf_y);
  nh.param("voxel_leaf_z",                params.voxel_leaf_z,      params.voxel_leaf_z);
  nh.param("denoise",                     params.denoise,          params.denoise);
  nh.param("denoise_mean_k",              params.denoise_mean_k,   params.denoise_mean_k);
  nh.param("denoise_std_dev_mult",        params.denoise_std_dev_mult, params.denoise_std_dev_mult);
  nh.param("clustering_tolerance",        params.cluster_tolerance, params.cluster_tolerance);
  nh.param("clustering_size_min",         params.cluster_size_min,  params.cluster_size_min);
  nh.param("clustering_size_max",         params.cluster_size_max,  params.cluster_size_max);
  nh.param("n_points_fov_min",            params.n_points_fov_min,  params.n_points_fov_min);
  nh.param("closest_cluster_size_min_meters", params.closest_cluster_size_min_meters, params.closest_cluster_size_min_meters);
  nh.param("filter_x_min",                params.filter_x_min,      params.filter_x_min);
  nh.param("filter_x_max",                params.filter_x_max,      params.filter_x_max);
  nh.param("filter_y_min",                params.filter_y_min,      params.filter_y_min);
  nh.param("filter_y_max",                params.filter_y_max,      params.filter_y_max);
  nh.param("out_frame",                   params.out_frame,         std::string(kOutFrameDefault));

  sub_cloud = nh.subscribe("/ouster2/points", 1, cloud_cb);
  sub_fov   = nh.subscribe("/nautic_annotator_node/cam_fov", 1, fov_cb);
  sub_img   = nh.subscribe("/nautic_annotator_node/detection_img", 1, img_cb);
  sub_bbox  = nh.subscribe("/nautic_annotator_node/bbox", 1, bbox_cb);
  sub_original_img = nh.subscribe("/nautic_annotator_node/original_img", 1, original_img_cb);
  pub_vessel= nh.advertise<sensor_msgs::PointCloud2>("/nautic_annotator_node/points", 1);

  ros::spin();
  return 0;
}
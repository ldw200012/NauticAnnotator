#!/usr/bin/python3

import os
import tf
import time
import math
import rospkg
import rospy
import numpy as np
import importlib
import torch
import cv2

from std_msgs.msg import Float32MultiArray, Int32MultiArray
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from damo.base_models.core.ops import RepConv
from damo.detectors.detector import build_local_model
from damo.utils.demo_utils import transform_img
from damo.structures.image_list import ImageList

class DetectionNode():
    def __init__(self):
        self.dot_cnt = 0
        self.dot_cnt_max = 5
        self.use_cuda = True
        self.device = 'cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu'

        # DAMO-YOLO model settings
        self.cam_pad = 0.0349066
        self.cam_bias = 0.0174533
        self.bbox_margin = 0
        self.boat_close_enough_ratio = 0.1
        self.conf = 0.5
        self.infer_size = [320, 320]
        self.backbone = "damoyolo_tinynasL35_M"

        # Intrinsic calibration (example values — update for your camera)
        self.fx0, self.fy0 = 646.7476, 645.9632
        self.cx0, self.cy0 = 646.2584, 362.9647
        self.k1_0, self.k2_0, self.p1_0, self.p2_0, self.k3_0 = 0, 0, 0, 0, 0
        self.fx1, self.fy1 = 644.8154, 644.0333
        self.ppx1, self.ppy1 = 646.2584, 362.9647
        self.W0, self.H0 = 1280, 720
        self.W1, self.H1 = 1280, 720

        sx, sy = self.W0 / self.W1, self.H0 / self.H1
        self.K0 = np.array([[self.fx0, 0, self.cx0], [0, self.fy0, self.cy0], [0, 0, 1]], dtype=np.float64)
        self.D0 = np.array([self.k1_0, self.k2_0, self.p1_0, self.p2_0, self.k3_0], dtype=np.float64)
        self.K_new = np.array([[sx*self.fx1, 0, sx*self.ppx1],
                               [0, sy*self.fy1, sy*self.ppy1],
                               [0, 0, 1]], dtype=np.float64)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.K0, self.D0, np.eye(3), self.K_new, (self.W0, self.H0), cv2.CV_16SC2)
        self.cam_fov = 2 * math.atan(self.W0 / (2 * self.K_new[0, 0]))

        # Load DAMO-YOLO model
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('nautic_annotator')
        self.ckpt_path = os.path.join(package_path, 'scripts/weights/' + self.backbone + ".pth")
        self.config = importlib.import_module("damo_configs." + self.backbone).Config()
        self.class_names = getattr(self.config.dataset, "class_names", [str(i) for i in range(self.config.model.head.num_classes)])
        self.config.dataset.size_divisibility = 0
        self.model = self._build_engine(self.config)

        # ROS setup
        self.bridge = CvBridge()
        self.img_sub = rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, self.img_cb, queue_size=1, callback_args=0)
        self.fov_pub = rospy.Publisher('/nautic_annotator_node/cam_fov', Float32MultiArray, queue_size=1)
        self.img_pub = rospy.Publisher('/nautic_annotator_node/detection_img', Image, queue_size=1)
        self.marker_pub = rospy.Publisher('/nautic_annotator_node/cam_fov_markers', MarkerArray, queue_size=1)
        self.bbox_pub = rospy.Publisher('/nautic_annotator_node/bbox', Int32MultiArray, queue_size=1)

    def _pad_image(self, img, target_size):
        n, c, h, w = img.shape
        assert h <= target_size[0] and w <= target_size[1]
        padded = torch.zeros((n, c, target_size[0], target_size[1]))
        padded[:, :c, :h, :w].copy_(img)
        return ImageList(padded, [img.shape[-2:]], [padded.shape[-2:]])

    def _build_engine(self, config):
        model = build_local_model(config, self.device)
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        model.load_state_dict(ckpt['model'], strict=True)
        for layer in model.modules():
            if isinstance(layer, RepConv):
                layer.switch_to_deploy()
        model.eval()
        dummy_input = torch.ones((1, 3, *self.infer_size)).to(self.device)
        _ = model(dummy_input)
        return model

    def preprocess(self, origin_img):
        img = transform_img(origin_img, 0, **self.config.test.augment.transform, infer_size=self.infer_size)
        oh, ow, _ = origin_img.shape
        img = self._pad_image(img.tensors, self.infer_size).to(self.device)
        return img, (ow, oh)

    def postprocess(self, preds, origin_shape=None):
        output = preds[0].resize(origin_shape)
        return output.bbox, output.get_field('scores'), output.get_field('labels')

    def forward(self, origin_image):
        image, shape = self.preprocess(origin_image)
        with torch.no_grad():
            output = self.model(image)
        return self.postprocess(output, origin_shape=shape)

    def boat_detection_availability(self, bboxes, scores, cls_inds):
        boat_bboxes = [(box, score) for box, score, cls in zip(bboxes, scores, cls_inds)
                       if int(cls) == 8 and score > self.conf]
        if boat_bboxes:
            return max(boat_bboxes, key=lambda x: x[0][3])
        return False

    def draw_boxes(self, img, result):
        if result:
            box, score = result
            box = box.cpu().numpy().astype(int)
            cls_id = 8  # Boat
            if score >= self.conf:
                color = (0, 255, 0)
                label = f"{self.class_names[cls_id]}: {score:.2f}"
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(img, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img

    def publish_fov_marker_array(self, left_theta, right_theta):
        marker_array = MarkerArray()

        # Delete all markers if both angles are zero
        if abs(left_theta) < 1e-6 and abs(right_theta) < 1e-6:
            delete_marker = Marker()
            delete_marker.action = Marker.DELETEALL
            marker_array.markers.append(delete_marker)
            self.marker_pub.publish(marker_array)
            return

        origin = [-0.035, 0.121, 0.224]
        theta_deg_left = math.degrees(left_theta)
        theta_deg_right = math.degrees(right_theta)

        for i, deg in enumerate(np.arange(theta_deg_right, theta_deg_left + 0.1, 1.0)):
            yaw = math.radians(deg)

            marker = Marker()
            marker.header.frame_id = "os_sensor"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "fov_directions"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            marker.pose.position.x = origin[0]
            marker.pose.position.y = origin[1]
            marker.pose.position.z = origin[2]

            q = tf.transformations.quaternion_from_euler(0, 0, yaw)
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]

            marker.scale.x = 1000
            marker.scale.y = 0.03
            marker.scale.z = 0.03

            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker.lifetime = rospy.Duration(0.2)
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    def img_cb(self, data, args):
        start = rospy.Time.now().to_sec() * 1000.0

        img = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        rectified = cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

        bboxes, scores, cls_inds = self.forward(rectified)
        cam_left_theta = 0.0
        cam_right_theta = -0.0

        if len(bboxes) > 0:
            result = self.boat_detection_availability(bboxes, scores, cls_inds)
            if result:
                box, _ = result
                img_h, img_w, _ = rectified.shape
                left_px, right_px = int(box[0]), int(box[2])
                cam_left_theta = (img_w/2 - left_px) * self.cam_fov / img_w + self.cam_pad + self.cam_bias
                cam_right_theta = (img_w/2 - right_px) * self.cam_fov / img_w - self.cam_pad - self.cam_bias
                
                # Publish bounding box coordinates
                bbox_msg = Int32MultiArray()
                bbox_msg.data = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                self.bbox_pub.publish(bbox_msg)

        self.fov_pub.publish(Float32MultiArray(data=[cam_left_theta, cam_right_theta]))
        self.publish_fov_marker_array(cam_left_theta, cam_right_theta)

        img_with_boxes = self.draw_boxes(rectified.copy(), result)

        try:
            self.img_pub.publish(self.bridge.cv2_to_imgmsg(img_with_boxes, "bgr8"))
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

        end = rospy.Time.now().to_sec() * 1000.0
        rospy.set_param("node1_processtime", round(end - start, 2))

if __name__ == '__main__':
    rospy.init_node('detection')
    try:
        DetectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

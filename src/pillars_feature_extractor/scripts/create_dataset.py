#!/usr/bin/python3

import os
import rospy
import rospkg
import numpy as np
import torch
import cv2

from model_exp1 import PointPillars
# PointLayer
# ==> torch.Size([num_points_per_pillar, 32, 4])    : Pseudo-image

# from model_exp2 import PointPillars
# PointLayer + PointEncoder
# ==> torch.Size([1, 64, 496, 432])                 : Feature Extracted

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

class PointPillarNode():

    def __init__(self, obj_class_idx):
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('pillars_feature_extractor')  # replace with your package name
        self.ckpt = os.path.join(package_path, "scripts/weights/epoch_160.pth")

        self.obj_image_cnt = 0
        self.obj_class_idx = obj_class_idx
        self.data_save_dir = os.path.join(package_path, "dataset/"+self.obj_class_idx)
        if not os.path.exists(self.data_save_dir):
            os.makedirs(self.data_save_dir, exist_ok=True)
        
        self.pcl_sub = rospy.Subscriber("/vps_node/points", PointCloud2, callback = self.pcl_cb, queue_size = 1)

    def point_range_filter(self, pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
        '''
        data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
        point_range: [x1, y1, z1, x2, y2, z2]
        '''
        flag_x_low = pts[:, 0] > point_range[0]
        flag_y_low = pts[:, 1] > point_range[1]
        flag_z_low = pts[:, 2] > point_range[2]
        flag_x_high = pts[:, 0] < point_range[3]
        flag_y_high = pts[:, 1] < point_range[4]
        flag_z_high = pts[:, 2] < point_range[5]
        keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
        pts = pts[keep_mask]
        return pts 

    def pcl_cb(self, msg):

        CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2
        }
        LABEL2CLASSES = {v:k for k, v in CLASSES.items()}
        pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

        model = PointPillars(nclasses=len(CLASSES)).cuda()
        model.load_state_dict(torch.load(self.ckpt))
        
        point_list = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))
        pc = np.array(list(point_list))

        if (np.shape(pc)[0] == 0):
            pass
        else:
            pc = self.point_range_filter(pc)
            pc_torch = torch.from_numpy(pc).float()

            model.eval()
            with torch.no_grad():
                pc_torch = pc_torch.cuda()
                result_filter = model(batched_pts=[pc_torch], mode='test')

            ###### HERE I WANT TO SEE THE SHAPE OF result_filter #####
            print("Shape of result_filter:")
            if isinstance(result_filter, torch.Tensor):
                print(result_filter.shape)
            elif isinstance(result_filter, dict):
                for key, value in result_filter.items():
                    if isinstance(value, torch.Tensor):
                        print(f"{key}: {value.shape}")
                    else:
                        print(f"{key}: {type(value)}")
            elif isinstance(result_filter, list):
                for i, item in enumerate(result_filter):
                    if isinstance(item, torch.Tensor):
                        print(f"Item {i}: {item.shape}")
                    else:
                        print(f"Item {i}: {type(item)}")
            else:
                print("Unknown type of result_filter")

            ### TEST TRIAL: Save the Pseudo-Image
            image = result_filter.cpu().detach().numpy()  # Convert to numpy
            image = (image - image.min()) / (image.max() - image.min()) * 255  # Normalize
            image = image.astype(np.uint8)  # Convert to uint8
            
            # HxWxC for image saving
            filepath = os.path.join(self.data_save_dir, self.obj_class_idx+"_c1_"+str(self.obj_image_cnt).zfill(8)+".png")
            print(filepath)
            cv2.imwrite(filepath, image)  # OpenCV expects BGR format

            self.obj_image_cnt += 1

if __name__ == '__main__':
    rospy.init_node('create_dataset')
    
    try:
        pillar_extraction_node = PointPillarNode("0008")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
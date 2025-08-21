#!/usr/bin/python3

import os
import rospkg
import numpy as np
import matplotlib.pyplot as plt
import cv2

def read_bin(filepath):
    """Read .bin file containing float32 XYZ[I]"""
    return np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)

def visualize_folder(folder_path, save_folder, idx):
    raw_path = os.path.join(folder_path, 'pts_raw.bin')
    cluster_path = os.path.join(folder_path, 'pts_xyz.bin')
    img_path = os.path.join(folder_path, 'img_det.png')
    cropped_path = os.path.join(folder_path, 'img_cropped.png')

    if not all(os.path.exists(p) for p in [raw_path, cluster_path, img_path]):
        print(f"Skipping {folder_path}: missing one or more files.")
        return

    pts_raw = read_bin(raw_path)
    pts_cluster = read_bin(cluster_path)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping {folder_path}: couldn't load image.")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load cropped image if available
    cropped_img = None
    if os.path.exists(cropped_path):
        cropped_img = cv2.imread(cropped_path)
        if cropped_img is not None:
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2])

    # Full detection image subplot (top row)
    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(img_rgb)
    ax_img.set_title("Detection Image")
    ax_img.axis('off')

    # Cropped image subplot (second row)
    if cropped_img is not None:
        ax_cropped = fig.add_subplot(gs[1, :])
        ax_cropped.imshow(cropped_img)
        ax_cropped.set_title("Cropped Boat Image")
        ax_cropped.axis('off')
    else:
        ax_cropped = fig.add_subplot(gs[1, :])
        ax_cropped.text(0.5, 0.5, 'No cropped image available', ha='center', va='center', transform=ax_cropped.transAxes)
        ax_cropped.set_title("Cropped Boat Image")
        ax_cropped.axis('off')

    # Full raw pointcloud
    ax_raw = fig.add_subplot(gs[2, 0])
    ax_raw.scatter(-pts_raw[:, 1], pts_raw[:, 0], c='gray', s=0.5)
    ax_raw.set_title("Full Pointcloud (Top-down)")
    ax_raw.set_xlabel('-Y [m]')
    ax_raw.set_ylabel('X [m]')
    ax_raw.set_xlim(-100, 100)
    ax_raw.set_ylim(-10, 100)

    # Best cluster overlay
    ax_overlay = fig.add_subplot(gs[2, 1])
    ax_overlay.scatter(-pts_raw[:, 1], pts_raw[:, 0], c='lightgray', s=0.5, label='Raw')
    ax_overlay.scatter(-pts_cluster[:, 1], pts_cluster[:, 0], c='red', s=1.0, label='Best Cluster')
    ax_overlay.set_title("Cluster Overlay (Top-down)")
    ax_overlay.set_xlabel('-Y [m]')
    ax_overlay.set_ylabel('X [m]')
    ax_overlay.set_xlim(-100, 100)
    ax_overlay.set_ylim(-10, 100)
    ax_overlay.legend()

    plt.tight_layout()

    # Save image to viewer directory
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f'viewer_{idx}.png')
    plt.savefig(save_path, dpi=200)
    print(f"Saved: {save_path}")

    plt.close()

def main():
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('nautic_annotator')
    base_folder = os.path.join(pkg_path, 'data')
    obj_folder = os.path.join(base_folder, 'object')
    save_folder = os.path.join(base_folder, 'viewer')

    if not os.path.exists(base_folder):
        print("No 'data/' folder found.")
        return

    subfolders = sorted([
        os.path.join(obj_folder, d) for d in os.listdir(obj_folder)
        if os.path.isdir(os.path.join(obj_folder, d))
    ])

    for idx, folder in enumerate(subfolders):
        visualize_folder(folder, save_folder, idx)

if __name__ == '__main__':
    main()

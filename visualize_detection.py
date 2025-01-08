import os
import numpy as np
import torch
import open3d as o3d
from PIL import Image
from torchvision import transforms as T
from detection import DetectionHead, load_multimodal_model, get_embeddings

# Define functions for loading data
def load_kitti_lidar_data(file_path):
    point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud

def load_calibration(calib_file):
    calib_data = {}
    with open(calib_file, 'r') as f:
        for line in f:
            if line.strip():
                key, *values = line.split()
                calib_data[key.rstrip(':')] = np.array(values, dtype=np.float32)
    return calib_data

def get_velo_to_cam_transform(calib_data):
    Tr_velo_to_cam = calib_data['Tr_velo_to_cam'].reshape(3, 4)
    Tr_velo_to_cam = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
    return np.linalg.inv(Tr_velo_to_cam)

# Load and preprocess image
def load_and_preprocess_image(image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    return image

# Visualize 2D bounding boxes
def visualize_image_with_boxes(image_path, bbox_2d):
    import cv2
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # Scale the predicted boxes to image dimensions
    for box in bbox_2d:
        x1, y1, x2, y2 = np.clip([box[0] * w, box[1] * h, box[2] * w, box[3] * h], 0, w)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.imshow("2D Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Create 3D bounding box
def create_bounding_box(h, w, l, x, y, z, ry, Tr_cam_to_velo):
    # Define corners of the 3D box
    corners = np.array([
        [l/2, h/2, w/2], [l/2, h/2, -w/2], [-l/2, h/2, -w/2], [-l/2, h/2, w/2],
        [l/2, -h/2, w/2], [l/2, -h/2, -w/2], [-l/2, -h/2, -w/2], [-l/2, -h/2, w/2]
    ]).T

    # Rotate and translate the corners
    rot_mat = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    corners = rot_mat @ corners + np.array([[x], [y], [z]])
    corners = np.vstack((corners, np.ones((1, 8))))  # Add homogeneous coordinate
    corners = Tr_cam_to_velo @ corners
    return corners[:3, :].T

# Visualize 3D bounding boxes with intensity-based colored point cloud
def visualize_point_cloud_with_boxes(points, boxes, Tr_cam_to_velo):
    xyz = points[:, :3]
    intensity = points[:, 3] if points.shape[1] == 4 else np.ones(points.shape[0])

    # Map intensity to RGB
    intensity = np.clip(intensity / np.max(intensity), 0, 1)
    colors = np.zeros((xyz.shape[0], 3))
    colors[:, 0] = intensity  # Red channel
    colors[:, 1] = intensity  # Green channel
    colors[:, 2] = 1.0 - intensity  # Blue channel

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries = [pcd]
    for box in boxes:
        corners = create_bounding_box(*box, Tr_cam_to_velo)
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([0, 0, 255])
        geometries.append(line_set)

    o3d.visualization.draw_geometries(geometries, window_name="LiDAR 3D Boxes with Color", width=800, height=600)

# Generate predictions
def generate_predictions(model, multimodal_model, image, point_cloud):
    multimodal_model.eval()
    model.eval()
    with torch.no_grad():
        if point_cloud.dim() == 2:
            point_cloud = point_cloud.unsqueeze(0)
        combined_emb = get_embeddings(multimodal_model, image, point_cloud)
        class_preds, bbox_2d_preds, bbox_3d_preds = model(combined_emb)
    return class_preds[0], bbox_2d_preds[0], bbox_3d_preds[0]

# Main
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = "C:/Users/abuba/Desktop/ENPM703/Final project/project_v2/Transformer_for_3d_obj_detection_in_LidarPC/kitti_3d_object_detection/training/image_2/003087.png"
    lidar_path = "C:/Users/abuba/Desktop/ENPM703/Final project/project_v2/Transformer_for_3d_obj_detection_in_LidarPC/kitti_3d_object_detection/training/velodyne/003087.bin"
    calib_path = "C:/Users/abuba/Desktop/ENPM703/Final project/project_v2/Transformer_for_3d_obj_detection_in_LidarPC/kitti_3d_object_detection/training/calib/003087.txt"
    detection_head_path = "detection_head_lightning.pth"
    best_model_path = "best_model_lightning.pth"

    multimodal_model = load_multimodal_model(best_model_path, device).to(device)
    detection_head = DetectionHead(input_dim=1024, num_classes=3).to(device)
    detection_head.load_state_dict(torch.load(detection_head_path, map_location=device))

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = load_and_preprocess_image(image_path, transform, device)

    point_cloud = load_kitti_lidar_data(lidar_path)
    calib_data = load_calibration(calib_path)
    Tr_cam_to_velo = get_velo_to_cam_transform(calib_data)

    class_preds, bbox_2d_preds, bbox_3d_preds = generate_predictions(
        detection_head, multimodal_model, image, torch.from_numpy(point_cloud[:, :3]).float().to(device)
    )
    print("Predicted Class:", torch.argmax(class_preds).item())

    visualize_image_with_boxes(image_path, [bbox_2d_preds.cpu().numpy()])
    visualize_point_cloud_with_boxes(point_cloud, [bbox_3d_preds.cpu().numpy()], Tr_cam_to_velo)

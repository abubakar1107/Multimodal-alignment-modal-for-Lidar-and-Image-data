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
    """
    Generate the 8 corners of a 3D bounding box in the LiDAR frame.
    h: height, w: width, l: length
    x, y, z: box center in the camera frame
    ry: rotation angle around the y-axis (yaw)
    Tr_cam_to_velo: transformation matrix from camera to LiDAR frame
    """
    # Define 8 corners of the 3D box in its local (centered) coordinate frame
    corners = np.array([
        [ l/2,  h/2,  w/2],
        [ l/2,  h/2, -w/2],
        [-l/2,  h/2, -w/2],
        [-l/2,  h/2,  w/2],
        [ l/2, -h/2,  w/2],
        [ l/2, -h/2, -w/2],
        [-l/2, -h/2, -w/2],
        [-l/2, -h/2,  w/2]
    ]).T  # Transpose to make shape (3, 8)

    # Rotation matrix around the y-axis (yaw rotation)
    rot_mat = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    # Rotate and translate to the box center
    corners = rot_mat @ corners + np.array([[x], [y], [z]])  # Apply rotation and center position

    # Transform from camera frame to LiDAR frame
    corners = np.vstack((corners, np.ones((1, 8))))  # Add homogeneous coordinates
    corners = Tr_cam_to_velo @ corners  # Apply transformation matrix
    return corners[:3, :].T  # Return as (8, 3) for 3D visualization


def visualize_point_cloud_with_boxes(points, boxes):
    # Convert points to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    print("Boxes:", boxes)
    # Create bounding boxes and add them to the visualization
    geometries = [pcd]
    for (h, w, l, x, y, z, ry) in boxes:
        corners = create_bounding_box(h, w, l, x, y, z, ry, Tr_cam_to_velo)
        print("Bounding Box Corners:\n", corners)
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Top face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Bottom face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Side edges
        ]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # Set color of the bounding box to black
        line_set.paint_uniform_color([0, 0, 0])  # Black color for visibility on white background
        geometries.append(line_set)
    
    # Visualize with thicker lines by setting `line_width`
    o3d.visualization.draw_geometries(geometries, width=800, height=600)

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
    image_path = "kitti_3d_object_detection/training/image_2/000015.png"
    lidar_path = "kitti_3d_object_detection/training/velodyne/000015.bin"
    calib_path = "kitti_3d_object_detection/training/calib/000015.txt"
    detection_head_path = "detection_head.pth"
    best_model_path = "best_model.pth"

    multimodal_model = load_multimodal_model(best_model_path, device).to(device)
    detection_head = DetectionHead(input_dim=1536, num_classes=3).to(device)
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

    bbox_3d_preds_np = bbox_3d_preds.cpu().numpy()
    visualize_point_cloud_with_boxes(
        point_cloud,
        [(bbox_3d_preds_np[0], bbox_3d_preds_np[1], bbox_3d_preds_np[2],  # h, w, l
          bbox_3d_preds_np[3], bbox_3d_preds_np[4], bbox_3d_preds_np[5],  # x, y, z
          bbox_3d_preds_np[6])],  # ry
    )



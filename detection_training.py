import os
import glob
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from detection import DetectionHead, load_multimodal_model, get_embeddings

# Define a new KITTIDataset class
class KITTILabelDataset(Dataset):
    def __init__(self, root_dir, transform=None, subset_fraction=1.0):
        self.image_dir = os.path.join(root_dir, "image_2")
        self.velodyne_dir = os.path.join(root_dir, "velodyne")
        self.label_dir = os.path.join(root_dir, "label_2")
        self.transform = transform

        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.png")))

        if subset_fraction < 1.0:
            subset_size = int(len(self.image_files) * subset_fraction)
            self.image_files = random.sample(self.image_files, subset_size)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        image = Image.open(img_path).convert('RGB')
        img_width, img_height = image.size  # Get image dimensions for normalization

        if self.transform:
            image = self.transform(image)

        # Load point cloud
        lidar_path = os.path.join(self.velodyne_dir, base_name + ".bin")
        points = torch.from_numpy(np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4))[:, :3]

        # Load label file
        label_path = os.path.join(self.label_dir, base_name + ".txt")
        labels, bbox_2d, bbox_3d = [], [], []

        with open(label_path, "r") as f:
            for line in f:
                data = line.strip().split()
                object_type = data[0]

                # Process only 3 classes: Car, Pedestrian, Cyclist
                if object_type not in ["Car", "Pedestrian", "Cyclist"]:
                    continue

                # Class label
                class_idx = self.class_to_idx(object_type)
                labels.append(class_idx)

                # 2D Bounding box: Normalize to [0, 1]
                bbox_2d.append([
                    float(data[4]) / img_width,   # left
                    float(data[5]) / img_height,  # top
                    float(data[6]) / img_width,   # right
                    float(data[7]) / img_height   # bottom
                ])

                # 3D Bounding box: Normalize with predefined scale
                bbox_3d.append([
                    float(data[8]) / 3.0,   # height
                    float(data[9]) / 3.0,   # width
                    float(data[10]) / 10.0, # length
                    float(data[11]) / 50.0, # x
                    float(data[12]) / 1.0,  # y
                    float(data[13]) / 50.0, # z
                    float(data[14]) / 3.14  # rotation_y
                ])

        return (
            image,
            points,
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(bbox_2d, dtype=torch.float32),
            torch.tensor(bbox_3d, dtype=torch.float32),
        )

    def class_to_idx(self, class_name):
        # Map the three target object types to numerical labels
        class_map = {
            "Car": 0,
            "Pedestrian": 1,
            "Cyclist": 2
        }
        return class_map.get(class_name, -1)


def custom_collate_fn(batch):
    images, point_clouds, labels, bbox_2d, bbox_3d = [], [], [], [], []
    max_points = 0
    max_objects = 0

    # Find maximum lengths for padding
    for img, pc, lbl, b2d, b3d in batch:
        images.append(img)
        point_clouds.append(pc)
        labels.append(lbl)
        bbox_2d.append(b2d)
        bbox_3d.append(b3d)
        max_points = max(max_points, pc.shape[0])
        max_objects = max(max_objects, lbl.shape[0])

    # Pad point clouds
    padded_pcs = []
    for pc in point_clouds:
        pad_len = max_points - pc.shape[0]
        pad = torch.zeros((pad_len, pc.shape[1]), dtype=pc.dtype)
        pc = torch.cat([pc, pad], dim=0)
        padded_pcs.append(pc)

    # Pad labels and bounding boxes
    padded_labels, padded_bbox_2d, padded_bbox_3d = [], [], []
    for lbl, b2d, b3d in zip(labels, bbox_2d, bbox_3d):
        pad_lbl = torch.full((max_objects - lbl.shape[0],), -1, dtype=torch.long)
        padded_labels.append(torch.cat([lbl, pad_lbl], dim=0))

        pad_b2d = torch.zeros((max_objects - b2d.shape[0], 4), dtype=torch.float32)
        padded_bbox_2d.append(torch.cat([b2d, pad_b2d], dim=0))

        pad_b3d = torch.zeros((max_objects - b3d.shape[0], 7), dtype=torch.float32)
        padded_bbox_3d.append(torch.cat([b3d, pad_b3d], dim=0))

    # Stack everything
    images = torch.stack(images, dim=0)  # (B, 3, H, W)
    point_clouds = torch.stack(padded_pcs, dim=0)  # (B, max_points, 3)
    labels = torch.stack(padded_labels, dim=0)  # (B, max_objects)
    bbox_2d = torch.stack(padded_bbox_2d, dim=0)  # (B, max_objects, 4)
    bbox_3d = torch.stack(padded_bbox_3d, dim=0)  # (B, max_objects, 7)

    return images, point_clouds, labels, bbox_2d, bbox_3d

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    model_path = "best_model.pth"
    data_root = "/teamspace/uploads/kitti_3d_object_detection/training"

    # Dataset and DataLoader
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = KITTILabelDataset(root_dir=data_root, transform=transform, subset_fraction=0.25)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)

    # Load frozen multimodal model
    multimodal_model = load_multimodal_model(model_path, device)
    multimodal_model.to(device).eval()

    # Initialize detection head
    detection_head = DetectionHead(input_dim=1024, num_classes=3)  # Adjust input_dim
    detection_head.apply(init_weights)
    detection_head.to(device)

    # Loss functions and optimizer
    criterion_cls = nn.CrossEntropyLoss()
    # criterion_bbox = nn.SmoothL1Loss()
    criterion_bbox = nn.HuberLoss(delta=1.0)

    # optimizer = optim.AdamW(detection_head.parameters(), lr=1e-3, weight_decay=1e-8)
    optimizer = optim.AdamW(detection_head.parameters(), lr=1e-4, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR by 0.5 every 10 epochs


    # Training loop
    detection_head.train()
    num_epochs = 1
    train_losses = []

    accumulation_steps = 10  # Accumulate gradients for 4 small batches
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (images, point_clouds, labels, bbox_2d, bbox_3d) in enumerate(dataloader):
            images, point_clouds = images.to(device), point_clouds.to(device)

            # Take the first label and bounding boxes
            labels = labels[:, 0].to(device)  # Shape: (B,)
            bbox_2d = bbox_2d[:, 0, :].to(device)  # Shape: (B, 4)
            bbox_3d = bbox_3d[:, 0, :].to(device)  # Shape: (B, 7)

            # Generate embeddings
            combined_emb = get_embeddings(multimodal_model, images, point_clouds)  # Shape: (B, input_dim)

            # Forward pass through detection head
            class_preds, bbox_2d_preds, bbox_3d_preds = detection_head(combined_emb)

            # Loss calculation
            loss_cls = criterion_cls(class_preds, labels)
            loss_bbox_2d = criterion_bbox(bbox_2d_preds, bbox_2d)
            loss_bbox_3d = criterion_bbox(bbox_3d_preds, bbox_3d)

            loss_cls_weight = 1.0
            loss_bbox_2d_weight = 0.5
            loss_bbox_3d_weight = 1.0

            loss = (loss_cls_weight * loss_cls + 
                    loss_bbox_2d_weight * loss_bbox_2d + 
                    loss_bbox_3d_weight * loss_bbox_3d)

            # Backpropagation with gradient accumulation
            loss = loss / accumulation_steps  # Scale loss to account for accumulation
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:  # Perform optimizer step
                torch.nn.utils.clip_grad_norm_(detection_head.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accumulation_steps  # Scale back for proper loss logging

            print(f"Epoch [{epoch+1}/{num_epochs}], Iteration [{batch_idx+1}/{len(dataloader)}], "
                  f"Loss: {loss.item() * accumulation_steps:.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")


    # Save detection head
    torch.save(detection_head.state_dict(), "detection_head.pth")
    print("Training complete and model saved!")

    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(["Loss"])
    plt.savefig("detection_training_loss.png")
    plt.show()

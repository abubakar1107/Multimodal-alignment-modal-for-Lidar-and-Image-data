import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as T
from model import MultiModalAlignmentModel
import itertools
import matplotlib.pyplot as plt

class KITTIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: path to 'kitti_3d_object_detection/training' directory
        transform: optional image transformations (resize, normalize, etc.)
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "image_2")
        self.velodyne_dir = os.path.join(root_dir, "velodyne")
        self.transform = transform

        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.png")))
        # Corresponding velodyne files are *.bin with the same base name.

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)

        # Load point cloud
        lidar_path = os.path.join(self.velodyne_dir, base_name + ".bin")
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4) # (N,4)
        points = points[:, :3] # (N,3)
        point_cloud = torch.from_numpy(points) # (N,4)

        return image, point_cloud


def custom_collate_fn(batch):
    images = []
    point_clouds = []
    max_len = 0

    # Determine max number of points in this batch
    for img, pc in batch:
        images.append(img)
        point_clouds.append(pc)
        if pc.shape[0] > max_len:
            max_len = pc.shape[0]

    # Pad point clouds
    padded_pcs = []
    for pc in point_clouds:
        if pc.shape[0] < max_len:
            pad_len = max_len - pc.shape[0]
            pad = torch.zeros((pad_len, pc.shape[1]), dtype=pc.dtype)
            pc = torch.cat([pc, pad], dim=0)
        padded_pcs.append(pc)

    # Stack
    images = torch.stack(images, dim=0)  # (B, 3, H, W)
    point_clouds = torch.stack(padded_pcs, dim=0) # (B, max_len, 4)

    return images, point_clouds


def cosine_similarity_loss(img_emb, lidar_emb):
    sim = F.cosine_similarity(img_emb, lidar_emb, dim=1)
    loss = 1.0 - sim.mean()
    return loss

if __name__ == "__main__":
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    root = "kitti_3d_object_detection/training"
    transform = T.Compose([
        T.Resize((224, 224)), # resize image to match ViT input size
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    dataset = KITTIDataset(root_dir=root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)

    model = MultiModalAlignmentModel().to(device)
    # Only train Q-Former & projections
    optimizer = torch.optim.AdamW(
        itertools.chain(model.qformer.parameters(),
                        model.img_proj.parameters(),
                        model.lidar_proj.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )

    num_epochs = 2
    model.train()
    train_losses = []


    for epoch in range(num_epochs):
        epoch_loss_sum = 0.0
        epoch_steps = 0
        for i, (images, point_clouds) in enumerate(dataloader, start=1):
            # Move data to GPU
            images = images.to(device)
            point_clouds = point_clouds.to(device)

            optimizer.zero_grad()
            img_emb, lidar_emb = model(images, point_clouds)
            loss = cosine_similarity_loss(img_emb, lidar_emb)
            loss.backward()
            optimizer.step()

            # Print iteration-level loss
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
            train_losses.append(loss.item())
            epoch_loss_sum += loss.item()
            epoch_steps += 1

        # Print epoch-level average loss
        epoch_avg_loss = epoch_loss_sum / epoch_steps
        print(f"====> Epoch [{epoch+1}/{num_epochs}] Finished - Average Loss: {epoch_avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")
    print("Training complete!")

    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(["Loss"])
    plt.savefig("training_loss.png")
    plt.show()

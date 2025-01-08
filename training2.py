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

torch.cuda.empty_cache()

class KITTIDataset(Dataset):
    def __init__(self, root_dir, transform=None, subset_fraction=1.0):
        """
        root_dir: path to 'kitti_3d_object_detection/training' directory
        transform: optional image transformations (resize, normalize, etc.)
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "image_2")
        self.velodyne_dir = os.path.join(root_dir, "velodyne")
        self.transform = transform

        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.png")))
        self.image_files = self.image_files[:int(len(self.image_files) * subset_fraction)]
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

    # Determine half the max number of points in this batch
    for img, pc in batch:
        images.append(img)
        point_clouds.append(pc)
        if pc.shape[0] > max_len:
            max_len = pc.shape[0]
    
    # Use half of the max_len
    max_len = max_len // 2

    # Truncate or pad point clouds
    processed_pcs = []
    for pc in point_clouds:
        if pc.shape[0] > max_len:
            # Truncate to max_len
            pc = pc[:max_len]
        else:
            # Pad to max_len
            pad_len = max_len - pc.shape[0]
            pad = torch.zeros((pad_len, pc.shape[1]), dtype=pc.dtype)
            pc = torch.cat([pc, pad], dim=0)
        processed_pcs.append(pc)

    # Stack
    images = torch.stack(images, dim=0)  # (B, 3, H, W)
    point_clouds = torch.stack(processed_pcs, dim=0)  # (B, max_len, 3)

    return images, point_clouds

def cosine_similarity_loss(img_emb, lidar_emb):
    sim = F.cosine_similarity(img_emb, lidar_emb, dim=1)
    loss = 1.0 - sim.mean()
    return loss

if __name__ == "__main__":
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    root = "/teamspace/uploads/kitti_3d_object_detection/training"
    val_root = "/teamspace/uploads/kitti_3d_object_detection/testing"   
    transform = T.Compose([
        T.Resize((224, 224)), # resize image to match ViT input size
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    dataset = KITTIDataset(root_dir=root, transform=transform, subset_fraction=0.25)
    vat_dataset = KITTIDataset(root_dir=val_root, transform=transform, subset_fraction=0.25)

    dataloader = DataLoader(dataset, batch_size=14, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(vat_dataset, batch_size=14, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)

    model = MultiModalAlignmentModel().to(device)
    # Only train Q-Former & projections
    optimizer = torch.optim.AdamW(
        itertools.chain(model.qformer.parameters(),
                        model.img_proj.parameters(),
                        model.lidar_proj.parameters()),
        lr=1e-5,
        weight_decay=1e-5
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    num_epochs = 1
    model.train()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')


    for epoch in range(num_epochs):
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for i, (images, point_clouds) in enumerate(dataloader, start=1):
            # Training step
            model.train()
            images = images.to(device)
            point_clouds = point_clouds.to(device)

            optimizer.zero_grad()
            img_emb, lidar_emb = model(images, point_clouds)
            loss = cosine_similarity_loss(img_emb, lidar_emb)
            loss.backward()
            optimizer.step()

            # Print training loss
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Step [{i}/{len(dataloader)}], Training Loss: {loss.item():.4f}")
            train_losses.append(loss.item())
            epoch_loss_sum += loss.item()
            epoch_steps += 1

            # Validation step (on a single validation batch)
            model.eval()
            with torch.no_grad():
                try:
                    val_iter = iter(val_dataloader)
                    val_images, val_point_clouds = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_dataloader)
                    val_images, val_point_clouds = next(val_iter)

                # Move data to device
                val_images = val_images.to(device)
                val_point_clouds = val_point_clouds.to(device)

                # Compute validation loss
                val_img_emb, val_lidar_emb = model(val_images, val_point_clouds)
                val_loss = cosine_similarity_loss(val_img_emb, val_lidar_emb)

                # Print validation loss for this batch
                print(f"Epoch [{epoch+1}/{num_epochs}], Training Step [{i}/{len(dataloader)}], Batch Validation Loss: {val_loss.item():.4f}")
                val_losses.append(val_loss.item())

                # Save model if validation loss improves
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    torch.save(model.state_dict(), "best_model.pth")
                    print("Model saved!")

        # Print epoch-level average loss
        epoch_avg_loss = epoch_loss_sum / epoch_steps
        print(f"====> Epoch [{epoch+1}/{num_epochs}] Finished - Average Training Loss: {epoch_avg_loss:.4f}")

        
    print("Model saved to model.pth")
    print("Training complete!")
    

    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(["Loss"])
    plt.savefig("training_loss.png")
    plt.show()

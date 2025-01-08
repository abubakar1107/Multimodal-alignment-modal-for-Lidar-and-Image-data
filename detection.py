import torch
import torch.nn as nn
from model import MultiModalAlignmentModel  # Your pre-trained multimodal model

# Detection head definition
class DetectionHead(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super(DetectionHead, self).__init__()

        # Shared feature extraction
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),  # Replace BatchNorm1d
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )


        # Classification branch
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(128, num_classes)  # Outputs class logits
        )

        # 2D Bounding Box branch
        self.bbox2d_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128,4),
              # Outputs x1, y1, x2, y2
        )

        # 3D Bounding Box branch
        # Simplified 3D Bounding Box branch
        self.bbox3d_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7)  # Outputs h, w, l, x, y, z, ry
        )


    def forward(self, embeddings):
        # Shared feature extraction
        shared_features = self.shared_fc(embeddings)

        # Branches for predictions
        class_preds = self.classifier(shared_features)    # Class predictions
        bbox_2d = self.bbox2d_branch(shared_features)     # 2D bbox predictions
        bbox_3d = self.bbox3d_branch(shared_features)     # 3D bbox predictions

        return class_preds, bbox_2d, bbox_3d


# Load pre-trained multimodal model
def load_multimodal_model(model_path, device):
    multimodal_model = MultiModalAlignmentModel()
    multimodal_model.load_state_dict(torch.load(model_path, map_location=device))
    multimodal_model.eval()
    return multimodal_model

# Generate embeddings
def get_embeddings(multimodal_model, images, point_clouds):
    with torch.no_grad():
        img_emb, lidar_emb = multimodal_model(images, point_clouds)
        combined_emb = torch.cat((img_emb, lidar_emb), dim=1)  # Combine both embeddings
    return combined_emb

import torch
import torch.nn as nn
import timm
from pointnet_loader import load_pretrained_pointnet

class ViTEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)

    def forward(self, images):
        # images: (B, 3, H, W)
        # forward_features typically returns (B, Seq_img, D)
        img_feats = self.vit.forward_features(images)
        return img_feats

class PointNetPlusPlusEncoder(nn.Module):
    def __init__(self, d_model=16):
        super().__init__()
        # Load the pretrained pointnet++ from your old project
        self.pointnet = load_pretrained_pointnet()
        
        # Freeze the PointNet++ weights if you don't want to fine-tune it
        for param in self.pointnet.parameters():
            param.requires_grad = False
        
        # Assuming pointnet outputs (B, N, 64)
        self.adapt_feature_dim = nn.Linear(64, d_model)

    def forward(self, point_clouds):

        # Pass through PointNet++
        with torch.no_grad():
            # self.pointnet should output something like (B, N, 64)
            lidar_feats = self.pointnet(point_clouds)
        
        # Project from 64 to d_model
        lidar_feats = self.adapt_feature_dim(lidar_feats)  # (B, N, d_model)

        return lidar_feats

class QFormer(nn.Module):
    def __init__(self, d_model=16, nhead=2, num_layers=2, num_queries=4):
        super().__init__()
        self.query_embeddings = nn.Parameter(torch.randn(num_queries, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.drop_out = nn.Dropout(p=0.2)

    def forward(self, feats):
        # feats: (B, Seq, D)
        B = feats.size(0)
        queries = self.query_embeddings.unsqueeze(0).expand(B, -1, -1)  # (B, num_queries, D)
        tokens = torch.cat([queries, feats], dim=1) # (B, num_queries+Seq, D)
        tokens = tokens.transpose(0,1) # (num_queries+Seq, B, D)
        output = self.transformer(tokens) # (num_queries+Seq, B, D)
        # output = self.drop_out(output)
        output = output.transpose(0,1) # (B, num_queries+Seq, D)

        q_output = output[:, :queries.size(1), :] # (B, num_queries, D)
        scene_embedding = q_output.mean(dim=1) # (B, D)
        return scene_embedding

class MultiModalAlignmentModel(nn.Module):
    def __init__(self, d_model=512, nhead=2, num_layers=2, num_queries=4, vit_name='vit_tiny_patch16_224'):
        super().__init__()
        self.image_encoder = ViTEncoder(model_name=vit_name, pretrained=True)
        self.lidar_encoder = PointNetPlusPlusEncoder(d_model=d_model)
        self.qformer = QFormer(d_model, nhead, num_layers, num_queries)
        self.vit_adapt_feature_dim = nn.Linear(192, d_model)

        self.img_proj = nn.Linear(d_model, d_model)
        self.lidar_proj = nn.Linear(d_model, d_model)

        

        # Freeze the encoders
        self.image_encoder.requires_grad_(False)
        self.lidar_encoder.requires_grad_(False)

    def forward(self, images, point_clouds):
        # No gradient for encoders
        with torch.no_grad():
            img_feats = self.image_encoder(images) 
            img_feats = self.vit_adapt_feature_dim(img_feats)        # (B, Seq_img, D)
            lidar_feats = self.lidar_encoder(point_clouds) # (B, Seq_lidar, D)

        img_emb = self.qformer(img_feats)     # (B, D)
        lidar_emb = self.qformer(lidar_feats) # (B, D)

        # img_emb = self.img_proj(img_emb)       # (B, D)
        # lidar_emb = self.lidar_proj(lidar_emb) # (B, D)

        return img_emb, lidar_emb

# PROJECT_V1/pointnet_loader.py
import sys
import os
import torch
from Pointnet_Pointnet2_pytorch.models.pointnet2_cls_msg import get_model

sys.path.append(os.path.abspath('./Pointnet_Pointnet2_pytorch/models'))

def load_pretrained_pointnet():
    model = get_model(num_class=40, normal_channel=False)  # Ensure normal_channel=False
    checkpoint_path = './Pointnet_Pointnet2_pytorch/log/classification/pointnet2_msg_normals/checkpoints/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Get state_dict from the checkpoint
    pretrained_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    
    # Filter out mismatched keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    
    # Update the model dictionary and load it
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    return model


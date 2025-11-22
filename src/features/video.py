import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import List

class VideoFeatureExtractor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading Video Model (ResNet50) on {self.device}...")
        
        # Load pre-trained ResNet50
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        
        # Remove the classification head (fc layer) to get features
        # ResNet50 output before fc is 2048 dim
        self.model.fc = nn.Identity()
        self.model.to(self.device)
        self.model.eval()
        
        self.preprocess = weights.transforms()

    def extract(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extracts features from a list of frames (numpy arrays).
        Returns: numpy array of shape (2048,) - averaged over frames.
        """
        if not frames:
            return np.zeros(2048)
            
        # Convert frames to tensors
        # Frames are expected to be RGB numpy arrays
        batch_tensors = []
        for frame in frames:
            # Convert numpy to PIL
            pil_img = Image.fromarray(frame)
            tensor_img = self.preprocess(pil_img)
            batch_tensors.append(tensor_img)
            
        batch = torch.stack(batch_tensors).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch)
            
        # Average pooling over the frames to get one vector per segment
        # features shape: (N_frames, 2048)
        pooled_features = torch.mean(features, dim=0).cpu().numpy()
        
        return pooled_features

    def get_embedding_dim(self) -> int:
        return 2048

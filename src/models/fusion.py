import torch
import torch.nn as nn
import torch.nn.functional as F

class RoME_Scorer(nn.Module):
    def __init__(self, text_dim: int, audio_dim: int, video_dim: int, role_dim: int, hidden_dim: int = 128):
        super(RoME_Scorer, self).__init__()
        
        # Feature projection layers to map all inputs to the same hidden dimension
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.role_proj = nn.Linear(role_dim, hidden_dim)
        
        # Attention Mechanism (Simple Dot-Product Attention)
        # We want to see how much each modality "aligns" with the Role
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # Final Scoring MLP
        # Input: Concat of (Weighted Modalities + Role) -> 2 * hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, text_emb, audio_emb, video_emb, role_emb):
        """
        Args:
            text_emb: (Batch, text_dim)
            audio_emb: (Batch, audio_dim)
            video_emb: (Batch, video_dim)
            role_emb: (Batch, role_dim) - Note: Role is usually constant for a batch of segments from same query
        """
        # Project features
        # Shape: (Batch, 1, Hidden)
        t = self.text_proj(text_emb).unsqueeze(1)
        a = self.audio_proj(audio_emb).unsqueeze(1)
        v = self.video_proj(video_emb).unsqueeze(1)
        r = self.role_proj(role_emb).unsqueeze(1)
        
        # Stack modalities: (Batch, 3, Hidden)
        modalities = torch.cat([t, a, v], dim=1)
        
        # Role acts as the "Query" for attention, Modalities are "Key/Value"
        # attn_output: (Batch, 1, Hidden)
        attn_output, _ = self.attention(query=r, key=modalities, value=modalities)
        
        # Concatenate Role context with the attended modality features
        # Shape: (Batch, 2 * Hidden)
        combined = torch.cat([r.squeeze(1), attn_output.squeeze(1)], dim=1)
        
        # Score
        score = self.classifier(combined)
        return score

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, S: int = 224):
        super().__init__()
        
        # Conv7×7 s2 3→32, MaxPool 3×3 s2 p1 → S/4
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Conv5×5 32→64 → S/4 (разрешение не меняется)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Conv3×3 s2 64→128 → S/8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Conv1×1 128→256 → S/8
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu4 = nn.ReLU(inplace=True)
        
        # Conv3×3 s2 256→256 → S/16
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu5 = nn.ReLU(inplace=True)
        
        # Conv1×1 256→512 → S/16
        self.conv6 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu6 = nn.ReLU(inplace=True)
        
        # Head: GlobalAvgPool, Linear 512→256, ReLU, Linear 256→100
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 256)
        self.relu_head = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 100)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1: Conv7×7 + BN + ReLU + Pool + BN
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Block 2: Conv5×5 + BN + ReLU
        x = self.conv2(x)
        x = self.relu2(x)
        
        # Block 3: Conv3×3 s2 + BN + ReLU
        x = self.conv3(x)
        x = self.relu3(x)
        
        # Block 4: Conv1×1 + BN + ReLU
        x = self.conv4(x)
        x = self.relu4(x)
        
        # Block 5: Conv3×3 s2 + BN + ReLU
        x = self.conv5(x)
        x = self.relu5(x)
        
        # Block 6: Conv1×1 + BN + ReLU
        x = self.conv6(x)
        x = self.relu6(x)
        
        # Head
        x = self.global_avg_pool(x)  # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)     # [B, 512]
        x = self.fc1(x)
        x = self.relu_head(x)
        x = self.fc2(x)
        
        return x

import os
import torch
import torch.nn as nn

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),   # 48x48 → 46x46
            nn.ReLU(),
            nn.MaxPool2d(2),      # → 23x23
            nn.Conv2d(32, 64, 3), # → 21x21
            nn.ReLU(),
            nn.MaxPool2d(2)       # → 10x10
        )

        # Auto-calculate FC input size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 48, 48)
            conv_out = self.conv(dummy)
            self.fc_input_size = conv_out.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def load_model(path="models/emotion_model.pth"):
    model = EmotionCNN()

    if os.path.exists(path) and os.path.getsize(path) > 0:
        model.load_state_dict(torch.load(path, map_location="cpu"))
        print("Emotion model loaded")
    else:
        print("Model file not found or empty. Using untrained model.")

    model.eval()
    return model

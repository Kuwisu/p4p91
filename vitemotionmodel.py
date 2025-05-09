import timm
import torch.nn as nn

# Load pretrained ViT model and modify classification head
class ViTEmotionClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(ViTEmotionClassifier, self).__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)  # Modify head

    def forward(self, x):
        return self.model(x)

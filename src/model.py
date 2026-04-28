import torch.nn as nn
import timm # PyTorch Image Models library

class RetinaNet(nn.Module):
    def __init__(self, num_classes=5):
        super(RetinaNet, self).__init__()
        # Load pre-trained EfficientNet-B4 [cite: 64]
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True)
        
        # Replace the classifier head
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        # Freeze weights for Phase 1 of training [cite: 78]
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Keep the new head unfrozen
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        # Unfreeze for fine-tuning in Phase 2 [cite: 78]
        for param in self.backbone.parameters():
            param.requires_grad = True
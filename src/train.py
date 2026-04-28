import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_dataloaders
from src.model import RetinaNet
from sklearn.metrics import cohen_kappa_score

# Focal Loss to handle class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_dataloaders("data/raw/train.csv", "data/processed", batch_size=16)
    
    model = RetinaNet().to(device)
    criterion = FocalLoss()
    
    # PHASE 1: Freeze Backbone
    print("Starting Phase 1: Training Head Only...")
    model.freeze_backbone()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(5): 
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Phase 1 - Epoch {epoch+1} complete.")

    # PHASE 2: Unfreeze All
    print("Starting Phase 2: Fine-tuning All Layers...")
    model.unfreeze_all()
    optimizer = optim.Adam(model.parameters(), lr=1e-5) # Lower LR for fine-tuning
    
    best_kappa = 0
    for epoch in range(15):
        model.train()
        # (Training loop logic same as above)
        
        # Validation for Checkpointing
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        print(f"Epoch {epoch+1} Val Kappa: {kappa:.4f}")
        
        if kappa > best_kappa:
            best_kappa = kappa
            torch.save(model.state_dict(), "outputs/checkpoints/best_model.pth")

if __name__ == "__main__":
    train_model()
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report, roc_curve, auc
from src.dataset import get_dataloaders
from src.model import RetinaNet

def evaluate_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    _, _, test_loader = get_dataloaders("data/raw/train.csv", "data/processed", batch_size=16)
    
    # Load Model
    model = RetinaNet()
    model.load_state_dict(torch.load(model_path))
    model.to(device).eval()
    
    all_preds = []
    all_labels = []
    all_probs = []

    print("Evaluating on Test Set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # 1. Quadratic Weighted Kappa [cite: 80, 91]
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    print(f"\nFinal Quadratic Weighted Kappa: {kappa:.4f}")

    # 2. Confusion Matrix [cite: 84, 93]
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No DR', 'Mild', 'Mod', 'Severe', 'Prolif'],
                yticklabels=['No DR', 'Mild', 'Mod', 'Severe', 'Prolif'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('outputs/plots/confusion_matrix.png')
    
    # 3. Classification Report [cite: 94]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

if __name__ == "__main__":
    evaluate_model("outputs/checkpoints/best_model.pth")
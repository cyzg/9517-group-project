import os
import shutil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import pretrainedmodels

# -------------------------- File Paths --------------------------
train_path = "/content/gdrive/MyDrive/9517 project/data/train"
test_path = "/content/gdrive/MyDrive/9517 project/data/test"
model_path = "/content/gdrive/MyDrive/9517 project/SENet_epoch12.pth"


# -------------------------- Clean and Create Output Folders --------------------------
shutil.rmtree("/content/gdrive/MyDrive/9517 project/SENet/gradcam_heatmaps", ignore_errors=True)
shutil.rmtree("/content/gdrive/MyDrive/9517 project/SENet/misclassified_images", ignore_errors=True)
shutil.rmtree("/content/gdrive/MyDrive/9517 project/SENet/similar_images", ignore_errors=True)

os.makedirs("/content/gdrive/MyDrive/9517 project/SENet/gradcam_heatmaps", exist_ok=True)
os.makedirs("/content/gdrive/MyDrive/9517 project/SENet/misclassified_images", exist_ok=True)
os.makedirs("/content/gdrive/MyDrive/9517 project/SENet/similar_images", exist_ok=True)

# -------------------------- Image Transformation for Test Set --------------------------
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------- Custom Dataset Class with File Paths --------------------------
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original = super().__getitem__(index)
        path = self.imgs[index][0]
        return original + (path,)

# -------------------------- Load Datasets --------------------------
train_data = datasets.ImageFolder(train_path)


test_data = ImageFolderWithPaths(test_path, transform=transform_test)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# -------------------------- Load Pretrained SENet and Modify Classifier --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = pretrainedmodels.__dict__['se_resnet50'](pretrained='imagenet')

# Replace final linear layer with one matching number of classes in dataset
model.last_linear = nn.Linear(model.last_linear.in_features, len(train_data.classes))
# Load fine-tuned weights
model.load_state_dict(torch.load(model_path))
model = model.to(device).eval()

# -------------------------- Define Feature Extractor (without classifier head) --------------------------
# Extract all layers except final avgpool + classifier
feature_extractor = nn.Sequential(*list(model.children())[:-2]).to(device).eval()

# -------------------------- Inference and Feature Collection --------------------------
all_preds, all_labels, all_paths, all_features = [], [], [], []
misclassified_info = []

with torch.no_grad():
    for imgs, labels, paths in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

        feats = feature_extractor(imgs).mean(dim=[2, 3])

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_paths.extend(paths)
        all_features.append(feats.cpu().numpy())

        for i in range(len(preds)):
            if preds[i] != labels[i]:
                misclassified_info.append({
                    'filename': paths[i],
                    'true_label': test_data.classes[labels[i]],
                    'predicted_label': test_data.classes[preds[i]],
                    'feature': feats[i].cpu().numpy()
                })

# -------------------------- Convert Features to DataFrame --------------------------
all_features = np.concatenate(all_features, axis=0)
# Create a DataFrame containing image paths, true/predicted labels, and feature vectors
df_all = pd.DataFrame({
    'path': all_paths,
    'label': [test_data.classes[y] for y in all_labels],
    'pred': [test_data.classes[y] for y in all_preds],
    'feature': list(all_features)
})

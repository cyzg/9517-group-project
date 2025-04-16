import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import efficientnet_b0
import torch.nn as nn
import pandas as pd  

# loading the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_data.classes))
model = model.to(device)
model.load_state_dict(torch.load("/content/gdrive/MyDrive/9517 project/Efficientnet_epoch12.pth"))
model.eval()


all_preds = []
all_labels = []
misclassified = []

# loading data and predicting
with torch.no_grad():
    for imgs, labels, paths in test_loader:  # 注意此处添加 paths（文件路径）
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

        # recording the wrong classification
        for path, pred, label in zip(paths, preds.cpu().numpy(), labels.numpy()):
            if pred != label:
                misclassified.append({
                    'filename': path,  
                    'true_label': test_data.classes[label],
                    'predicted_label': test_data.classes[pred]
                })

# output the report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=test_data.classes))

# making confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=test_data.classes, yticklabels=test_data.classes, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# recording the wrong predicting data
df_misclassified = pd.DataFrame(misclassified)
df_misclassified.to_csv("/content/gdrive/MyDrive/9517 project/misclassified_samples_ess_12.csv", index=False)
print(f"record {len(misclassified)} wrong data，saved to  misclassified_samples.csv")

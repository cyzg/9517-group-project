import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pretrainedmodels
import pandas as pd  
# loading the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = pretrainedmodels.__dict__['se_resnet50'](pretrained='imagenet')

# change the output to 15 layer
num_features = model.last_linear.in_features
model.last_linear = nn.Linear(num_features, 15)

model = model.to(device)
model.load_state_dict(torch.load("/content/gdrive/MyDrive/9517 project/SENet_epoch12.pth"))
model.eval()

all_preds = []
all_labels = []
misclassified = []


with torch.no_grad():
    for imgs, labels, paths in test_loader:  
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

        # recording the wrong data
        for path, pred, label in zip(paths, preds.cpu().numpy(), labels.numpy()):
            if pred != label:
                misclassified.append({
                    'filename': path,  # 或 path.split('/')[-1] 只保留文件名
                    'true_label': test_data.classes[label],
                    'predicted_label': test_data.classes[pred]
                })



#output the classification report
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

# recording the wrong classification data
df_misclassified = pd.DataFrame(misclassified)
df_misclassified.to_csv("/content/gdrive/MyDrive/9517 project/misclassified_samples_senet_12.csv", index=False)
print(f"record {len(misclassified)} wrong data，saved to  misclassified_samples.csv")

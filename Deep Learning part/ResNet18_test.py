import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  

# get the model we use
model.load_state_dict(torch.load("/content/gdrive/MyDrive/9517 project/resnet18_epoch12.pth"))
model.eval()

all_preds = []
all_labels = []
misclassified = []


# begin to tets the data
with torch.no_grad():
    for imgs, labels, paths in test_loader:  
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

        # recording the wrong classfication
        for path, pred, label in zip(paths, preds.cpu().numpy(), labels.numpy()):
            if pred != label:
                misclassified.append({
                    'filename': path,  
                    'true_label': test_data.classes[label],
                    'predicted_label': test_data.classes[pred]
                })






# output the result
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=test_data.classes))

# making the confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=test_data.classes, yticklabels=test_data.classes, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# save the wrong classificaton result
df_misclassified = pd.DataFrame(misclassified)
df_misclassified.to_csv("/content/gdrive/MyDrive/9517 project/misclassified_samples_res18_12.csv", index=False)
print(f"record {len(misclassified)} wrong data，saved to  misclassified_samples.csv")

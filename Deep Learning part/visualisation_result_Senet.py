import cv2
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[0, class_idx]
        loss.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam
def overlay_heatmap(img_path, cam):
    raw = cv2.imread(img_path)
    raw = cv2.resize(raw, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(raw, 0.5, heatmap, 0.5, 0)
    return overlay

# Grad-CAM target layer for SENet: layer4
target_layer = model.layer4
gradcam = GradCAM(model, target_layer)

similarity_records = []
for item in misclassified_info:
    f1 = item['feature'].reshape(1, -1)
    pred_class = item['predicted_label']
    candidates = df_all[df_all['label'] == pred_class]
    candidate_feats = np.stack(candidates['feature'].values)
    similarities = cosine_similarity(f1, candidate_feats).flatten()
    max_idx = np.argmax(similarities)
    similar_path = candidates.iloc[max_idx]['path']

    similarity_records.append({
        'wrong_image': item['filename'],
        'true_label': item['true_label'],
        'predicted_label': item['predicted_label'],
        'similar_image': similar_path,
        'similarity': similarities[max_idx]
    })
    shutil.copy(item['filename'], os.path.join("/content/gdrive/MyDrive/9517 project/SENet/misclassified_images", os.path.basename(item['filename'])))
    shutil.copy(similar_path, os.path.join("/content/gdrive/MyDrive/9517 project/SENet/similar_images", os.path.basename(similar_path)))

    input_image = Image.open(item['filename']).convert('RGB')
    input_tensor = transform_test(input_image).unsqueeze(0).to(device)
    cam = gradcam.generate(input_tensor)
    heatmap_image = overlay_heatmap(item['filename'], cam)
    heatmap_path = os.path.join("/content/gdrive/MyDrive/9517 project/SENet/gradcam_heatmaps", os.path.basename(item['filename']))
    cv2.imwrite(heatmap_path, heatmap_image)

df_sim = pd.DataFrame(similarity_records)
df_sim.to_csv("/content/gdrive/MyDrive/9517 project/SENet/misclassified_similarity.csv", index=False)

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=test_data.classes))
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=test_data.classes, yticklabels=test_data.classes, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

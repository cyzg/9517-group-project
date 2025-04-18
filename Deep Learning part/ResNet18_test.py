train_path = "/content/gdrive/MyDrive/9517 project/data/train"
test_path = "/content/gdrive/MyDrive/9517 project/data/test"
model_path = "/content/gdrive/MyDrive/9517 project/resnet18_epoch12.pth"

shutil.rmtree("/content/gdrive/MyDrive/9517 project/Resnet/gradcam_heatmaps", ignore_errors=True)
shutil.rmtree("/content/gdrive/MyDrive/9517 project/Resnet/misclassified_images", ignore_errors=True)
shutil.rmtree("/content/gdrive/MyDrive/9517 project/Resnet/similar_images", ignore_errors=True)
os.makedirs("/content/gdrive/MyDrive/9517 project/Resnet/gradcam_heatmaps", exist_ok=True)
os.makedirs("/content/gdrive/MyDrive/9517 project/Resnet/misclassified_images", exist_ok=True)
os.makedirs("/content/gdrive/MyDrive/9517 project/Resnet/similar_images", exist_ok=True)

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original = super().__getitem__(index)
        path = self.imgs[index][0]
        return original + (path,)

train_data = datasets.ImageFolder(train_path)
test_data = ImageFolderWithPaths(test_path, transform=transform_test)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))
model.load_state_dict(torch.load(model_path))
model = model.to(device).eval()
feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device).eval()

all_preds, all_labels, all_paths, all_features = [], [], [], []
misclassified_info = []

with torch.no_grad():
    for imgs, labels, paths in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        feats = feature_extractor(imgs).squeeze(-1).squeeze(-1)

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

all_features = np.concatenate(all_features, axis=0)
df_all = pd.DataFrame({
    'path': all_paths,
    'label': [test_data.classes[y] for y in all_labels],
    'pred': [test_data.classes[y] for y in all_preds],
    'feature': list(all_features)
})




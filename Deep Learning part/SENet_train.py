import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report
import pretrainedmodels
from tqdm import tqdm
from torchvision.datasets import ImageFolder


train_path = "/content/gdrive/MyDrive/9517 project/data/train" #the path of training data
test_path = "/content/gdrive/MyDrive/9517 project/data/test" #the path of testing data
class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]  
        return img, label, path


# training data augmentation
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),   
    transforms.ToTensor(),           
    transforms.Normalize(            
        [0.485, 0.456, 0.406],       
        [0.229, 0.224, 0.225]        
    )
])


train_data = datasets.ImageFolder(train_path, transform=transform_train)
test_data = ImageFolderWithPaths(test_path, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

# The define of model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = pretrainedmodels.__dict__['se_resnet50'](pretrained='imagenet')

# change the output to 15 layer (because our data have 15 class)
num_features = model.last_linear.in_features
model.last_linear = nn.Linear(num_features, 15)

model = model.to(device)
# the configuration of training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# start training
for epoch in range(12): #change the number of epoch you wanna train
    print(f"start {epoch+1}  epoch")
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")
    
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
torch.save(model.state_dict(), "/content/gdrive/MyDrive/9517 project/SENet_epoch12.pth")

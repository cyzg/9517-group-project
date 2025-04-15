import os, shutil
import random
from pathlib import Path

original_dir = Path("/content/drive/MyDrive/9517 project/skyview") #The address to put the original data
train_dir = Path("/content/drive/MyDrive/9517 project/data/train") #The address to put the trainning data
test_dir = Path("/content/drive/MyDrive/9517 project/data/test") #The address to put the testing data

# clear the content in the old directory
for d in [train_dir, test_dir]:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)


random.seed(42)

for category in sorted(os.listdir(original_dir)):
    cat_path = original_dir / category
    if not cat_path.is_dir(): continue

    imgs = list(cat_path.glob("*"))
    # mix up the order of the photo
    random.shuffle(imgs)
    split = int(0.8 * len(imgs))
    train_imgs, test_imgs = imgs[:split], imgs[split:]
    
    (train_dir / category).mkdir(parents=True)
    (test_dir / category).mkdir(parents=True)

    for img in train_imgs:
        shutil.copy(img, train_dir / category / img.name)
    for img in test_imgs:
        shutil.copy(img, test_dir / category / img.name)

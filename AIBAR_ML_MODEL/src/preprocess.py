import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import gc  # Garbage Collector interface

from unet_module import UNet


class CustomDataset(Dataset):
    def __init__(self, image_paths, annotation_file, image_transform=None, mask_transform=None):
        self.image_paths = image_paths
        self.coco = COCO(annotation_file)
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.image_transform:
            img = self.image_transform(img)

        img_id = os.path.splitext(os.path.basename(img_path))[0]
        img_ids = self.coco.getImgIds(imgIds=[int(img_id)])
        ann_ids = self.coco.getAnnIds(imgIds=img_ids)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        
        for ann in anns:
            ann_mask = self.coco.annToMask(ann)
            if ann_mask.shape != mask.shape:
                ann_mask = cv2.resize(ann_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = np.maximum(mask, ann_mask)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = torch.as_tensor(mask, dtype=torch.int64)  # Ensure dtype is torch.int64 for CrossEntropyLoss
        return img, mask.squeeze() if mask.ndim > 2 else mask

        
        return img, mask

# Define transforms for images
image_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define transforms for masks
mask_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

if __name__ == "__main__":
    print("Starting script...")

    # Use CPU for training
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Dataset paths
    DATASET_DIR = "/home/shki/Documents/CoCoDataset/subset"
    ANNOTATIONS_PATH = "/home/shki/Documents/CoCoDataset/annotations/instances_train2017.json"

    # Loading dataset
    image_paths = glob.glob(os.path.join(DATASET_DIR, '*.jpg'))
    dataset = CustomDataset(
        image_paths=image_paths,
        annotation_file=ANNOTATIONS_PATH,
        image_transform=image_transforms,
        mask_transform=mask_transforms
    )
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    # Data loaders with reduced batch size and num_workers for CPU
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Initialize the model and move it to the CPU
    net = UNet(n_class=80).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # Using Adam optimizer

    # Training loop
    min_val_loss = float('inf')  # Initialize min_val_loss
    for epoch in range(100):
        net.train()
        running_loss = 0.0
        for i, (inputs, masks) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
            inputs = inputs.to(device)
            masks = masks.to(device)  # Masks are already of dtype torch.int64

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0:  # Print every 10 batches
                print(f'Batch {i} training loss: {loss.item()}')

        # Validation phase
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (inputs, masks) in enumerate(val_loader):
                inputs = inputs.to(device)
                masks = masks.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} validation loss: {avg_val_loss:.4f}")

        if avg_val_loss < min_val_loss:
            print(f"Validation loss decreased. Saving model...")
            torch.save(net.state_dict(), os.path.join(DATASET_DIR, 'best_unet_model.pth'))
            min_val_loss = avg_val_loss

    print('Finished Training')
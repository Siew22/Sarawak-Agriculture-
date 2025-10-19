# train/train_model_2.py (12GB VRAM - 最终旗舰野战版)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader, Dataset
import os
import json
from tqdm import tqdm
import time
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- 12GB VRAM 自研旗舰配置 ---
DATA_DIR = '../../PlantVillage-Dataset/raw/color'
MODEL_SAVE_PATH = '../../models_store/pepper_model_b2_v3_robust.pth' # v3 代表鲁棒性增强版
LABELS_PATH = '../../models_store/disease_labels.json'
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS = 80 # 更多的数据和更难的任务，需要更长的训练
LEARNING_RATE = 0.001
MODEL_ARCHITECTURE = 'efficientnet_b2'

class PlantVillageDataset(Dataset):
    """自定义数据集类，以无缝集成 Albumentations 图像增强库。"""
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root=root_dir)
        self.transform = transform
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu': 
        print("⚠️  警告: 未检测到CUDA, 训练会非常慢。")
    else: 
        print(f"✅ 检测到CUDA设备: {torch.cuda.get_device_name(0)}")

    # --- 自动更新标签文件 ---
    print("正在扫描数据集并更新标签文件...")
    try:
        class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
        NUM_CLASSES = len(class_names)
        label_map = {str(i): class_name for i, class_name in enumerate(class_names)}
        with open(LABELS_PATH, 'w') as f:
            json.dump(label_map, f, indent=4)
        print(f"✅ 标签文件已更新，共找到 {NUM_CLASSES} 个类别。")
    except FileNotFoundError:
        print(f"❌ 错误: 数据集目录 '{DATA_DIR}' 未找到！请检查路径是否为 '../../PlantVillage-Dataset/raw/color'。")
        return
    
    # --- 极限野外模拟 数据增强 (Albumentations) ---
    data_transforms = {
        'train': A.Compose([
            A.RandomResizedCrop(height=260, width=260, scale=(0.6, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            A.OneOf([
                A.GaussNoise(p=1.0), A.ISONoise(p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0), A.GaussianBlur(blur_limit=7, p=1.0),
            ], p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(288, 288),
            A.CenterCrop(260, 260),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    }

    full_dataset = PlantVillageDataset(root_dir=DATA_DIR)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42) 
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    }

    print(f"正在构建一个全新的 '{MODEL_ARCHITECTURE}' 模型 (100% 完全自研)...")
    model = models.efficientnet_b2(weights=None, num_classes=NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - 5, eta_min=1e-6)
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    start_time = time.time()
    best_acc = 0.0

    print("\n--- 开始“旗舰野战”训练 ---")
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS} | 当前学习率: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 25)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0
            
            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}")
            for inputs, labels in progress_bar:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                progress_bar.set_postfix(loss=f'{loss.item():.4f}')

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"🎉 新的最佳自研旗舰模型已保存 (Accuracy: {best_acc:.4f}) 🎉")
        
        scheduler.step()

    time_elapsed = time.time() - start_time
    print(f'\n--- 训练完成 ---')
    print(f'总耗时: {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    print(f'🏆 最佳验证集准确率: {best_acc:4f}')

if __name__ == "__main__":
    train()

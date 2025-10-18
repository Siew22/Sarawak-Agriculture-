# train/train_model.py (6GB VRAM - 优化增强版)
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

# --- 6GB VRAM 自研优化配置 ---
DATA_DIR = '../../PlantVillage-Dataset/raw/color'
MODEL_SAVE_PATH = '../../models_store/scratch_model_b0_6gb_v2.pth' # 新版本命名
LABELS_PATH = '../../models_store/disease_labels.json'

# --- 为6GB VRAM 精心调校的参数 ---
BATCH_SIZE = 16  # 保持一个安全的批量大小
NUM_WORKERS = 2  # 减少CPU和内存压力
NUM_EPOCHS = 50  # 从零训练需要足够的轮次
LEARNING_RATE = 0.001
MODEL_ARCHITECTURE = 'efficientnet_b0'

# 自定义Dataset类以集成Albumentations
class PlantVillageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root=root_dir)
        self.transform = transform
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu': 
        print("⚠️  警告: 未检测到CUDA, 训练会非常慢。")
    else: 
        print(f"✅ 检测到CUDA设备: {torch.cuda.get_device_name(0)}")

    try:
        with open(LABELS_PATH, 'r') as f:
            NUM_CLASSES = len(json.load(f))
    except FileNotFoundError:
        print(f"❌ 错误: 标签文件 '{LABELS_PATH}' 未找到！")
        return
    print(f"✅ 数据集共有 {NUM_CLASSES} 个类别。")

    # --- 专为6GB VRAM优化的数据增强策略 ---
    data_transforms = {
        'train': A.Compose([
            # --- ↓↓↓ 关键修复：使用新的函数调用语法 ↓↓↓ ---
            A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2), 
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # ... (val 部分的代码无需修改)
        'val': A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    }

    full_dataset = PlantVillageDataset(root_dir=DATA_DIR)
    
    # 划分数据集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)

    # 应用transform
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    }

    print(f"正在构建一个全新的 '{MODEL_ARCHITECTURE}' 模型 (100% 完全自研)...")
    # --- 核心保证：weights=None ---
    model = models.efficientnet_b0(weights=None, num_classes=NUM_CLASSES)
    model = model.to(device)

    # --- 引入优化的训练策略 ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    start_time = time.time()
    best_acc = 0.0

    # --- 训练主循环 ---
    print("\n--- 开始优化版训练 ---")
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
                print(f"🎉 新的最佳自研经济版模型已保存 (Accuracy: {best_acc:.4f}) 🎉")
        
        scheduler.step()

    time_elapsed = time.time() - start_time
    print(f'\n--- 训练完成 ---')
    print(f'总耗时: {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    print(f'🏆 最佳验证集准确率: {best_acc:4f}')

if __name__ == "__main__":
    train()
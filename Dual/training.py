import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader
import time
import logging
import csv
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from tqdm import tqdm

# --- Logging Setup ---
def setup_logging(log_file: str):
    # Clears existing handlers to prevent duplicate logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() # Also print logs to console
        ],
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# --- Helper Class for Squeeze-and-Excitation Layer ---
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- REVISED Model Class with Attentional Fusion ---
class PathologyFoundationModelPipeline(nn.Module):
    def __init__(self, device: str, num_classes: int, feature_dim: int = 1280):
        super(PathologyFoundationModelPipeline, self).__init__()
        self.feature_dim, self.num_classes, self.device = feature_dim, num_classes, device
        
        # --- (Extractors setup remains unchanged) ---
        self.rgb_extractor = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0, global_pool='').to(self.device)
        self.fluor_extractor = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0, global_pool='').to(self.device)
        
        fluor_conv = self.fluor_extractor.conv_stem
        new_fluor_conv = nn.Conv2d(
            in_channels=1, out_channels=fluor_conv.out_channels,
            kernel_size=fluor_conv.kernel_size, stride=fluor_conv.stride,
            padding=fluor_conv.padding, bias=fluor_conv.bias is not None
        ).to(self.device)
        with torch.no_grad():
            new_fluor_conv.weight = nn.Parameter(fluor_conv.weight.mean(dim=1, keepdim=True))
            if new_fluor_conv.bias is not None:
                new_fluor_conv.bias.copy_(fluor_conv.bias)
        self.fluor_extractor.conv_stem = new_fluor_conv

        # --- MODIFIED: Convolutional Fusion with Squeeze-and-Excitation Attention ---
        fusion_channels = 512
        self.fusion_conv = nn.Sequential(
            # 1. Project channels down from 2560 to 512
            nn.Conv2d(feature_dim * 2, fusion_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
            
            # 2. Process spatially
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),

            # 3. Apply channel-wise attention to the fused map
            SELayer(channel=fusion_channels)
        ).to(self.device)

        # --- (Classifier and transforms remain unchanged) ---
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(fusion_channels, num_classes).to(self.device)
        
        self.rgb_transforms = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.fluor_transforms = transforms.Normalize(mean=[0.174], std=[0.133])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # --- (Forward pass logic remains unchanged) ---
        rgb_images = self.rgb_transforms(images[:, :3, :, :])
        fluor_images = self.fluor_transforms(images[:, 3:4, :, :])
        
        with torch.autocast(device_type=self.device.split(':')[0], dtype=torch.float16):
            rgb_feature_map = self.rgb_extractor.forward_features(rgb_images)
            fluor_feature_map = self.fluor_extractor.forward_features(fluor_images)
        
        combined_map = torch.cat((rgb_feature_map, fluor_feature_map), dim=1).float()
        fused_map = self.fusion_conv(combined_map)
        pooled_features = self.avgpool(fused_map).flatten(1)
        logits = self.classifier(pooled_features)
        return logits

# --- REVISED Dataset Class ---
class PairedTileDataset(Dataset):
    def __init__(self, he_paths, fluor_paths, labels, is_train=True, fixed_size=(512, 512)):
        self.he_paths, self.fluor_paths, self.labels = he_paths, fluor_paths, labels

        # Define transforms based on whether it's a training or validation set
        if is_train:
            # Spatial transforms are applied to both images identically
            self.spatial_transforms = transforms.Compose([
                transforms.Resize(fixed_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ])
            # Color transforms are applied separately
            self.he_transforms = transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
                transforms.ToTensor()
            ])
            self.fluor_transforms = transforms.Compose([
                transforms.ToTensor()
            ])
        else: # Validation/Testing
            self.spatial_transforms = transforms.Resize(fixed_size)
            self.he_transforms = transforms.ToTensor()
            self.fluor_transforms = transforms.ToTensor()

    def __len__(self):
        return len(self.he_paths)

    def __getitem__(self, idx):
        try:
            he_path, fluor_path = self.he_paths[idx], self.fluor_paths[idx]
            he_img = Image.open(he_path).convert('RGB')
            fluor_img = Image.open(fluor_path).convert('L')
            label = self.labels[idx]
            
            # Apply spatial transforms with the same seed for synchronization
            seed = torch.seed()
            torch.manual_seed(seed)
            he_img = self.spatial_transforms(he_img)
            torch.manual_seed(seed)
            fluor_img = self.spatial_transforms(fluor_img)

            # Apply color transforms and convert to tensor
            # NOTE: Normalization is NOT done here. It is handled in the model's forward pass.
            he_tensor = self.he_transforms(he_img)
            fluor_tensor = self.fluor_transforms(fluor_img)
            
            # Combine tensors to create a 4-channel input
            combined_tensor = torch.cat((he_tensor, fluor_tensor), dim=0)
            
            return combined_tensor, label, he_path, fluor_path
        except Exception as e:
            # Return None for the collate function to filter out
            # print(f"Warning: Error loading image pair {he_path}, {fluor_path}: {e}. Skipping.")
            return None

def paired_collate_fn(batch):
    # Filter out samples that failed to load
    batch = [item for item in batch if item is not None]
    if not batch: return None, None, None, None # Handle cases where a whole batch fails
    
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    he_paths = [item[2] for item in batch]
    fluor_paths = [item[3] for item in batch]
    return images, labels, he_paths, fluor_paths

def linear_warmup_scheduler(epoch, warmup_epochs=10):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0

def main():
    # --- Configuration ---
    TRAIN_HE_PATH = "./subtype2_n"
    TRAIN_FLUOR_PATH = "./f_subtype2"
    VAL_HE_PATH = "./subtype_n"
    VAL_FLUOR_PATH = "./f_subtype"
    CHECKPOINT_DIR = "checkpoints" # Use a new directory
    
    # Hyperparameters
    EPOCHS = 150
    BATCH_SIZE = 32
    IMAGE_SIZE = 512
    LR = 1e-4
    ACCUMULATION_STEPS = 1
    WARMUP_EPOCHS = 10
    PATIENCE = 25
    NUM_WORKERS = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Setup ---
    torch.backends.cudnn.benchmark = True
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    LOG_FILE = os.path.join(CHECKPOINT_DIR, "training_log.txt")
    setup_logging(LOG_FILE)
    logging.info(f"Using device: {DEVICE}")

    # --- Load Data Paths ---
    all_data = {}
    for phase, he_dir, fluor_dir in [("train", TRAIN_HE_PATH, TRAIN_FLUOR_PATH), ("val", VAL_HE_PATH, VAL_FLUOR_PATH)]:
        logging.info(f"Scanning {phase} data...")
        he_paths, fluor_paths, labels = [], [], []
        class_names = sorted([d for d in os.listdir(he_dir) if os.path.isdir(os.path.join(he_dir, d))])
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        
        for class_name, class_idx in class_to_idx.items():
            for img_name in os.listdir(os.path.join(he_dir, class_name)):
                he_path = os.path.join(he_dir, class_name, img_name)
                fluor_path = os.path.join(fluor_dir, class_name, img_name)
                if os.path.exists(fluor_path):
                    he_paths.append(he_path)
                    fluor_paths.append(fluor_path)
                    labels.append(class_idx)
        all_data[phase] = (np.array(he_paths), np.array(fluor_paths), np.array(labels))
        logging.info(f"Found {len(he_paths)} paired {phase} images.")

    idx_to_class = {i: name for name, i in class_to_idx.items()}
    if len(all_data["train"][0]) == 0 or len(all_data["val"][0]) == 0:
        raise ValueError("No data found for training or validation.")

    # --- Class Weights and DataLoaders ---
    class_weights = compute_class_weight('balanced', classes=np.unique(all_data["train"][2]), y=all_data["train"][2])
    class_weights = torch.tensor(np.minimum(class_weights, 10.0), dtype=torch.float).to(DEVICE)
    logging.info(f"Using capped class weights: {class_weights.tolist()}")

    train_loader = DataLoader(
        PairedTileDataset(*all_data["train"], is_train=True, fixed_size=(IMAGE_SIZE, IMAGE_SIZE)),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, drop_last=True, collate_fn=paired_collate_fn
    )
    val_loader = DataLoader(
        PairedTileDataset(*all_data["val"], is_train=False, fixed_size=(IMAGE_SIZE, IMAGE_SIZE)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, collate_fn=paired_collate_fn
    )
    logging.info(f"Training set: {len(train_loader.dataset)} images | Validation set: {len(val_loader.dataset)} images.")

    # --- Model, Optimizer, and Loss ---
    model = PathologyFoundationModelPipeline(device=DEVICE, num_classes=len(class_names))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4, amsgrad=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.startswith("cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda e: linear_warmup_scheduler(e, WARMUP_EPOCHS))

    # --- Training State ---
    best_val_accuracy = 0.0
    epochs_no_improve = 0
    best_checkpoint_path, best_results_path = None, None

    # --- Main Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        for images, labels, _, _ in train_pbar:
            if images is None: continue # Skip failed batches
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
            with torch.autocast(device_type=DEVICE.split(':')[0], dtype=torch.float16, enabled=DEVICE.startswith("cuda")):
                logits = model(images)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            
            if (train_pbar.n + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            running_loss += loss.item()
            train_pbar.set_postfix(loss=running_loss / (train_pbar.n + 1))

        # --- Validation Loop ---
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_true_labels, all_pred_labels = [], []
        val_results = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)
            for images, labels, he_paths, fluor_paths in val_pbar:
                if images is None: continue
                images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                
                with torch.autocast(device_type=DEVICE.split(':')[0], dtype=torch.float16, enabled=DEVICE.startswith("cuda")):
                    logits = model(images)
                    loss = criterion(logits, labels)
                
                val_loss += loss.item()
                predicted_indices = torch.argmax(logits, dim=1)
                total += labels.size(0)
                correct += (predicted_indices == labels).sum().item()
                all_true_labels.extend(labels.cpu().numpy())
                all_pred_labels.extend(predicted_indices.cpu().numpy())
        
        val_accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
        
        report_str = classification_report(all_true_labels, all_pred_labels, target_names=class_names, zero_division=0, digits=3)
        logging.info(f"\nClassification Report for Epoch {epoch+1}:\n{report_str}")

        # --- Checkpointing and Early Stopping ---
        if val_accuracy > best_val_accuracy:
            logging.info(f"New best accuracy! {val_accuracy:.2f}% (previously {best_val_accuracy:.2f}%)")
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0

            if best_checkpoint_path and os.path.exists(best_checkpoint_path):
                os.remove(best_checkpoint_path)
            
            best_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_checkpoint_epoch_{epoch+1}_acc_{val_accuracy:.2f}.pth")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, best_checkpoint_path)
            logging.info(f"Saved new best checkpoint to {best_checkpoint_path}")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement for {epochs_no_improve} epoch(s). Best accuracy remains {best_val_accuracy:.2f}%.")
        
        if epochs_no_improve >= PATIENCE:
            logging.info(f"Early stopping triggered at epoch {epoch+1}. No improvement for {PATIENCE} epochs.")
            break
        
        # Update Learning Rate
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            scheduler.step()

    logging.info(f"\nTraining finished. Best validation accuracy: {best_val_accuracy:.2f}%")
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "final_model_state.pth"))

if __name__ == "__main__":
    main()

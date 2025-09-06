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

# --- Logging Setup ---
def setup_logging(log_file: str):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# --- Model Class for Fluorescence Images ---
class FluorescenceClassificationModel(nn.Module):
    def __init__(self, device: str, num_classes: int, feature_dim: int = 1280):
        super(FluorescenceClassificationModel, self).__init__()
        self.feature_dim, self.num_classes, self.device = feature_dim, num_classes, device
        
        # Feature extractor for fluorescence images
        self.extractor = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0).to(self.device)
        
        # Modify the first convolution layer to accept 1 channel (grayscale)
        first_conv = self.extractor.conv_stem
        new_first_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        ).to(self.device)
        
        # Initialize new weights by averaging the original pre-trained weights
        with torch.no_grad():
            new_first_conv.weight = nn.Parameter(first_conv.weight.mean(dim=1, keepdim=True))
            if new_first_conv.bias is not None:
                new_first_conv.bias.copy_(first_conv.bias)
        self.extractor.conv_stem = new_first_conv
        
        # Ensure all parameters are trainable
        for param in self.extractor.parameters():
            param.requires_grad = True
        
        # Classifier layer
        self.classifier = nn.Linear(feature_dim, num_classes).to(self.device)
        
        # Normalization for fluorescence images
        self.transforms = transforms.Normalize(mean=[0.174], std=[0.133])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if torch.isnan(images).any() or torch.isinf(images).any():
            logging.warning("NaN or Inf detected in input images.")
            return torch.zeros((images.size(0), self.num_classes), device=self.device)
        
        # Apply normalization
        normalized_images = self.transforms(images).to(self.device)
        
        # Use mixed precision for feature extraction
        with torch.autocast(device_type=self.device.split(':')[0], dtype=torch.float16):
            features = self.extractor(normalized_images)
        
        # Classifier runs in full precision
        logits = self.classifier(features.float())
        return logits

# --- Dataset Class for Fluorescence Images ---
class FluorescenceTileDataset(Dataset):
    def __init__(self, fluor_paths, labels, is_train=True, fixed_size=(512, 512)):
        self.fluor_paths, self.labels = fluor_paths, labels
        self.is_train = is_train
        
        # Define the transformation pipeline
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize(fixed_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1), # Saturation is irrelevant for grayscale
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(fixed_size),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.fluor_paths)

    def __getitem__(self, idx):
        try:
            fluor_path = self.fluor_paths[idx]
            fluor_img = Image.open(fluor_path).convert('L') # Convert to grayscale
            label = self.labels[idx]
            
            tensor = self.transform(fluor_img)
            
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                raise ValueError(f"Invalid values in tensor for image: {fluor_path}")
            
            return tensor, label, fluor_path
        except Exception as e:
            print(f"Error loading image {fluor_path}: {e}. Skipping.")
            return None

def fluor_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        raise ValueError("Empty batch after filtering invalid images.")
    
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    paths = [item[2] for item in batch]
    return images, labels, paths

# --- Warm-Up Scheduler ---
def linear_warmup_scheduler(epoch, warmup_epochs=10):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0

if __name__ == "__main__":
    # --- Setup ---
    TRAIN_FLUOR_PATH = "./f_subtype1"
    VAL_FLUOR_PATH = "./f_subtype"
    EPOCHS, BATCH_SIZE = 150, 32
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR = "checkpoints"
    LR = 1e-4
    ACCUMULATION_STEPS = 1
    WARMUP_EPOCHS = 10
    PATIENCE = 25

    torch.backends.cudnn.benchmark = True
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    LOG_FILE = os.path.join(CHECKPOINT_DIR, "training_log.txt")
    setup_logging(LOG_FILE)

    # --- Load Training Data ---
    print("Scanning training data...")
    train_fluor_paths, train_labels = [], []
    class_names = sorted([d for d in os.listdir(TRAIN_FLUOR_PATH) if os.path.isdir(os.path.join(TRAIN_FLUOR_PATH, d))])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    print(f"Found {len(class_names)} classes: {class_to_idx}")

    for cn, li in class_to_idx.items():
        fluor_class_path = os.path.join(TRAIN_FLUOR_PATH, cn)
        if not os.path.exists(fluor_class_path):
            print(f"Warning: Class directory missing: {fluor_class_path}")
            continue
        for img_name in os.listdir(fluor_class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                fluor_path = os.path.join(fluor_class_path, img_name)
                train_fluor_paths.append(fluor_path)
                train_labels.append(li)

    train_fluor_paths = np.array(train_fluor_paths)
    train_labels = np.array(train_labels)
    print(f"Found {len(train_fluor_paths)} training images.")

    # --- Load Validation Data ---
    print("Scanning validation data...")
    val_fluor_paths, val_labels = [], []
    for cn, li in class_to_idx.items():
        fluor_class_path = os.path.join(VAL_FLUOR_PATH, cn)
        if not os.path.exists(fluor_class_path):
            print(f"Warning: Class directory missing: {fluor_class_path}")
            continue
        for img_name in os.listdir(fluor_class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                fluor_path = os.path.join(fluor_class_path, img_name)
                val_fluor_paths.append(fluor_path)
                val_labels.append(li)

    val_fluor_paths = np.array(val_fluor_paths)
    val_labels = np.array(val_labels)
    print(f"Found {len(val_fluor_paths)} validation images.")

    if len(train_fluor_paths) == 0 or len(val_fluor_paths) == 0:
        raise ValueError("No data available for training or validation. Cannot proceed.")

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = np.minimum(class_weights, 10.0)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"Capped class weights: {class_weights.tolist()}")

    # Data Loaders
    train_loader = DataLoader(
        FluorescenceTileDataset(train_fluor_paths, train_labels, is_train=True, fixed_size=(512, 512)),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2, drop_last=True, collate_fn=fluor_collate_fn
    )
    val_loader = DataLoader(
        FluorescenceTileDataset(val_fluor_paths, val_labels, is_train=False, fixed_size=(512, 512)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2, collate_fn=fluor_collate_fn
    )

    print(f"Training set: {len(train_loader.dataset)} images.")
    print(f"Validation set: {len(val_loader.dataset)} images.")

    # Model and Training Setup
    model = FluorescenceClassificationModel(device=DEVICE, num_classes=len(class_names))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4, amsgrad=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.cuda.amp.GradScaler() if DEVICE.startswith("cuda") else None
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: linear_warmup_scheduler(epoch, WARMUP_EPOCHS))

    best_val_accuracy = 0.0
    best_checkpoint_path = None
    best_results_path = None
    epochs_no_improve = 0

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss, epoch_start_time = 0.0, time.time()
        # ... (The rest of the training and validation loop is identical, just updated for clarity)
        data_time, compute_time = 0.0, 0.0
        optimizer.zero_grad(set_to_none=True)

        for i, (images, labels, _) in enumerate(train_loader):
            data_start = time.time()
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            data_time += time.time() - data_start
            
            compute_start = time.time()
            with torch.autocast(device_type=DEVICE.split(':')[0], dtype=torch.float16, enabled=DEVICE.startswith("cuda")):
                logits = model(images)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logging.warning(f"NaN or Inf in logits at batch {i+1}. Skipping batch.")
                    continue
                loss = criterion(logits, labels) / ACCUMULATION_STEPS
            
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN or Inf in loss at batch {i+1}. Skipping batch.")
                continue
            
            if scaler:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            running_loss += loss.item() * ACCUMULATION_STEPS
            compute_time += time.time() - compute_start

            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                optimizer.zero_grad(set_to_none=True)
                if (i + 1) % 10 == 0:
                    print(f"\rEpoch {epoch+1}/{EPOCHS}, Batch {i+1}/{len(train_loader)}, Loss: {running_loss / (i+1):.4f}", end='')

        avg_train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else float('nan')
        epoch_time = time.time() - epoch_start_time
        log_message = (f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Epoch Time: {epoch_time:.2f}s, "
                       f"Data Time: {data_time:.2f}s, Compute Time: {compute_time:.2f}s")
        logging.info(log_message)

        # Evaluation on Validation Dataset
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        val_results = []
        all_true_labels, all_pred_labels = [], []
        with torch.no_grad():
            for images, labels, paths in val_loader:
                images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                with torch.autocast(device_type=DEVICE.split(':')[0], dtype=torch.float16, enabled=DEVICE.startswith("cuda")):
                    logits = model(images)
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        logging.warning(f"NaN or Inf in validation logits. Skipping batch.")
                        continue
                    loss = criterion(logits, labels)
                val_loss += loss.item()
                probabilities = F.softmax(logits, dim=1)
                predicted_indices = torch.argmax(probabilities, dim=1)
                total += labels.size(0)
                correct += (predicted_indices == labels).sum().item()
                all_true_labels.extend(labels.cpu().numpy())
                all_pred_labels.extend(predicted_indices.cpu().numpy())
                for i in range(len(paths)):
                    true_idx, pred_idx = labels[i].item(), predicted_indices[i].item()
                    result_record = {
                        "fluor_path": paths[i],
                        "true_label": idx_to_class[true_idx],
                        "predicted_label": idx_to_class[pred_idx],
                        "is_correct": true_idx == pred_idx,
                        "confidence": probabilities[i][pred_idx].item()
                    }
                    for class_idx, class_name in idx_to_class.items():
                        result_record[f"prob_{class_name}"] = probabilities[i][class_idx].item()
                    val_results.append(result_record)

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('nan')
        val_accuracy = 100 * correct / total if total > 0 else 0.0
        if (epoch + 1) % 5 == 0 or val_accuracy > best_val_accuracy:
            print(f"\nEpoch {epoch+1} Evaluation on Validation Dataset:")
            print(classification_report(all_true_labels, all_pred_labels, target_names=class_names, zero_division=0))
        print(f"Epoch {epoch+1} Validation | Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.2f}%")
        log_message = f"Epoch {epoch+1} Validation: Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%"
        logging.info(log_message)

        # Save checkpoint and results if best accuracy
        if val_accuracy > best_val_accuracy:
            if best_checkpoint_path and os.path.exists(best_checkpoint_path): os.remove(best_checkpoint_path)
            if best_results_path and os.path.exists(best_results_path): os.remove(best_results_path)

            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            best_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_checkpoint_epoch_{epoch+1}_acc_{val_accuracy:.2f}.pth")
            checkpoint_dict = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
            if scaler: checkpoint_dict['scaler_state_dict'] = scaler.state_dict()
            torch.save(checkpoint_dict, best_checkpoint_path)

            best_results_path = os.path.join(CHECKPOINT_DIR, f"best_validation_results_epoch_{epoch+1}_acc_{val_accuracy:.2f}.csv")
            headers = ["fluor_path", "true_label", "predicted_label", "is_correct", "confidence"] + [f"prob_{name}" for name in class_names]
            with open(best_results_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                writer.writerows(val_results)
            print(f"New best accuracy {val_accuracy:.2f}% at epoch {epoch+1}. Saved checkpoint and validation results.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} as validation accuracy has not improved for {PATIENCE} epochs.")
                break

        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            scheduler.step()

    print(f"\nTraining finished. Best validation accuracy: {best_val_accuracy:.2f}%")
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "last_epoch_pipeline.pth"))

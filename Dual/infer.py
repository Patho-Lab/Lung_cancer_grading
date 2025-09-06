import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset, DataLoader
import csv
import logging
from tqdm import tqdm
from pathlib import Path

# --- Logging Setup ---
def setup_logging(log_file: str):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# --- Model Definition ---
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

class PathologyFoundationModelPipeline(nn.Module):
    def __init__(self, device: str, num_classes: int, feature_dim: int = 1280):
        super(PathologyFoundationModelPipeline, self).__init__()
        self.feature_dim, self.num_classes, self.device = feature_dim, num_classes, device
        self.rgb_extractor = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0, global_pool='').to(self.device)
        self.fluor_extractor = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0, global_pool='').to(self.device)
        
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

        fusion_channels = 512
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim * 2, fusion_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
            SELayer(channel=fusion_channels)
        ).to(self.device)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(fusion_channels, num_classes).to(self.device)
        
        self.rgb_transforms = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.fluor_transforms = transforms.Normalize(mean=[0.174], std=[0.133])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
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

# --- Dataset and Dataloader ---
class InferencePairedTileDataset(Dataset):
    def __init__(self, he_paths, fluor_paths, fixed_size=(512, 512)):
        self.he_paths, self.fluor_paths = he_paths, fluor_paths
        self.transforms = transforms.Compose([
            transforms.Resize(fixed_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.he_paths)

    def __getitem__(self, idx):
        try:
            he_img = Image.open(self.he_paths[idx]).convert('RGB')
            fluor_img = Image.open(self.fluor_paths[idx]).convert('L')
            he_tensor = self.transforms(he_img)
            fluor_tensor = self.transforms(fluor_img)
            combined_tensor = torch.cat((he_tensor, fluor_tensor), dim=0)
            return combined_tensor, self.he_paths[idx]
        except Exception as e:
            print(f"\nWarning: Error loading image pair {self.he_paths[idx]}, {self.fluor_paths[idx]}: {e}. Skipping.")
            return None

def inference_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None, None
    images = torch.stack([item[0] for item in batch])
    he_paths = [item[1] for item in batch]
    return images, he_paths

# --- Main Inference Script ---
if __name__ == "__main__":
    # --- Configuration ---
    INFERENCE_HE_PATH = ""
    INFERENCE_FLUOR_PATH = ""
    CHECKPOINT_PATH = ""
    OUTPUT_DIR = "./inference_results"
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    IMAGE_SIZE = 512
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    LOG_FILE = os.path.join(OUTPUT_DIR, "inference_log.txt")
    setup_logging(LOG_FILE)
    
    # --- Class Definitions ---
    class_names = ['acinar', 'complex', 'lepidic', 'micropapillary', 'normal', 'papillary', 'solid']
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    
    display_name_map = {
        'acinar': 'Aci.', 'complex': 'Com.', 'lepidic': 'Lep.',
        'micropapillary': 'Mic.', 'papillary': 'Pap.',
        'solid': 'Sol.', 'normal': 'Peri.'
    }
    display_class_names = [display_name_map.get(name, name) for name in class_names]
    logging.info(f"Display class names: {display_class_names}")

    # --- Model Loading ---
    model = PathologyFoundationModelPipeline(device=DEVICE, num_classes=len(class_names))
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    logging.info(f"Loaded model from {CHECKPOINT_PATH}")
    
    # --- Find WSI Subfolders ---
    wsi_subfolders = [d for d in os.listdir(INFERENCE_HE_PATH) if os.path.isdir(os.path.join(INFERENCE_HE_PATH, d))]
    if not wsi_subfolders:
        logging.error(f"No WSI subfolders found in {INFERENCE_HE_PATH}")
        exit(1)
    logging.info(f"Found {len(wsi_subfolders)} WSI subfolders: {wsi_subfolders}")

    # --- Supported Image Extensions ---
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

    # --- Process Each WSI ---
    for wsi_folder in wsi_subfolders:
        logging.info(f"Processing WSI: {wsi_folder}")
        he_wsi_path = os.path.join(INFERENCE_HE_PATH, wsi_folder)
        fluor_wsi_path = os.path.join(INFERENCE_FLUOR_PATH, wsi_folder)
        
        if not os.path.exists(fluor_wsi_path):
            logging.warning(f"Fluorescence folder {fluor_wsi_path} does not exist. Skipping WSI {wsi_folder}.")
            continue
        
        # Load tiles for this WSI
        inf_he_paths, inf_fluor_paths = [], []
        # Get all fluorescence files and their stems
        fluor_files = {Path(f).stem: os.path.join(fluor_wsi_path, f) for f in os.listdir(fluor_wsi_path)
                       if f.lower().endswith(supported_extensions)}
        
        # Match H&E files with fluorescence files based on stem
        for img_name in os.listdir(he_wsi_path):
            if img_name.lower().endswith(supported_extensions):
                he_path = os.path.join(he_wsi_path, img_name)
                img_stem = Path(img_name).stem
                if img_stem in fluor_files:
                    inf_he_paths.append(he_path)
                    inf_fluor_paths.append(fluor_files[img_stem])
                else:
                    logging.warning(f"No matching fluorescence image found for {he_path}. Skipping.")
        
        if not inf_he_paths:
            logging.warning(f"No valid image pairs found for WSI {wsi_folder}. Skipping.")
            continue
        
        # Create DataLoader for this WSI
        inf_dataset = InferencePairedTileDataset(inf_he_paths, inf_fluor_paths, fixed_size=(IMAGE_SIZE, IMAGE_SIZE))
        inf_loader = DataLoader(
            inf_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
            pin_memory=True, prefetch_factor=2, collate_fn=inference_collate_fn
        )
        logging.info(f"Loaded {len(inf_dataset)} images for WSI {wsi_folder}")
        
        # --- Inference Loop ---
        inf_results = []
        with torch.no_grad():
            pbar = tqdm(inf_loader, desc=f"Inferring WSI {wsi_folder}")
            for images, he_paths in pbar:
                if images is None: continue
                images = images.to(DEVICE, non_blocking=True)
                
                with torch.autocast(device_type=DEVICE.split(':')[0], dtype=torch.float16, enabled=DEVICE.startswith("cuda")):
                    logits = model(images)
                    probabilities = F.softmax(logits, dim=1)
                    predicted_indices = torch.argmax(probabilities, dim=1)
                
                for i in range(len(he_paths)):
                    pred_idx = predicted_indices[i].item()
                    inf_results.append({
                        "he_path": he_paths[i],
                        "predicted_label": display_name_map.get(idx_to_class[pred_idx]),
                        "confidence": round(probabilities[i][pred_idx].item(), 4)
                    })
        
        # --- Save CSV for this WSI ---
        output_csv = os.path.join(OUTPUT_DIR, f"{wsi_folder}_inference_results.csv")
        headers = ["he_path", "predicted_label", "confidence"]
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(inf_results)
        logging.info(f"Inference completed for WSI {wsi_folder}. Results saved to {output_csv}")

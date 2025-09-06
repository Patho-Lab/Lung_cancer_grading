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
    """Clears existing handlers and sets up logging to a file and the console."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# --- Model Definition (Must be identical to train.py) ---
class PathologyClassificationModel(nn.Module):
    def __init__(self, device: str, num_classes: int, feature_dim: int = 1280):
        super(PathologyClassificationModel, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.device = device
        
        # Single feature extractor for RGB images
        self.extractor = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0).to(self.device)
        
        # Classifier
        self.classifier = nn.Linear(self.feature_dim, self.num_classes).to(self.device)
        
        # Standard normalization for RGB images
        self.transforms = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Normalize the images
        normalized_images = self.transforms(images).to(self.device)
        
        # Use mixed precision for feature extraction
        with torch.autocast(device_type=self.device.split(':')[0], dtype=torch.float16):
            features = self.extractor(normalized_images)
        
        # Classifier runs in full precision
        logits = self.classifier(features.float())
        return logits

# --- Dataset and Dataloader for Single Image Inference ---
class InferenceSingleImageDataset(Dataset):
    def __init__(self, image_paths, fixed_size=(512, 512)):
        self.image_paths = image_paths
        # Simple transforms for inference: Resize and convert to tensor. Normalization is in the model.
        self.transform = transforms.Compose([
            transforms.Resize(fixed_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            tensor = self.transform(image)
            return tensor, image_path
        except Exception as e:
            print(f"\nWarning: Error loading image {self.image_paths[idx]}: {e}. Skipping.")
            return None

def inference_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None, None
    images = torch.stack([item[0] for item in batch])
    paths = [item[1] for item in batch]
    return images, paths

# --- Main Inference Script ---
if __name__ == "__main__":
    # --- Configuration ---
    # UPDATE THESE PATHS
    INFERENCE_IMAGE_PATH = ""  # Path to the parent folder containing WSI subfolders
    CHECKPOINT_PATH = "./checkpoints" # Path to the trained model checkpoint
    OUTPUT_DIR = ""
    
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    IMAGE_SIZE = 512
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    LOG_FILE = os.path.join(OUTPUT_DIR, "inference_log.txt")
    setup_logging(LOG_FILE)
    
    # --- Class Definitions (Must match training script) ---
    class_names = ['acinar', 'complex', 'lepidic', 'micropapillary', 'normal', 'papillary', 'solid']
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    
    # Optional: for shorter names in the output CSV
    display_name_map = {
        'acinar': 'Aci.', 'complex': 'Com.', 'lepidic': 'Lep.',
        'micropapillary': 'Mic.', 'papillary': 'Pap.',
        'solid': 'Sol.', 'normal': 'Peri.'
    }
    logging.info(f"Using class mapping: {idx_to_class}")

    # --- Model Loading ---
    model = PathologyClassificationModel(device=DEVICE, num_classes=len(class_names))
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    # Handle both dictionary-based checkpoints and raw state dicts
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    logging.info(f"Loaded model from {CHECKPOINT_PATH}")
    
    # --- Find WSI Subfolders ---
    wsi_subfolders = [d for d in os.listdir(INFERENCE_IMAGE_PATH) if os.path.isdir(os.path.join(INFERENCE_IMAGE_PATH, d))]
    if not wsi_subfolders:
        logging.error(f"No WSI subfolders found in {INFERENCE_IMAGE_PATH}")
        exit(1)
    logging.info(f"Found {len(wsi_subfolders)} WSI subfolders to process.")

    # --- Supported Image Extensions ---
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

    # --- Process Each WSI ---
    for wsi_folder in sorted(wsi_subfolders):
        logging.info(f"Processing WSI: {wsi_folder}")
        wsi_tile_path = os.path.join(INFERENCE_IMAGE_PATH, wsi_folder)
        
        # Load all image tiles for this WSI
        inf_image_paths = [
            os.path.join(wsi_tile_path, f) 
            for f in os.listdir(wsi_tile_path) 
            if f.lower().endswith(supported_extensions)
        ]
        
        if not inf_image_paths:
            logging.warning(f"No image files found for WSI {wsi_folder}. Skipping.")
            continue
        
        # Create DataLoader for this specific WSI
        inf_dataset = InferenceSingleImageDataset(inf_image_paths, fixed_size=(IMAGE_SIZE, IMAGE_SIZE))
        inf_loader = DataLoader(
            inf_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
            pin_memory=True, prefetch_factor=2, collate_fn=inference_collate_fn
        )
        logging.info(f"Loaded {len(inf_dataset)} image tiles for WSI {wsi_folder}")
        
        # --- Inference Loop for the current WSI ---
        inf_results = []
        with torch.no_grad():
            pbar = tqdm(inf_loader, desc=f"Inferring WSI {wsi_folder}")
            for images, paths in pbar:
                if images is None: continue # Skip bad batches
                images = images.to(DEVICE, non_blocking=True)
                
                with torch.autocast(device_type=DEVICE.split(':')[0], dtype=torch.float16, enabled=DEVICE.startswith("cuda")):
                    logits = model(images)
                    probabilities = F.softmax(logits, dim=1)
                    predicted_indices = torch.argmax(probabilities, dim=1)
                
                # Record results for each image in the batch
                for i in range(len(paths)):
                    pred_idx = predicted_indices[i].item()
                    predicted_class_name = idx_to_class[pred_idx]
                    inf_results.append({
                        "image_path": paths[i],
                        "predicted_label": display_name_map.get(predicted_class_name, predicted_class_name),
                        "confidence": round(probabilities[i][pred_idx].item(), 4)
                    })
        
        # --- Save CSV for this WSI ---
        output_csv = os.path.join(OUTPUT_DIR, f"{wsi_folder}_inference_results.csv")
        headers = ["image_path", "predicted_label", "confidence"]
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(inf_results)
        logging.info(f"Inference completed for WSI {wsi_folder}. Results saved to {output_csv}")

    logging.info("All WSIs have been processed.")

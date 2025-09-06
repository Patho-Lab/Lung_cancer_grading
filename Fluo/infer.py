import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset, DataLoader
import csv
import logging
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
# Required for confidence intervals
from statsmodels.stats.proportion import proportion_confint

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

# --- Model Class (to match the trained model's architecture) ---
class FluorescenceClassificationModel(nn.Module):
    def __init__(self, device: str, num_classes: int, feature_dim: int = 1280):
        super(FluorescenceClassificationModel, self).__init__()
        self.feature_dim, self.num_classes, self.device = feature_dim, num_classes, device
        
        self.extractor = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0).to(self.device)
        
        first_conv = self.extractor.conv_stem
        new_first_conv = nn.Conv2d(
            in_channels=1, out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size, stride=first_conv.stride,
            padding=first_conv.padding, bias=first_conv.bias is not None
        ).to(self.device)
        with torch.no_grad():
            new_first_conv.weight = nn.Parameter(first_conv.weight.mean(dim=1, keepdim=True))
            if new_first_conv.bias is not None:
                new_first_conv.bias.copy_(first_conv.bias)
        self.extractor.conv_stem = new_first_conv
        
        self.classifier = nn.Linear(feature_dim, num_classes).to(self.device)
        self.transforms = transforms.Normalize(mean=[0.174], std=[0.133])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        normalized_images = self.transforms(images).to(self.device)
        with torch.autocast(device_type=self.device.split(':')[0], dtype=torch.float16):
            features = self.extractor(normalized_images)
        logits = self.classifier(features.float())
        return logits

# --- Inference Dataset Class ---
class InferenceFluorescenceTileDataset(Dataset):
    def __init__(self, fluor_paths, labels, fixed_size=(512, 512)):
        self.fluor_paths, self.labels = fluor_paths, labels
        self.transform = transforms.Compose([
            transforms.Resize(fixed_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.fluor_paths)

    def __getitem__(self, idx):
        try:
            fluor_path = self.fluor_paths[idx]
            fluor_img = Image.open(fluor_path).convert('L')
            label = self.labels[idx]
            tensor = self.transform(fluor_img)
            return tensor, label, fluor_path
        except Exception as e:
            print(f"Error loading image {fluor_path}: {e}. Skipping.")
            return None

def inference_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: raise ValueError("Empty batch after filtering invalid images.")
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    paths = [item[2] for item in batch]
    return images, labels, paths

# --- Reporting Functions ---
def plot_multiclass_roc_auc(true_labels, pred_scores, display_class_names, output_dir):
    n_classes = len(display_class_names)
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        y_true_binary = (true_labels == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(y_true_binary, pred_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    plt.figure(figsize=(14, 12))
    colors = plt.cm.get_cmap('tab10', n_classes)
    for i, color in zip(range(n_classes), colors.colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2.5, label=f'{display_class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2.5)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=28)
    plt.ylabel('True Positive Rate', fontsize=28)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(loc="lower right", fontsize=25)
    plt.tight_layout()
    
    # REVISED: Save the plot as an SVG file
    roc_plot_path = os.path.join(output_dir, "roc_auc_curves.svg")
    plt.savefig(roc_plot_path, format='svg'); plt.close()
    print(f"\nROC/AUC curve plot saved to {roc_plot_path}")

# MODIFIED: Function updated to calculate F1-Score CI via bootstrapping
def generate_and_save_report_with_ci(true_labels, pred_labels, cm, display_class_names, output_dir, n_bootstraps=1000, ci_level=0.95):
    report_data = []
    print("\n" + "="*120); print("Classification Report with 95% Confidence Intervals"); print("="*120)
    header = f"{'Class':<18} | {'Precision (95% CI)':^25} | {'Recall (95% CI)':^25} | {'F1-Score (95% CI)':^25}"
    print(header); print("-" * len(header))

    n_samples = len(true_labels)

    for i, name in enumerate(display_class_names):
        # Point estimates from the full dataset
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = tp + fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # CIs for Precision and Recall (analytical)
        prec_ci_low, prec_ci_high = proportion_confint(count=tp, nobs=tp + fp, method='wilson')
        rec_ci_low, rec_ci_high = proportion_confint(count=tp, nobs=tp + fn, method='wilson')
        
        # Bootstrap CI for F1-score
        bootstrapped_f1_scores = []
        for _ in range(n_bootstraps):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_true = true_labels[indices]
            boot_pred = pred_labels[indices]
            
            b_tp = np.sum((boot_true == i) & (boot_pred == i))
            b_fp = np.sum((boot_true != i) & (boot_pred == i))
            b_fn = np.sum((boot_true == i) & (boot_pred != i))
            
            b_precision = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0
            b_recall = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0
            
            b_f1 = 2 * (b_precision * b_recall) / (b_precision + b_recall) if (b_precision + b_recall) > 0 else 0
            bootstrapped_f1_scores.append(b_f1)
            
        alpha = (1.0 - ci_level) * 100
        f1_ci_low = np.percentile(bootstrapped_f1_scores, alpha / 2.0)
        f1_ci_high = np.percentile(bootstrapped_f1_scores, 100 - (alpha / 2.0))

        prec_str = f"{precision:.3f} ({prec_ci_low:.3f}, {prec_ci_high:.3f})"
        rec_str = f"{recall:.3f} ({rec_ci_low:.3f}, {rec_ci_high:.3f})"
        f1_str = f"{f1_score:.3f} ({f1_ci_low:.3f}, {f1_ci_high:.3f})"
        
        print(f"{name:<18} | {prec_str:^25} | {rec_str:^25} | {f1_str:^25}")
        
        report_data.append({
            "Class": name, "Precision": f"{precision:.3f}", "Precision_CI_Lower": f"{prec_ci_low:.3f}",
            "Precision_CI_Upper": f"{prec_ci_high:.3f}", "Recall": f"{recall:.3f}", "Recall_CI_Lower": f"{rec_ci_low:.3f}",
            "Recall_CI_Upper": f"{rec_ci_high:.3f}", "F1-Score": f"{f1_score:.3f}", 
            "F1-Score_CI_Lower": f"{f1_ci_low:.3f}", "F1-Score_CI_Upper": f"{f1_ci_high:.3f}", "Support": support
        })

    total_correct, total_samples = np.sum(np.diag(cm)), np.sum(cm)
    accuracy = total_correct / total_samples
    acc_ci_low, acc_ci_high = proportion_confint(count=total_correct, nobs=total_samples, method='wilson')
    
    print("-" * len(header))
    print(f"Overall Accuracy: {accuracy:.3f} (95% CI: {acc_ci_low:.3f}, {acc_ci_high:.3f})")
    print("="*120)

    report_path = os.path.join(output_dir, "classification_report_with_ci.csv")
    csv_headers = ["Class", "Precision", "Precision_CI_Lower", "Precision_CI_Upper", "Recall", "Recall_CI_Lower", "Recall_CI_Upper", "F1-Score", "F1-Score_CI_Lower", "F1-Score_CI_Upper", "Support"]
    with open(report_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader(); writer.writerows(report_data)
        writer.writerow({})
        writer.writerow({
            "Class": "Overall Accuracy", "Precision": f"{accuracy:.3f}", "Precision_CI_Lower": f"{acc_ci_low:.3f}",
            "Precision_CI_Upper": f"{acc_ci_high:.3f}", "Support": total_samples
        })
    print(f"\nClassification report with CIs saved to {report_path}")

# --- Main Inference Script ---
if __name__ == "__main__":
    INFERENCE_FLUOR_PATH = "./f_subtype"
    CHECKPOINT_PATH = ""
    OUTPUT_DIR = "./inference"
    BATCH_SIZE = 32
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    LOG_FILE = os.path.join(OUTPUT_DIR, "inference_log.txt")
    setup_logging(LOG_FILE)
    
    class_names = sorted([d for d in os.listdir(INFERENCE_FLUOR_PATH) if os.path.isdir(os.path.join(INFERENCE_FLUOR_PATH, d))])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    
    display_name_map = {
        'acinar': 'Aci.', 'complex': 'Com.', 'lepidic': 'Lep.',
        'micropapillary': 'Mic.', 'papillary': 'Pap.',
        'solid': 'Sol.', 'normal': 'Peri.'
    }
    display_class_names = [display_name_map.get(name, name) for name in class_names]
    print(f"Display class names: {display_class_names}")

    print("Scanning inference data...")
    inf_fluor_paths, inf_labels = [], []
    for cn, li in class_to_idx.items():
        class_path = os.path.join(INFERENCE_FLUOR_PATH, cn)
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                inf_fluor_paths.append(os.path.join(class_path, img_name))
                inf_labels.append(li)
    
    inf_loader = DataLoader(
        InferenceFluorescenceTileDataset(inf_fluor_paths, inf_labels, fixed_size=(512, 512)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=8,
        pin_memory=True, prefetch_factor=2, collate_fn=inference_collate_fn
    )
    
    model = FluorescenceClassificationModel(device=DEVICE, num_classes=len(class_names))
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {CHECKPOINT_PATH}")
    
    all_true_labels, all_pred_labels, all_pred_scores, inf_results = [], [], [], []
    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(inf_loader):
            images = images.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type=DEVICE.split(':')[0], dtype=torch.float16, enabled=DEVICE.startswith("cuda")):
                probabilities = F.softmax(model(images), dim=1)
                predicted_indices = torch.argmax(probabilities, dim=1)
            
            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(predicted_indices.cpu().numpy())
            all_pred_scores.extend(probabilities.cpu().numpy())
            
            for i in range(len(paths)):
                pred_idx, true_idx = predicted_indices[i].item(), labels[i].item()
                result_record = {
                    "fluor_path": paths[i],
                    "true_label": display_name_map.get(idx_to_class[true_idx]),
                    "predicted_label": display_name_map.get(idx_to_class[pred_idx]),
                    "confidence": round(probabilities[i][pred_idx].item(), 3)
                }
                for class_idx, class_name in idx_to_class.items():
                    result_record[f"prob_{display_name_map.get(class_name)}"] = round(probabilities[i][class_idx].item(), 3)
                inf_results.append(result_record)
            print(f"\rProcessed batch {batch_idx+1}/{len(inf_loader)}", end='')
    
    output_csv = os.path.join(OUTPUT_DIR, "inference_results.csv")
    headers = ["fluor_path", "true_label", "predicted_label", "confidence"] + [f"prob_{name}" for name in display_class_names]
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader(); writer.writerows(inf_results)
    print(f"\nInference completed. Detailed predictions saved to {output_csv}")
    
    all_true_labels, all_pred_labels, all_pred_scores = np.array(all_true_labels), np.array(all_pred_labels), np.array(all_pred_scores)
    
    print("\n" + "="*80); print("Standard Classification Report (from scikit-learn)"); print("="*80)
    print(classification_report(all_true_labels, all_pred_labels, target_names=display_class_names, zero_division=0, digits=3))
    
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    
    # MODIFIED: Updated function call to pass necessary arrays for bootstrapping
    generate_and_save_report_with_ci(all_true_labels, all_pred_labels, cm, display_class_names, OUTPUT_DIR)
    
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=display_class_names, yticklabels=display_class_names,
                     annot_kws={"size": 25})
                     
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=25)                    
    
    ax.set_xlabel('Predicted', fontsize=28)
    ax.set_ylabel('Ground Truth', fontsize=28)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=25)
    plt.setp(ax.get_yticklabels(), fontsize=25)

    plt.tight_layout()
    
    # REVISED: Save the plot as an SVG file
    cm_plot_path = os.path.join(OUTPUT_DIR, "confusion_matrix.svg")
    plt.savefig(cm_plot_path, format='svg'); plt.close()
    print(f"\nConfusion matrix plot saved to {cm_plot_path}")

    plot_multiclass_roc_auc(all_true_labels, all_pred_scores, display_class_names, OUTPUT_DIR)

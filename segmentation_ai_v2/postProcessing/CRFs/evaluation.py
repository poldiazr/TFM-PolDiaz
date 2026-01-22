import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

VERTEBRA_NAMES = {
    0: "background",1: "sacrum",2: "vertebrae_S1",3: "vertebrae_L5",4: "vertebrae_L4",5: "vertebrae_L3",6: "vertebrae_L2",7: "vertebrae_L1",
    8: "vertebrae_T12",9: "vertebrae_T11",10: "vertebrae_T10",11: "vertebrae_T9",12: "vertebrae_T8",13: "vertebrae_T7",14: "vertebrae_T6",
    15: "vertebrae_T5",16: "vertebrae_T4",17: "vertebrae_T3",18: "vertebrae_T2",19: "vertebrae_T1",20: "vertebrae_C7",21: "vertebrae_C6",
    22: "vertebrae_C5",23: "vertebrae_C4",24: "vertebrae_C3",25: "vertebrae_C2",26: "vertebrae_C1"
}

# METRICS
def dice_per_label(gt, pred, labels):
    scores = {}
    for lab in labels:
        gt_bin = (gt == lab)
        pred_bin = (pred == lab)
        inter = np.sum(gt_bin & pred_bin)
        denom = np.sum(gt_bin) + np.sum(pred_bin)
        scores[lab] = 2 * inter / (denom + 1e-8)
    return scores

def iou_per_label(gt, pred, labels):
    scores = {}
    for lab in labels:
        gt_bin = (gt == lab)
        pred_bin = (pred == lab)
        inter = np.sum(gt_bin & pred_bin)
        union = np.sum(gt_bin | pred_bin)
        scores[lab] = inter / (union + 1e-8)
    return scores

def precision_per_label(gt, pred, labels):
    scores = {}
    for lab in labels:
        gt_bin = (gt == lab)
        pred_bin = (pred == lab)
        tp = np.sum(gt_bin & pred_bin)
        fp = np.sum((~gt_bin) & pred_bin)
        scores[lab] = tp / (tp + fp + 1e-8)
    return scores

def recall_per_label(gt, pred, labels):
    scores = {}
    for lab in labels:
        gt_bin = (gt == lab)
        pred_bin = (pred == lab)
        tp = np.sum(gt_bin & pred_bin)
        fn = np.sum(gt_bin & (~pred_bin))
        scores[lab] = tp / (tp + fn + 1e-8)
    return scores

def diff_map_per_label(gt, pred, labels):
    maps = {}
    for lab in labels:
        gt_bin = (gt == lab)
        pred_bin = (pred == lab)
        diff = np.zeros_like(gt, dtype=np.uint8)
        diff[(pred_bin == True) & (gt_bin == False)] = 1  # FP
        diff[(pred_bin == False) & (gt_bin == True)] = 2  # FN
        maps[lab] = diff
    return maps

def metric_stats(metric_dict):
    vals = list(metric_dict.values())
    return np.mean(vals), np.std(vals)


# PLOTS
def save_combined_diff_plot(ct, pred, crf, diff_pred, diff_crf, labels, metrics_pred, metrics_crf, out_path):
    mid = ct.shape[1] // 2

    def combine_diff(diff_map_dict, pred_mask_dict):
        combined = np.zeros_like(diff_map_dict[labels[0]], dtype=np.uint8)
        for lab in labels:
            pred_bin = pred_mask_dict[lab]
            combined[(diff_map_dict[lab] == 0) & pred_bin] = 1  
            combined[diff_map_dict[lab] == 1] = 2  
            combined[diff_map_dict[lab] == 2] = 3  
        return combined

    pred_mask_dict_pred = {lab: (pred == lab) for lab in labels}
    pred_mask_dict_crf  = {lab: (crf == lab) for lab in labels}

    combined_pred = combine_diff(diff_pred, pred_mask_dict_pred)
    combined_crf  = combine_diff(diff_crf, pred_mask_dict_crf)

    cmap = ListedColormap(['none','#C7FFC4', 'red', 'blue'])
    legend_elements = [
        Patch(facecolor='#C7FFC4', label='Correct Prediction'),
        Patch(facecolor='red', label='False Positive'),
        Patch(facecolor='blue', label='False Negative')
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 22), gridspec_kw={'height_ratios': [8, 8, 6]})

    mid = 95

    # Figure 1: prediction no CRF
    axes[0].imshow(ct[:, :, mid], cmap='gray')
    axes[0].imshow(combined_pred[:, :, mid], cmap=cmap, alpha=0.5)
    axes[0].set_title(
        f"Prediction (no CRF)\n"
        f"Dice={metrics_pred['dice_mean']:.3f}±{metrics_pred['dice_std']:.3f}, "
        f"IoU={metrics_pred['iou_mean']:.3f}±{metrics_pred['iou_std']:.3f}, "
        f"Precision={metrics_pred['precision_mean']:.3f}±{metrics_pred['precision_std']:.3f}, "
        f"Recall={metrics_pred['recall_mean']:.3f}±{metrics_pred['recall_std']:.3f}",
        fontsize=11
    )
    axes[0].axis('off')
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Figure 2: prediction with CRF
    axes[1].imshow(ct[:, :, mid], cmap='gray')
    axes[1].imshow(combined_crf[:, :, mid], cmap=cmap, alpha=0.5)
    axes[1].set_title(
        f"Prediction with CRF\n"
        f"Dice={metrics_crf['dice_mean']:.3f}±{metrics_crf['dice_std']:.3f}, "
        f"IoU={metrics_crf['iou_mean']:.3f}±{metrics_crf['iou_std']:.3f}, "
        f"Precision={metrics_crf['precision_mean']:.3f}±{metrics_crf['precision_std']:.3f}, "
        f"Recall={metrics_crf['recall_mean']:.3f}±{metrics_crf['recall_std']:.3f}",
        fontsize=11
    )
    axes[1].axis('off')
    axes[1].legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Figure 3: Metrics per label
    table_text = ""
    for lab in labels:
        name = VERTEBRA_NAMES[lab]
        table_text += (
            f"{name}:   "
            f"Dice {metrics_pred['dice_per_label'][lab]:.3f} / {metrics_crf['dice_per_label'][lab]:.3f}   "
            f"IoU {metrics_pred['iou_per_label'][lab]:.3f} / {metrics_crf['iou_per_label'][lab]:.3f}   "
            f"Prec {metrics_pred['precision_per_label'][lab]:.3f} / {metrics_crf['precision_per_label'][lab]:.3f}   "
            f"Rec {metrics_pred['recall_per_label'][lab]:.3f} / {metrics_crf['recall_per_label'][lab]:.3f}\n"
        )

    axes[2].text(
        0.01, 0.99,
        "Per-vertebra metrics (Pred / CRF)\n\n" + table_text,
        fontsize=10,
        va='top',
        family='monospace'
    )
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[OK] Saved → {out_path}")



# Pipeline
def evaluate_multilabel_with_crf(gt_path, pred_path, crf_path, ct_path, out_path):
    gt = nib.load(gt_path).get_fdata().astype(np.int32)
    pred = nib.load(pred_path).get_fdata().astype(np.int32)
    crf = nib.load(crf_path).get_fdata().astype(np.int32)
    ct = nib.load(ct_path).get_fdata().astype(np.float32)
    ct = (ct - np.min(ct)) / (np.max(ct) - np.min(ct) + 1e-8)

    labels = np.unique(gt)
    labels = labels[labels != 0]
    labels = [lab for lab in labels 
            if np.any(pred == lab) and np.any(crf == lab)]
    print("Labels present:", [VERTEBRA_NAMES[lab] for lab in labels])

    # Metrics per label
    dice_pred = dice_per_label(gt, pred, labels)
    iou_pred = iou_per_label(gt, pred, labels)
    prec_pred = precision_per_label(gt, pred, labels)
    rec_pred = recall_per_label(gt, pred, labels)

    dice_crf = dice_per_label(gt, crf, labels)
    iou_crf = iou_per_label(gt, crf, labels)
    prec_crf = precision_per_label(gt, crf, labels)
    rec_crf = recall_per_label(gt, crf, labels)

    # Global mean + std
    metrics_pred = {
        "dice_mean": np.mean(list(dice_pred.values())),
        "dice_std": np.std(list(dice_pred.values())),
        "iou_mean": np.mean(list(iou_pred.values())),
        "iou_std": np.std(list(iou_pred.values())),
        "precision_mean": np.mean(list(prec_pred.values())),
        "precision_std": np.std(list(prec_pred.values())),
        "recall_mean": np.mean(list(rec_pred.values())),
        "recall_std": np.std(list(rec_pred.values())),
        "dice_per_label": dice_pred,
        "iou_per_label": iou_pred,
        "precision_per_label": prec_pred,
        "recall_per_label": rec_pred
    }

    metrics_crf = {
        "dice_mean": np.mean(list(dice_crf.values())),
        "dice_std": np.std(list(dice_crf.values())),
        "iou_mean": np.mean(list(iou_crf.values())),
        "iou_std": np.std(list(iou_crf.values())),
        "precision_mean": np.mean(list(prec_crf.values())),
        "precision_std": np.std(list(prec_crf.values())),
        "recall_mean": np.mean(list(rec_crf.values())),
        "recall_std": np.std(list(rec_crf.values())),
        "dice_per_label": dice_crf,
        "iou_per_label": iou_crf,
        "precision_per_label": prec_crf,
        "recall_per_label": rec_crf
    }

    diff_pred = diff_map_per_label(gt, pred, labels)
    diff_crf = diff_map_per_label(gt, crf, labels)

    save_combined_diff_plot(ct, pred, crf, diff_pred, diff_crf, labels, metrics_pred, metrics_crf, out_path)

    print("\n===== GLOBAL METRICS (mean ± std) =====")
    print(f"Pred → Dice={metrics_pred['dice_mean']:.3f}±{metrics_pred['dice_std']:.3f}, "
          f"IoU={metrics_pred['iou_mean']:.3f}±{metrics_pred['iou_std']:.3f}, "
          f"Prec={metrics_pred['precision_mean']:.3f}±{metrics_pred['precision_std']:.3f}, "
          f"Rec={metrics_pred['recall_mean']:.3f}±{metrics_pred['recall_std']:.3f}")

    print(f"CRF  → Dice={metrics_crf['dice_mean']:.3f}±{metrics_crf['dice_std']:.3f}, "
          f"IoU={metrics_crf['iou_mean']:.3f}±{metrics_crf['iou_std']:.3f}, "
          f"Prec={metrics_crf['precision_mean']:.3f}±{metrics_crf['precision_std']:.3f}, "
          f"Rec={metrics_crf['recall_mean']:.3f}±{metrics_crf['recall_std']:.3f}")

    print("\n===== Metrics per vertebra =====")
    for lab in labels:
        print(f"{VERTEBRA_NAMES[lab]}: Pred → D={dice_pred[lab]:.3f}, I={iou_pred[lab]:.3f}, "
              f"P={prec_pred[lab]:.3f}, R={rec_pred[lab]:.3f} | "
              f"CRF → D={dice_crf[lab]:.3f}, I={iou_crf[lab]:.3f}, "
              f"P={prec_crf[lab]:.3f}, R={rec_crf[lab]:.3f}")

# Main
if __name__ == "__main__":
    gt_path = r"C:\Users\PolDiaz\Desktop\PostProcessat\estudiN=5\groundTruth\sub-400_remap.nii"
    pred_path = r"C:\Users\PolDiaz\Desktop\PostProcessat\estudiN=5\predictions\sub-400.nii.gz"
    ct_path = r"C:\Users\PolDiaz\Desktop\PostProcessat\estudiN=5\imagesTs\sub-400_0000.nii"

    crf_path = r"C:\Users\PolDiaz\Desktop\PostProcessat\estudiN=5\Results\400\CRF\400_6.nii.gz"
    out_path = r"C:\Users\PolDiaz\Desktop\PostProcessat\estudiN=5\Results\400\Evaluation\400_6.png"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    evaluate_multilabel_with_crf(gt_path, pred_path, crf_path, ct_path, out_path)

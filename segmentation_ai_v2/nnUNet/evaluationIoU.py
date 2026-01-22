import os
import csv
import numpy as np
import nibabel as nib

########### FUNCTIONS ############

def iou_score_binary(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """
    Compute the Intersection over Union (IoU) between two binary masks

    Parameters:
    - gt_mask : np.ndarray
        Ground-truth binary mask
    - pred_mask : np.ndarray
        Predicted binary mask

    Returns:
    - float
        Intersection over Union score in the range [0, 1]
        
    """
    gt = gt_mask.astype(bool)
    pr = pred_mask.astype(bool)
    inter = np.logical_and(gt, pr).sum()
    union = gt.sum() + pr.sum() - inter
    if union == 0:
        return np.nan
    return inter / union


def iou_per_label(gt_vol: np.ndarray, pred_vol: np.ndarray, labels: list = None):
    """
    Compute Intersection over Union (IoU) for each label in a multi-label volume

    Parameters:
    - gt_vol : np.ndarray
        Ground-truth labeled volume
    - pred_vol : np.ndarray
        Predicted labeled volume
    - labels : list, optional
        List of labels to evaluate. If None, all labels present in either volume
        (excluding 0/background) will be used.

    Returns:
    - dict
        Dictionary mapping label -> IoU score (float)

    """
    if labels is None:
        labels = sorted(set(np.unique(gt_vol)).union(set(np.unique(pred_vol))))

    results = {}
    for lab in labels:
        if lab == 0:
            continue  # Skipping background

        gt_mask = (gt_vol == lab)
        pred_mask = (pred_vol == lab)

        if not gt_mask.any() or not pred_mask.any():
            continue  # Labels must be present in both

        results[lab] = float(iou_score_binary(gt_mask, pred_mask))

    return results

#   Anatomial regions

REGIONS = {
    "Cervical": list(range(20, 27)),   # 20–26
    "Thoracic": list(range(8, 20)),    # 8–19
    "Lumbral":  list(range(3, 8)),     # 3–7
}


def compute_region_stats(global_results):
    """
    Compute mean IoU metrics and sample counts for predefined anatomical regions

    Parameters:
    - global_results : list of dict
        List of dictionaries containing per-label metrics for each case. 
        Each dictionary must contain at least the keys:
            - "Label": int, the label of the vertebra
            - "IoU_Fine": float, IoU score for fine-tuned prediction
            - "IoU_Orig": float, IoU score for original prediction
            - "IoU_Diff": float, difference between fine and original IoU scores

    Returns:
    - dict
        Dictionary where each key is a region name (from REGIONS) and the value 
        is another dictionary with:
            - "N_samples": int
            - "Mean_IoU_Fine": float, mean IoU score for fine-tuned predictions
            - "Mean_IoU_Orig": float, mean IoU score for original predictions
            - "Mean_IoU_Diff": float, mean difference between fine and original IoU

    """
    region_stats = {}

    for region_name, labels in REGIONS.items():
        vals_fine, vals_orig, diffs = [], [], []

        for r in global_results:
            if r["Label"] in labels:
                if not np.isnan(r["IoU_Fine"]):
                    vals_fine.append(r["IoU_Fine"])
                if not np.isnan(r["IoU_Orig"]):
                    vals_orig.append(r["IoU_Orig"])
                if not np.isnan(r["IoU_Diff"]):
                    diffs.append(r["IoU_Diff"])

        region_stats[region_name] = {
            "N_samples": len(vals_fine),
            "Mean_IoU_Fine": float(np.mean(vals_fine)) if vals_fine else np.nan,
            "Mean_IoU_Orig": float(np.mean(vals_orig)) if vals_orig else np.nan,
            "Mean_IoU_Diff": float(np.mean(diffs)) if diffs else np.nan,
        }

    return region_stats


########### MAIN ############

if __name__ == "__main__":

    # Paths definiton
    GT_FOLDER = r"C:\Users\PolDiaz\Desktop\nnUNet_test\labelsTs_remap"
    PRED_FINE_FOLDER = r"C:\Users\PolDiaz\Desktop\nnUNet_test\TestPredictions_Fine-Tune"
    PRED_ORIG_FOLDER = r"C:\Users\PolDiaz\Desktop\nnUNet_test\TestPredictions_OriginalModel"
    OUTPUT_CSV = r"C:\Users\PolDiaz\Desktop\nnUNet_test\Results\IoU_results_global.csv"

    cases = sorted(f for f in os.listdir(GT_FOLDER) if f.endswith(".nii") or f.endswith(".nii.gz"))

    global_results = []
    label_counts = {}

    for case_file in cases:

        gt_path = os.path.join(GT_FOLDER, case_file)
        print(gt_path)
        base_name = case_file.replace("_remap", "")
        pred_fine_path = os.path.join(PRED_FINE_FOLDER, base_name)
        pred_orig_path = os.path.join(PRED_ORIG_FOLDER, base_name)

        if not os.path.exists(pred_fine_path) or not os.path.exists(pred_orig_path):
            print(f"Missing prediction for: {case_file}")
            continue

        gt = nib.load(gt_path).get_fdata().astype(np.int32)
        pred_fine = nib.load(pred_fine_path).get_fdata().astype(np.int32)
        pred_orig = nib.load(pred_orig_path).get_fdata().astype(np.int32)

        iou_fine = iou_per_label(gt, pred_fine)
        iou_orig = iou_per_label(gt, pred_orig)

        for lab in iou_fine:
            label_counts[lab] = label_counts.get(lab, 0) + 1

        for lab in iou_fine:
            i_f = iou_fine[lab]
            i_o = iou_orig.get(lab, np.nan)
            diff = i_f - i_o if not np.isnan(i_o) else np.nan

            global_results.append({
                "Case": base_name,
                "Label": lab,
                "IoU_Fine": i_f,
                "IoU_Orig": i_o,
                "IoU_Diff": diff
            })


    #   Mean per label
    mean_per_label = {}
    for lab, n in label_counts.items():
        vals_fine = [r["IoU_Fine"] for r in global_results if r["Label"] == lab]
        vals_orig = [r["IoU_Orig"] for r in global_results if r["Label"] == lab]
        diffs = [r["IoU_Diff"] for r in global_results if r["Label"] == lab]

        mean_per_label[lab] = {
            "N_samples": n,
            "Mean_IoU_Fine": float(np.mean(vals_fine)),
            "Mean_IoU_Orig": float(np.mean(vals_orig)),
            "Mean_IoU_Diff": float(np.mean(diffs)),
        }

    # Mean per region
    region_stats = compute_region_stats(global_results)

    # Saving CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["==== PER LABEL ===="])
        writer.writerow(["Label", "N_samples", "Mean_IoU_Fine", "Mean_IoU_Orig", "Mean_IoU_Diff"])
        for lab in sorted(mean_per_label.keys()):
            row = [
                lab,
                mean_per_label[lab]["N_samples"],
                mean_per_label[lab]["Mean_IoU_Fine"],
                mean_per_label[lab]["Mean_IoU_Orig"],
                mean_per_label[lab]["Mean_IoU_Diff"],
            ]
            writer.writerow(row)

        writer.writerow([])
        writer.writerow(["==== PER REGION ===="])
        writer.writerow(["Region", "N_samples", "Mean_IoU_Fine", "Mean_IoU_Orig", "Mean_IoU_Diff"])

        for region, stats in region_stats.items():
            writer.writerow([
                region,
                stats["N_samples"],
                stats["Mean_IoU_Fine"],
                stats["Mean_IoU_Orig"],
                stats["Mean_IoU_Diff"],
            ])

    print("\nCSV generated:", OUTPUT_CSV)

    print("\n=== MEAN IoU PER REGION ===")
    for reg, stats in region_stats.items():
        print(f"{reg}: N={stats['N_samples']}  |  Fine={stats['Mean_IoU_Fine']:.4f}  |  Orig={stats['Mean_IoU_Orig']:.4f}  |  Diff={stats['Mean_IoU_Diff']:.4f}")

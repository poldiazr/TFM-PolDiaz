import os
import csv
import numpy as np
import nibabel as nib

########### FUNCTIONS ############

def dice_score_binary(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """
    Compute the Dice Similarity Coefficient (DSC) between two binary masks

    Parameters:
    - gt_mask : np.ndarray
        Ground-truth binary mask
    - pred_mask : np.ndarray
        Predicted binary mask

    Returns:
    - float
        Dice Similarity Coefficient in the range [0, 1]

    """
    gt = gt_mask.astype(bool)
    pr = pred_mask.astype(bool)
    inter = np.logical_and(gt, pr).sum()
    denom = gt.sum() + pr.sum()
    if denom == 0:
        return np.nan
    return 2.0 * inter / denom


def dice_per_label(gt_vol: np.ndarray, pred_vol: np.ndarray, labels: list = None):
    """
    Compute Dice Similarity Coefficient (DSC) for each label in a multi-label volume

    Parameters:
    - gt_vol : np.ndarray
        Ground-truth labeled volume. Each integer represents a different label
    - pred_vol : np.ndarray
        Predicted labeled volume
    - labels : list, optional
        List of labels to evaluate. If None, all labels present in either volume
        (excluding 0/background) will be used

    Returns:
    - dict
        Dictionary mapping label -> Dice score (float)

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

        results[lab] = float(dice_score_binary(gt_mask, pred_mask))

    return results


#   Anatomical regions
REGIONS = {
    "Cervical": list(range(20, 27)),   # 20–26
    "Thoracic": list(range(8, 20)),    # 8–19
    "Lumbar":   list(range(3, 8)),     # 3–7
}


def compute_region_stats(global_results):
    """
    Compute mean metrics and sample counts for predefined anatomical regions.

    Parameters:
    - global_results : list of dict
        List of dictionaries containing per-label metrics for each case. 
        Each dictionary must contain at least the keys:
            - "Label": int, the label of the vertebra
            - "DSC_Fine": float, Dice score for fine-tuned prediction
            - "DSC_Orig": float, Dice score for original prediction
            - "DSC_Diff": float, difference between fine and original Dice scores

    Returns:
    - dict
        Dictionary where each key is a region name (from REGIONS) and the value 
        is another dictionary with:
            - "N_samples": int
            - "Mean_DSC_Fine": float, mean Dice score for fine-tuned predictions
            - "Mean_DSC_Orig": float, mean Dice score for original predictions
            - "Mean_DSC_Diff": float, mean difference between fine and original Dice

    """
    region_stats = {}

    for region_name, labels in REGIONS.items():
        vals_fine, vals_orig, diffs = [], [], []

        for r in global_results:
            if r["Label"] in labels:
                if not np.isnan(r["DSC_Fine"]):
                    vals_fine.append(r["DSC_Fine"])
                if not np.isnan(r["DSC_Orig"]):
                    vals_orig.append(r["DSC_Orig"])
                if not np.isnan(r["DSC_Diff"]):
                    diffs.append(r["DSC_Diff"])

        region_stats[region_name] = {
            "N_samples": len(vals_fine),
            "Mean_DSC_Fine": float(np.mean(vals_fine)) if vals_fine else np.nan,
            "Mean_DSC_Orig": float(np.mean(vals_orig)) if vals_orig else np.nan,
            "Mean_DSC_Diff": float(np.mean(diffs)) if diffs else np.nan,
        }

    return region_stats

########### MAIN ############

if __name__ == "__main__":

    # Path definition
    GT_FOLDER = r"C:\Users\PolDiaz\Desktop\nnUNet_test\labelsTs_remap"
    PRED_FINE_FOLDER = r"C:\Users\PolDiaz\Desktop\nnUNet_test\TestPredictions_Fine-Tune"
    PRED_ORIG_FOLDER = r"C:\Users\PolDiaz\Desktop\nnUNet_test\TestPredictions_OriginalModel"
    OUTPUT_CSV = r"C:\Users\PolDiaz\Desktop\nnUNet_test\Results\DSC_results_global.csv"

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

        dice_fine = dice_per_label(gt, pred_fine)
        dice_orig = dice_per_label(gt, pred_orig)

        # Global count for each label
        for lab in dice_fine:
            label_counts[lab] = label_counts.get(lab, 0) + 1

        # Saving results for each label
        for lab in dice_fine:
            dsc_f = dice_fine[lab]
            dsc_o = dice_orig.get(lab, np.nan)
            diff = dsc_f - dsc_o if not np.isnan(dsc_o) else np.nan

            global_results.append({
                "Case": base_name,
                "Label": lab,
                "DSC_Fine": dsc_f,
                "DSC_Orig": dsc_o,
                "DSC_Diff": diff
            })


    # Mean per label
    mean_per_label = {}
    for lab, n in label_counts.items():
        vals_fine = [r["DSC_Fine"] for r in global_results if r["Label"] == lab]
        vals_orig = [r["DSC_Orig"] for r in global_results if r["Label"] == lab]
        diffs = [r["DSC_Diff"] for r in global_results if r["Label"] == lab]

        mean_per_label[lab] = {
            "N_samples": n,
            "Mean_DSC_Fine": float(np.mean(vals_fine)),
            "Mean_DSC_Orig": float(np.mean(vals_orig)),
            "Mean_DSC_Diff": float(np.mean(diffs)),
        }

    # Mean per region
    region_stats = compute_region_stats(global_results)

    # Saving CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["==== PER LABEL ===="])
        writer.writerow(["Label", "N_samples", "Mean_DSC_Fine", "Mean_DSC_Orig", "Mean_DSC_Diff"])
        for lab in sorted(mean_per_label.keys()):
            row = [
                lab,
                mean_per_label[lab]["N_samples"],
                mean_per_label[lab]["Mean_DSC_Fine"],
                mean_per_label[lab]["Mean_DSC_Orig"],
                mean_per_label[lab]["Mean_DSC_Diff"],
            ]
            writer.writerow(row)

        writer.writerow([])  
        writer.writerow(["==== PER REGION ===="])
        writer.writerow(["Region", "N_samples", "Mean_DSC_Fine", "Mean_DSC_Orig", "Mean_DSC_Diff"])

        for region, stats in region_stats.items():
            writer.writerow([
                region,
                stats["N_samples"],
                stats["Mean_DSC_Fine"],
                stats["Mean_DSC_Orig"],
                stats["Mean_DSC_Diff"],
            ])

    print("\nCSV generated:", OUTPUT_CSV)

    print("\n=== MEAN DSC PER REGION ===")
    for reg, stats in region_stats.items():
        print(f"{reg}: N={stats['N_samples']}  |  Fine={stats['Mean_DSC_Fine']:.4f}  |  Orig={stats['Mean_DSC_Orig']:.4f}  |  Diff={stats['Mean_DSC_Diff']:.4f}")

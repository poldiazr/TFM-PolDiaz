import os
import csv
import numpy as np
import nibabel as nib
from medpy.metric.binary import hd95

########### FUNCTIONS ############

def hd95_per_label(gt_vol: np.ndarray, pred_vol: np.ndarray, labels: list = None):
    """
    Compute the 95th percentile Hausdorff Distance (HD95) for each label in a multi-label volume using MedPy

    Parameters:
    - gt_vol : np.ndarray
        Ground-truth labeled volume. Each integer represents a different label
    - pred_vol : np.ndarray
        Predicted labeled volume
    labels : list, optional
        List of labels to evaluate. If None, all labels present in either volume
        (excluding 0/background) will be used

    Returns:
    - dict
        Dictionary mapping label -> HD95 distance (float, in voxel units)

    """
    if labels is None:
        labels = sorted(set(np.unique(gt_vol)).union(set(np.unique(pred_vol))))

    results = {}
    for lab in labels:
        if lab == 0:
            continue  # Skip background

        gt_mask = (gt_vol == lab)
        pred_mask = (pred_vol == lab)

        if not gt_mask.any() or not pred_mask.any():
            continue

        try:
            results[lab] = float(hd95(gt_mask.astype(bool), pred_mask.astype(bool)))
        except ValueError:  # if not valid borders
            results[lab] = np.nan

    return results


# Anatomical regions
REGIONS = {
    "Cervical": list(range(20, 27)),   # 20–26
    "Thoracic": list(range(8, 20)),    # 8–19
    "Lumbar":   list(range(3, 8)),     # 3–7
}


def compute_region_stats(global_results):
    """
    Compute mean HD95 metrics and sample counts for predefined anatomical regions.

    Parameters:
    - global_results : list of dict
        List of dictionaries containing per-label metrics for each case
        Each dictionary must contain at least the keys:
            - "Label": int, the label of the vertebra
            - "HD95_Fine": float, HD95 for fine-tuned prediction
            - "HD95_Orig": float, HD95 for original prediction
            - "HD95_Diff": float, difference between fine and original HD95

    Returns:
    - dict
        Dictionary where each key is a region name (from REGIONS) and the value 
        is another dictionary with:
            - "N_samples": int
            - "Mean_HD95_Fine": float, mean HD95 for fine-tuned predictions
            - "Mean_HD95_Orig": float, mean HD95 for original predictions
            - "Mean_HD95_Diff": float, mean difference between fine and original HD95
            
    """

    region_stats = {}

    for region_name, labels in REGIONS.items():
        vals_fine, vals_orig, diffs = [], [], []

        for r in global_results:
            if r["Label"] in labels:
                if not np.isnan(r["HD95_Fine"]):
                    vals_fine.append(r["HD95_Fine"])
                if not np.isnan(r["HD95_Orig"]):
                    vals_orig.append(r["HD95_Orig"])
                if not np.isnan(r["HD95_Diff"]):
                    diffs.append(r["HD95_Diff"])

        region_stats[region_name] = {
            "N_samples": len(vals_fine),
            "Mean_HD95_Fine": float(np.mean(vals_fine)) if vals_fine else np.nan,
            "Mean_HD95_Orig": float(np.mean(vals_orig)) if vals_orig else np.nan,
            "Mean_HD95_Diff": float(np.mean(diffs)) if diffs else np.nan,
        }

    return region_stats


########### MAIN ############

if __name__ == "__main__":

    # Path defition
    GT_FOLDER = r"C:\Users\PolDiaz\Desktop\nnUNet_test\labelsTs_remap"
    PRED_FINE_FOLDER = r"C:\Users\PolDiaz\Desktop\nnUNet_test\TestPredictions_Fine-Tune"
    PRED_ORIG_FOLDER = r"C:\Users\PolDiaz\Desktop\nnUNet_test\TestPredictions_OriginalModel"
    OUTPUT_CSV = r"C:\Users\PolDiaz\Desktop\nnUNet_test\Results\HD95_medpy_results_global.csv"

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

        hd95_fine = hd95_per_label(gt, pred_fine)
        hd95_orig = hd95_per_label(gt, pred_orig)

        for lab in hd95_fine:
            label_counts[lab] = label_counts.get(lab, 0) + 1

        for lab in hd95_fine:
            h_f = hd95_fine[lab]
            h_o = hd95_orig.get(lab, np.nan)
            diff = h_f - h_o if not np.isnan(h_o) else np.nan

            global_results.append({
                "Case": base_name,
                "Label": lab,
                "HD95_Fine": h_f,
                "HD95_Orig": h_o,
                "HD95_Diff": diff
            })

    # Mean per label
    mean_per_label = {}
    for lab, n in label_counts.items():
        vals_fine = [r["HD95_Fine"] for r in global_results if r["Label"] == lab]
        vals_orig = [r["HD95_Orig"] for r in global_results if r["Label"] == lab]
        diffs = [r["HD95_Diff"] for r in global_results if r["Label"] == lab]

        mean_per_label[lab] = {
            "N_samples": n,
            "Mean_HD95_Fine": float(np.mean(vals_fine)),
            "Mean_HD95_Orig": float(np.mean(vals_orig)),
            "Mean_HD95_Diff": float(np.mean(diffs)),
        }

    # Mean per region
    region_stats = compute_region_stats(global_results)

    # Saving CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["==== PER LABEL ===="])
        writer.writerow(["Label", "N_samples", "Mean_HD95_Fine", "Mean_HD95_Orig", "Mean_HD95_Diff"])
        for lab in sorted(mean_per_label.keys()):
            row = [
                lab,
                mean_per_label[lab]["N_samples"],
                mean_per_label[lab]["Mean_HD95_Fine"],
                mean_per_label[lab]["Mean_HD95_Orig"],
                mean_per_label[lab]["Mean_HD95_Diff"],
            ]
            writer.writerow(row)

        writer.writerow([]) 
        writer.writerow(["==== PER REGION ===="])
        writer.writerow(["Region", "N_samples", "Mean_HD95_Fine", "Mean_HD95_Orig", "Mean_HD95_Diff"])

        for region, stats in region_stats.items():
            writer.writerow([
                region,
                stats["N_samples"],
                stats["Mean_HD95_Fine"],
                stats["Mean_HD95_Orig"],
                stats["Mean_HD95_Diff"],
            ])

    print("\nCSV generated:", OUTPUT_CSV)

    print("\n=== MEAN HD95 PER REGION ===")
    for reg, stats in region_stats.items():
        print(f"{reg}: N={stats['N_samples']}  |  Fine={stats['Mean_HD95_Fine']:.4f}  |  Orig={stats['Mean_HD95_Orig']:.4f}  |  Diff={stats['Mean_HD95_Diff']:.4f}")

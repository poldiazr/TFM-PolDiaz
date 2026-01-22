import os
import numpy as np
import nibabel as nib
import pandas as pd

def sens_spec_per_label(gt_vol: np.ndarray, pred_vol: np.ndarray, labels: list = None):
    """
    Compute Sensitivity and Specificity for each label.
    """
    if labels is None:
        labels = sorted(set(np.unique(gt_vol)).union(set(np.unique(pred_vol))))

    results = {}
    for lab in labels:
        if lab == 0:
            continue # skip background

        gt_mask = (gt_vol == lab)
        pred_mask = (pred_vol == lab)

        TP = np.logical_and(gt_mask, pred_mask).sum()
        FP = np.logical_and(np.logical_not(gt_mask), pred_mask).sum()
        FN = np.logical_and(gt_mask, np.logical_not(pred_mask)).sum()
        TN = np.logical_and(np.logical_not(gt_mask), np.logical_not(pred_mask)).sum()

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan

        results[lab] = {"Sensitivity": sensitivity, "Specificity": specificity}

    return results

def precision_per_label(gt_vol: np.ndarray, pred_vol: np.ndarray, labels: list = None):
    """
    Compute Precision for each label
    """
    if labels is None:
        labels = sorted(set(np.unique(gt_vol)).union(set(np.unique(pred_vol))))

    results = {}
    for lab in labels:
        if lab == 0:
            continue  # skip background

        gt_mask = (gt_vol == lab)
        pred_mask = (pred_vol == lab)

        TP = np.logical_and(gt_mask, pred_mask).sum()
        FP = np.logical_and(np.logical_not(gt_mask), pred_mask).sum()

        precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
        results[lab] = precision

    return results

# Main
GT_FOLDER = r"C:\Users\PolDiaz\Desktop\nnUNet_test\labelsTs_remap"
PRED_FINE_FOLDER = r"C:\Users\PolDiaz\Desktop\nnUNet_test\TestPredictions_Fine-Tune"
PRED_ORIG_FOLDER = r"C:\Users\PolDiaz\Desktop\nnUNet_test\TestPredictions_OriginalModel"
OUTPUT_EXCEL = r"C:\Users\PolDiaz\Desktop\nnUNet_test\Results\Metrics_mean_per_label.xlsx"

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

    # Metrics computation
    sensspec_fine = sens_spec_per_label(gt, pred_fine)
    sensspec_orig = sens_spec_per_label(gt, pred_orig)
    prec_fine = precision_per_label(gt, pred_fine)
    prec_orig = precision_per_label(gt, pred_orig)

    for lab in sensspec_fine:
        label_counts[lab] = label_counts.get(lab, 0) + 1
        global_results.append({
            "Case": base_name,
            "Label": lab,
            "Sensitivity_Fine": sensspec_fine[lab]["Sensitivity"],
            "Specificity_Fine": sensspec_fine[lab]["Specificity"],
            "Precision_Fine": prec_fine[lab],
            "Sensitivity_Orig": sensspec_orig.get(lab, {}).get("Sensitivity", np.nan),
            "Specificity_Orig": sensspec_orig.get(lab, {}).get("Specificity", np.nan),
            "Precision_Orig": prec_orig.get(lab, np.nan)
        })


df = pd.DataFrame(global_results)
labels = sorted(df["Label"].unique())
mean_per_label = {}

for lab in labels:
    df_lab = df[df["Label"] == lab]

    # Filtreting NaN values
    sens_f = df_lab["Sensitivity_Fine"].replace(0, np.nan).dropna()
    spec_f = df_lab["Specificity_Fine"].replace(0, np.nan).dropna()
    prec_f = df_lab["Precision_Fine"].replace(0, np.nan).dropna()
    sens_o = df_lab["Sensitivity_Orig"].replace(0, np.nan).dropna()
    spec_o = df_lab["Specificity_Orig"].replace(0, np.nan).dropna()
    prec_o = df_lab["Precision_Orig"].replace(0, np.nan).dropna()

    mean_per_label[lab] = {
        "N_samples": len(sens_f),
        "Mean_Sensitivity_Fine": float(np.mean(sens_f)) if len(sens_f) > 0 else np.nan,
        "Mean_Specificity_Fine": float(np.mean(spec_f)) if len(spec_f) > 0 else np.nan,
        "Mean_Precision_Fine": float(np.mean(prec_f)) if len(prec_f) > 0 else np.nan,
        "Mean_Sensitivity_Orig": float(np.mean(sens_o)) if len(sens_o) > 0 else np.nan,
        "Mean_Specificity_Orig": float(np.mean(spec_o)) if len(spec_o) > 0 else np.nan,
        "Mean_Precision_Orig": float(np.mean(prec_o)) if len(prec_o) > 0 else np.nan,
    }

# Saving
df_mean_labels = pd.DataFrame.from_dict(mean_per_label, orient='index').reset_index().rename(columns={'index':'Label'})
df_mean_labels.to_excel(OUTPUT_EXCEL, index=False)

print(f"Excel saved: {OUTPUT_EXCEL}")
print(df_mean_labels)

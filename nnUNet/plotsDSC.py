import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Load Dice score Excel Results
file_path = r"C:\Users\PolDiaz\Desktop\nnUNet_test\Results\DSC_global.xlsx"

df_labels = pd.read_excel(file_path, sheet_name="DSC_perLabel", header=1)
df_regions = pd.read_excel(file_path, sheet_name="DSC_perRegion", header=1)

# Compute difference Fine - Original
df_labels["DSC_Diff"] = df_labels["Mean_DSC_Fine"] - df_labels["Mean_DSC_Orig"]
df_regions["DSC_Diff"] = df_regions["Mean_DSC_Fine"] - df_regions["Mean_DSC_Orig"]

# Vertebra label mapping
vertebra_names = {
    3: "L5", 4: "L4", 5: "L3", 6: "L2", 7: "L1",
    8: "T12", 9: "T11", 10: "T10", 11: "T9", 12: "T8",
    13: "T7", 14: "T6", 15: "T5", 16: "T4", 17: "T3",
    18: "T2", 19: "T1",
    20: "C7", 21: "C6", 22: "C5", 23: "C4",
    24: "C3", 25: "C2", 26: "C1"
}

df_labels["Label_Name"] = df_labels["Label"].map(vertebra_names)

# Ordering vertebrae from C1 to L5
ordered_labels = ["C1", "C2", "C3", "C4", "C5", "C6", "C7",
                  "T1", "T2", "T3", "T4", "T5", "T6", "T7",
                  "T8", "T9", "T10", "T11", "T12",
                  "L1", "L2", "L3", "L4", "L5"]

df_labels = df_labels.set_index("Label_Name").loc[ordered_labels].reset_index()

# Create output folder
output_dir = "results_dsc_plots"
os.makedirs(output_dir, exist_ok=True)

# Plot difference for each vertebra (fine-tuned vs original)
x = np.arange(len(df_labels))*0.6
plt.figure(figsize=(12,6))
plt.bar(
    x,
    df_labels["DSC_Diff"],
    color="red",
    edgecolor="darkred",
    alpha=0.5,
    width=0.4
)

plt.xticks(x, df_labels["Label_Name"]) 
plt.title("Difference in Dice score (Fine-tuned - Original) per vertebra")
plt.xlabel("Vertebra")
plt.ylabel("Difference mean Dice score")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "DSC_diff_by_label.png"), dpi=300)
plt.close()

# Boxplot per vertebral region
def vertebra_to_region(v):
    if v.startswith("C"): return "Cervical"
    if v.startswith("T"): return "Thoracic"
    if v.startswith("L"): return "Lumbar"
    return "Unknown"

df_labels["Region"] = df_labels["Label_Name"].apply(vertebra_to_region)

regions = ["Cervical", "Thoracic", "Lumbar"]
orig_data = [df_labels[df_labels["Region"]==r]["Mean_DSC_Orig"] for r in regions]
fine_data = [df_labels[df_labels["Region"]==r]["Mean_DSC_Fine"] for r in regions]

# Boxplot positioning
positions_orig = [1,3,5]    
positions_fine = [1.65,3.65,5.65] 

plt.figure(figsize=(10,6))

b_orig = plt.boxplot(orig_data, positions=positions_orig, widths=0.6, patch_artist=True)
b_fine = plt.boxplot(fine_data, positions=positions_fine, widths=0.6, patch_artist=True)

for patch in b_orig['boxes']:
    patch.set_facecolor("#DD0E0E")  
for patch in b_fine['boxes']:
    patch.set_facecolor("#dddddd")  
for median in b_orig['medians']:
    median.set_color("#000000")  
    median.set_linewidth(1)
for median in b_fine['medians']:
    median.set_color("#000000")  
    median.set_linewidth(1)


plt.xticks([1.35,3.35,5.35], regions)  
plt.title("Comparison of Dice scores across vertebral regions (Original vs Fine-tuned)")
plt.ylabel("Dice score")
plt.grid(True, axis="y", alpha=0.3)

plt.plot([], color='#DD0E0E', label="Original")
plt.plot([], color='#dddddd', label="Fine-tuned")
plt.legend(loc="lower left")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "DSC_boxplot.png"), dpi=300)
plt.close()

# Plot Dice score per vertebra with sample size
fig, ax1 = plt.subplots(figsize=(12,6))
ax1.plot(
    df_labels["Label_Name"],
    df_labels["Mean_DSC_Fine"],
    marker="o",
    linewidth=2,
    color="#b81414",
    label="Mean Dice score"
)
ax1.set_xlabel("Vertebra")
ax1.set_ylabel("Mean Dice score")
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.bar(
    df_labels["Label_Name"],
    df_labels["N_samples"],
    alpha=0.25,
    color="red",
    label="N samples"
)
ax2.set_ylabel("N samples")
plt.title("Mean Dice score (fine-tuned model) per vertebra with sample size")

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower left")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "DSC_fine_by_label_with_Nsamples.png"), dpi=300)
plt.close()

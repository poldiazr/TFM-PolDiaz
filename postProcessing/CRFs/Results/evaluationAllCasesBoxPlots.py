import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

filepath = r"C:\Users\PolDiaz\Desktop\PostProcessat\CRF_Results\Results.xlsx"
T = pd.read_excel(filepath)

metrics = ["Dice", "IoU", "Precision", "Recall"]
initial_cols = [f"{m}_No CRF" for m in metrics]
processed_cols = [f"{m}_CRF" for m in metrics]

def extract_mean(s):
    if isinstance(s, str):
        return float(s.split("±")[0].strip())
    return np.nan

for col in initial_cols:
    T[col] = T[col].ffill()  

# Extracting data
data_list, group_list, metric_list = [], [], []

for metric, init_col, proc_col in zip(metrics, initial_cols, processed_cols):
    init_vals = T[init_col].apply(extract_mean)
    proc_vals = T[proc_col].apply(extract_mean)
    
    mask = init_vals.notna() & proc_vals.notna()
    
    # Baseline
    data_list.extend(init_vals[mask])
    group_list.extend(["Baseline (No CRF)"] * mask.sum())
    metric_list.extend([metric] * mask.sum())
    
    # CRF
    data_list.extend(proc_vals[mask])
    group_list.extend(["CRF Post-Processed"] * mask.sum())
    metric_list.extend([metric] * mask.sum())

df_plot = pd.DataFrame({
    "Value": data_list,
    "Group": group_list,
    "Metric": metric_list
})

# Plot
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(9,6))
ax = plt.gca()

plt.title(
    r"$\bf{Impact\ of\ CRF\ Post-Processing\ on\ Segmentation\ Performance}$"
    "\nN = 10",
    fontsize=16
)

# Boxplot
sns.boxplot(
    data=df_plot,
    x="Metric",
    y="Value",
    hue="Group",
    palette={"Baseline (No CRF)":"#dddddd", "CRF Post-Processed":"#FF0000"},
    width=0.65,
    ax=ax,
    fliersize=0  
    )

ax.set_xlabel("Metrics", fontsize=13)
ax.set_ylabel("Metric Value", fontsize=13)
ax.set_ylim([0.65, 1.0])
ax.grid(axis="y", alpha=0.3)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, loc='lower right', frameon=True, framealpha=0.9)

plt.tight_layout()

# Saving
output_path = r"C:\Users\PolDiaz\Desktop\PostProcessat\CRF_Results\CRF_N10_Boxplot.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

filepath = r"C:\Users\PolDiaz\Desktop\PostProcessat\Chan-Vese_Results\Results.xlsx"
T = pd.read_excel(filepath, sheet_name="Per_Case")

metrics = ["Dice", "IoU", "Precision", "Recall"]
initial_cols = [f"{m}_Initial" for m in metrics]
processed_cols = [f"{m}_Refined" for m in metrics]

def extract_mean(s):
    if isinstance(s, str):
        return float(s.split("±")[0].strip())
    return np.nan

# Extracting data
data_list, group_list, metric_list = [], [], []

for metric, init_col, proc_col in zip(metrics, initial_cols, processed_cols):
    init_vals = T[init_col].apply(extract_mean)
    proc_vals = T[proc_col].apply(extract_mean)
    
    mask = init_vals.notna() & proc_vals.notna()
    
    # Initial
    data_list.extend(init_vals[mask])
    group_list.extend(["Initial"] * mask.sum())
    metric_list.extend([metric] * mask.sum())
    
    # Refined
    data_list.extend(proc_vals[mask])
    group_list.extend(["Refined"] * mask.sum())
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
    r"$\bf{Impact\ of\ Chan-Vese\ Refinement\ on\ Segmentation\ Performance}$"
    "\nN = 10",
    fontsize=14
)

sns.boxplot(
    data=df_plot,
    x="Metric",
    y="Value",
    hue="Group",
    palette={"Initial":"#dddddd", "Refined":"#FF0000"},
    width=0.65,
    ax=ax,
    fliersize=0  
)

ax.set_xlabel("Metrics", fontsize=13)
ax.set_ylabel("Metric Value", fontsize=13)
ax.set_ylim([0.65, 1.0])
ax.grid(axis="y", alpha=0.3)

legend_elements = [
    Patch(facecolor="#FF0000", edgecolor="black", label="Chan-Vese Refined"),
    Patch(facecolor="#dddddd", edgecolor="black", label="Baseline (no Chan-Vese)")
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True, framealpha=0.9)

plt.tight_layout()

# Saving
output_path = r"C:\Users\PolDiaz\Desktop\PostProcessat\ChanVese_ReferenceCase_Boxplot.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

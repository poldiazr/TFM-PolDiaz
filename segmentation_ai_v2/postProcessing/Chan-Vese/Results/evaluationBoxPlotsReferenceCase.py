import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

filepath = r"C:\Users\PolDiaz\Desktop\PostProcessat\Chan-Vese_Results\Results.xlsx"
T = pd.read_excel(filepath, decimal=',')

initial_cols = [
    "Dice_Initial", "IoU_Initial", "Precision_Initial", "Recall_Initial"
]

processed_cols = [
    "Dice_Refined", "IoU_Refined", "Precision_Refined", "Recall_Refined"
]

metric_names = ["Dice", "IoU", "Precision", "Recall"]

def extract_mean(s):
    if isinstance(s, str):
        return float(s.replace(',', '.').split("±")[0].strip())
    return float(s)

# Extracting data
data_list, group_list, metric_list = [], [], []

for metric, init_col, proc_col in zip(metric_names, initial_cols, processed_cols):

    # Initial values
    init_vals = T[init_col].apply(extract_mean).values
    data_list.extend(init_vals)
    group_list.extend(["Initial"] * len(init_vals))
    metric_list.extend([metric] * len(init_vals))

    # Refined values
    proc_vals = T[proc_col].apply(extract_mean).values
    data_list.extend(proc_vals)
    group_list.extend(["Refined"] * len(proc_vals))
    metric_list.extend([metric] * len(proc_vals))

df = pd.DataFrame({
    "Value": data_list,
    "Group": group_list,
    "Metric": metric_list
})

# Plot
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(9,6))
plt.title(
    r"$\bf{Impact\ of\ Chan-Vese\ Refinement\ on\ Segmentation\ Performance}$"
    "\nReference Case",
    fontsize=14
)

sns.boxplot(
    data=df,
    x="Metric",
    y="Value",
    hue="Group",
    palette={"Initial": "#dddddd", "Refined": "#FF0000"},
    width=0.6,
    fliersize=0
)

plt.xlabel("Metrics", fontsize=13)
plt.ylabel("Metric Value", fontsize=13)
plt.ylim([0.65, 1])
plt.grid(axis="y", alpha=0.3)

legend_elements = [
    Patch(facecolor="#FF0000", edgecolor='black', label="Chan-Vese Refined"),
    Patch(facecolor="#dddddd", edgecolor='black', label="Baseline (no Chan-Vese)")

]
plt.legend(handles=legend_elements, loc="lower right", frameon=True, framealpha=0.9)

plt.tight_layout()

# Saving
output_path = r"C:\Users\PolDiaz\Desktop\PostProcessat\Chan-Vese_Results\ChanVese_ReferenceCase_Boxplot.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

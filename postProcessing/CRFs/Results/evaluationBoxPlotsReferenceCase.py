import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

filepath = r"C:\Users\PolDiaz\Desktop\PostProcessat\CRF_Results\Results_250086.xlsx"
T = pd.read_excel(filepath)

initial_cols = [
    "Dice_No CRF", "IoU_No CRF", "Precision_No CRF", "Recall_No CRF"
]

processed_cols = [
    "Dice_CRF", "IoU_CRF", "Precision_CRF", "Recall_CRF"
]

metric_names = ["Dice", "IoU", "Precision", "Recall"]

def extract_mean(s):
    if isinstance(s, str):
        return float(s.split("±")[0].strip())
    return np.nan

# Extractingc values
data_list, group_list, metric_list = [], [], []

for metric, init_col, proc_col in zip(metric_names, initial_cols, processed_cols):

    # Baseline
    init_value = extract_mean(T[init_col].dropna().values[0])
    data_list.append(init_value)
    group_list.append("Initial")
    metric_list.append(metric)

    # CRF
    proc_vals = T[proc_col].apply(extract_mean).values
    data_list.extend(proc_vals)
    group_list.extend(["Processed"] * len(proc_vals))
    metric_list.extend([metric] * len(proc_vals))

df = pd.DataFrame({
    "Value": data_list,
    "Group": group_list,
    "Metric": metric_list
})

spacing = 1
metric_to_x = {m: i * spacing for i, m in enumerate(metric_names)}
df["x_pos"] = df["Metric"].map(metric_to_x)

# Plot
sns.set(style="whitegrid", font_scale=1.2)

plt.figure(figsize=(9, 6))
plt.title(
    r"$\bf{Impact\ of\ CRF\ Post-Processing\ on\ Segmentation\ Performance}$"
    "\nReference Case",
    fontsize=16
)

# Boxplots
sns.boxplot(
    data=df[df["Group"] == "Processed"],
    x="x_pos",
    y="Value",
    width=0.55,
    color="#FF0000",
    boxprops=dict(alpha=0.9)
)

# Dash line
for metric in metric_names:
    x = metric_to_x[metric]
    init_val = df[
        (df["Metric"] == metric) & (df["Group"] == "Initial")
    ]["Value"].values[0]

    plt.scatter(
        x, init_val,
        s=40,
        color="black",
        zorder=5
    )

    plt.plot(
        [x - 0.2, x + 0.2],
        [init_val, init_val],
        linestyle="--",
        color="black",
        linewidth=1.5,
        zorder=4
    )

plt.xticks(list(metric_to_x.values()), metric_names)
plt.xlabel("Metrics")
plt.ylabel("Metric Value")
plt.ylim([0.8, 0.95])
plt.grid(axis="y", alpha=0.3)

plt.xlim(
    min(metric_to_x.values()) - 0.35,
    max(metric_to_x.values()) + 0.35
)


# Legend
legend_elements = [
    Patch(facecolor="#FF0000", edgecolor="gray", label="CRF Post-Processed"),
    Line2D(
        [0], [0],
        marker='o',
        color='black',
        label="Baseline (no CRF)",
        markerfacecolor='black',
        markersize=6,
        linestyle='--'
    )
]
plt.legend(
    handles=legend_elements,
    loc="lower right",
    frameon=True,
    framealpha=0.9
)

plt.tight_layout()

# Saving
output_path = r"C:\Users\PolDiaz\Desktop\PostProcessat\CRF_Results\CRF_ReferenceCase.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

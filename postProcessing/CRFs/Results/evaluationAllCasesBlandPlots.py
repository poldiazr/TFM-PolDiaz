import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon
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
data = {}
for metric, init_col, proc_col in zip(metrics, initial_cols, processed_cols):
    init_vals = T[init_col].apply(extract_mean)
    proc_vals = T[proc_col].apply(extract_mean)
    mask = init_vals.notna() & proc_vals.notna()
    
    data[metric] = pd.DataFrame({
        "Initial": init_vals[mask],
        "Processed": proc_vals[mask],
        "Case": T["Case"][mask]
    })

# Significance test
sig_results = {}
for metric in metrics:
    df_metric = data[metric]
    initial_vals = df_metric["Initial"]
    processed_vals = df_metric["Processed"]
    
    t_stat, p_val_t = ttest_rel(processed_vals, initial_vals)
    
    try:
        w_stat, p_val_w = wilcoxon(processed_vals, initial_vals)
    except ValueError:
        w_stat, p_val_w = np.nan, np.nan
    
    sig_results[metric] = {
        "p_val_t": p_val_t,
        "p_val_w": p_val_w,
        "mean_initial": np.mean(initial_vals),
        "mean_processed": np.mean(processed_vals)
    }

# Plot
sns.set(style="whitegrid", font_scale=1.2)
fig, axes = plt.subplots(2, 2, figsize=(11,9))
axes = axes.flatten()
fig.suptitle(
    r"$\bf{Impact\ of\ CRF\ Post-Processing\ on\ Segmentation\ Performance}$" "\nN = 10",
    fontsize=16, x=0.45
)
cases = sorted(T["Case"].dropna().unique())
colors = plt.cm.Reds(np.linspace(0.6, 1.0, len(cases)))

handles, labels = [], []

for ax, metric in zip(axes, metrics):
    df_metric = data[metric]
    
    # Scatter per case
    for i, case in enumerate(cases):
        df_case = df_metric[df_metric["Case"] == case]
        mean_vals = (df_case["Initial"] + df_case["Processed"]) / 2
        diff_vals = df_case["Processed"] - df_case["Initial"]
        
        sc = ax.scatter(mean_vals, diff_vals, color=colors[i], alpha=0.7, label=f"Case {int(case)}")
        if metric == metrics[0]:
            handles.append(sc)
            labels.append(f"Case {int(case)}")
    
    # Mean and ±1.96 SD
    mean_diff = np.mean(df_metric["Processed"] - df_metric["Initial"])
    sd_diff = np.std(df_metric["Processed"] - df_metric["Initial"], ddof=1)
    ax.axhline(mean_diff, color='gray', linestyle='--', linewidth=2)
    ax.axhline(mean_diff + 1.96*sd_diff, color='red', linestyle='--', linewidth=1.5)
    ax.axhline(mean_diff - 1.96*sd_diff, color='red', linestyle='--', linewidth=1.5)

    x_pos = df_metric["Initial"].min() 
    ax.text(x_pos, mean_diff + 0.001, f'{mean_diff:.3f}', color='gray', ha='left', va='bottom', fontsize=12, fontweight='bold')

    # Significance
    if metric in ["Precision", "Recall"]:
        p_val = sig_results[metric]["p_val_t"]
        if p_val < 0.05:
            direction = '↑' if sig_results[metric]["mean_processed"] > sig_results[metric]["mean_initial"] else '↓'
            ax.text(df_metric["Initial"].max(), mean_diff + 0.02, f'* {direction}', color='red', fontsize=17, ha='right', fontweight='bold')
    
    ax.set_title(f"{metric}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Average of Baseline and CRF", fontsize=12)
    ax.set_ylabel("Difference (CRF − Baseline)", fontsize=12)
    ax.grid(True, alpha=0.3)

fig.legend(
    handles=handles + [
        Line2D([0],[0], color='gray', linestyle='--', linewidth=2),
        Line2D([0],[0], color='red', linestyle='--', linewidth=1.5),
        Line2D([0],[0], marker='*', color='red', linestyle='None', markersize=6, label='p < 0.05 ↑/↓')
    ],
    labels=labels + ['Mean diff', '±1.96 SD','p < 0.05 ↑/↓'],
    loc='center right',
    borderaxespad=2,
    fontsize=10,
    frameon=True,
    framealpha=0.9
)

plt.tight_layout(rect=[0,0,0.85,1])

# Saving
output_path = r"C:\Users\PolDiaz\Desktop\PostProcessat\CRF_Results\CRF_N10_BlandPlot.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

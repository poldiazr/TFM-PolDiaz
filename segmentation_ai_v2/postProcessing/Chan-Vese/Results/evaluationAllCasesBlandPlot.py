import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon
from matplotlib.lines import Line2D

filepath = r"C:\Users\PolDiaz\Desktop\PostProcessat\Chan-Vese_Results\Results.xlsx"
T = pd.read_excel(filepath, sheet_name="Per_Vertebra", decimal=',')

metrics = ["Dice", "IoU", "Precision", "Recall"]
initial_cols = [f"{m}_Initial" for m in metrics]
processed_cols = [f"{m}_Refined" for m in metrics]

def extract_mean(s):
    if isinstance(s, str):
        return float(s.replace(',', '.').split("±")[0].strip())
    return float(s)

# Extracting data
data = {}
for metric, init_col, proc_col in zip(metrics, initial_cols, processed_cols):
    init_vals = T[init_col].apply(extract_mean)
    proc_vals = T[proc_col].apply(extract_mean)
    mask = init_vals.notna() & proc_vals.notna()
    
    data[metric] = pd.DataFrame({
        "Initial": init_vals[mask],
        "Refined": proc_vals[mask],
        "Vertebra": T["Vertebra"][mask]
    })

# Significance test
sig_results = {}
for metric in metrics:
    df_metric = data[metric]
    initial_vals = df_metric["Initial"]
    refined_vals = df_metric["Refined"]
    
    t_stat, p_val_t = ttest_rel(refined_vals, initial_vals)
    
    try:
        w_stat, p_val_w = wilcoxon(refined_vals, initial_vals)
    except ValueError:
        w_stat, p_val_w = np.nan, np.nan
    
    sig_results[metric] = {
        "p_val_t": p_val_t,
        "p_val_w": p_val_w,
        "mean_initial": np.mean(initial_vals),
        "mean_refined": np.mean(refined_vals)
    }

# Plot
sns.set(style="whitegrid", font_scale=1.2)
fig, axes = plt.subplots(2, 2, figsize=(11,9))
axes = axes.flatten()
fig.suptitle(
    r"$\bf{Impact\ of\ Chan-Vese\ Refinement\ on\ Segmentation\ Performance}$" "\nN = 10",
    fontsize=14, x=0.45
)

vertebras = sorted(T["Vertebra"].dropna().unique())
colors = plt.cm.Reds(np.linspace(0.5, 1.0, len(vertebras)))

for ax, metric in zip(axes, metrics):
    df_metric = data[metric]
    
    # Scatter per vertebra
    for i, vertebra in enumerate(vertebras):
        df_v = df_metric[df_metric["Vertebra"] == vertebra]
        mean_vals = (df_v["Initial"] + df_v["Refined"]) / 2
        diff_vals = df_v["Refined"] - df_v["Initial"]
        
        ax.scatter(mean_vals, diff_vals, color=colors[i], alpha=0.7)
    
    # Mean and ±1.96 SD
    mean_diff = np.mean(df_metric["Refined"] - df_metric["Initial"])
    sd_diff = np.std(df_metric["Refined"] - df_metric["Initial"], ddof=1)
    ax.axhline(mean_diff, color='gray', linestyle='--', linewidth=2)
    ax.axhline(mean_diff + 1.96*sd_diff, color='red', linestyle='--', linewidth=1.5)
    ax.axhline(mean_diff - 1.96*sd_diff, color='red', linestyle='--', linewidth=1.5)

    x_pos = df_metric["Initial"].min() 
    ax.text(x_pos, mean_diff + 0.001, f'{mean_diff:.3f}', color='gray', ha='left', va='bottom', fontsize=12, fontweight='bold')

    # Significance p-value
    p_val = sig_results[metric]["p_val_t"]
    if p_val < 0.05:
        direction = '↑' if sig_results[metric]["mean_refined"] > sig_results[metric]["mean_initial"] else '↓'
        ax.text(df_metric["Initial"].max(), mean_diff + 0.02, f'* {direction}', 
                color='blue', fontsize=14, ha='right', fontweight='bold')
    
    ax.set_title(f"{metric}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Average Baseline and Chan-Vese", fontsize=12)
    ax.set_ylabel("Difference (Chan-Vese − Baseline)", fontsize=12)
    ax.grid(True, alpha=0.3)

fig.legend(
    handles=[
        Line2D([0],[0], color='gray', linestyle='--', linewidth=2, label='Mean diff'),
        Line2D([0],[0], color='red', linestyle='--', linewidth=1.5, label='±1.96 SD'),
        Line2D([0],[0], marker='o', color='red', linestyle='None', markersize=6, alpha=0.7, label='All Vertebrae'),
        Line2D([0],[0], marker='*', color='blue', linestyle='None', markersize=6, label='p < 0.05 ↑/↓')
    ],
    loc='center right',
    borderaxespad=2,
    fontsize=10,
    frameon=True,
    framealpha=0.9
)

plt.tight_layout(rect=[0,0,0.85,1])

# Saving
output_path = r"C:\Users\PolDiaz\Desktop\PostProcessat\Chan-Vese_Results\ChanVese_BlandPlot_AllVertebra_Red.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

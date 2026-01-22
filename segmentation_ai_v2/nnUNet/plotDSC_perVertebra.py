import pandas as pd
import matplotlib.pyplot as plt
import os

excel_path = r"C:\Users\PolDiaz\Desktop\nnUNet_test\Results\DSC_per_vertebra.xlsx"
output_dir = r"C:\Users\PolDiaz\Desktop\nnUNet_test\Results\plots"
os.makedirs(output_dir, exist_ok=True)

# Loading data
df = pd.read_excel(excel_path)
vertebra_names = {
    3: "L5", 4: "L4", 5: "L3", 6: "L2", 7: "L1",
    8: "T12", 9: "T11", 10: "T10", 11: "T9", 12: "T8",
    13: "T7", 14: "T6", 15: "T5", 16: "T4", 17: "T3",
    18: "T2", 19: "T1",
    20: "C7", 21: "C6", 22: "C5", 23: "C4",
    24: "C3", 25: "C2", 26: "C1"
}
df["Vertebra"] = df["Label"].map(vertebra_names)

ordered_labels = ["C1", "C2", "C3", "C4", "C5", "C6", "C7",
                  "T1", "T2", "T3", "T4", "T5", "T6", "T7",
                  "T8", "T9", "T10", "T11", "T12",
                  "L1", "L2", "L3", "L4", "L5"]
df["Vertebra"] = pd.Categorical(df["Vertebra"], categories=ordered_labels, ordered=True)
df = df.sort_values("Vertebra")

# Extracting data
box_data = [df[df["Vertebra"]==v]["DSC_Fine"].values for v in ordered_labels]

# Plot
plt.figure(figsize=(14,6))
b = plt.boxplot(box_data, patch_artist=True)

for patch in b['boxes']:
    patch.set_facecolor("#DD0E0E")
for median in b['medians']:
    median.set_color("#000000")
    median.set_linewidth(1)

plt.xticks(range(1, len(ordered_labels)+1), ordered_labels, rotation=45)
plt.ylabel("Dice score")
plt.xlabel("Vertebra")
plt.title("Distribution of Dice Score (Fine-tuned model) per vertebra")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

# Saving result
plt.savefig(os.path.join(output_dir, "DSC_boxplot_per_vertebra.png"), dpi=300)
plt.show()

import os
import numpy as np
import nibabel as nib
from collections import Counter

# Folder
data_dir = r"C:\Users\PolDiaz\Desktop\nnUNet_test\labelsTs_remap"

# Vertebra labels
labels = list(range(26, 2, -1))  # [26, 25, ..., 3]

# Counter
presence_counts = Counter()
i = 0
# Iterating through the labels folder
for fname in os.listdir(data_dir):
    print(fname)
    if fname.endswith(".nii.gz"):
        path = os.path.join(data_dir, fname)
        img = nib.load(path)
        data = np.asanyarray(img.dataobj, dtype=np.int32)
        present_labels = np.unique(data)
        for label in labels:
            if label in present_labels:
                presence_counts[label] += 1

# Results
for label in labels:
    print(f"Label {label:>2}: {presence_counts.get(label, 0):>4}")

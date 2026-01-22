import os
import nibabel as nib
import numpy as np

# Paths definition
GT_FOLDER = r"C:\Users\PolDiaz\Desktop\nnUNet_test\labelsTs"
OUTPUT_FOLDER = r"C:\Users\PolDiaz\Desktop\nnUNet_test\labelsTs_remap"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Iterating through the folder
for filename in os.listdir(GT_FOLDER):
    if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
        continue

    input_path = os.path.join(GT_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, filename.replace(".nii", "_remap.nii"))

    print(f"Processing {filename}")

    img = nib.load(input_path)
    data = img.get_fdata().astype(np.int32)
    new_data = np.zeros_like(data)

    # Inverted relabel 1→26, 24→3
    for old in range(1, 25):
        new = 27 - old
        new_data[data == old] = new

    # Saving remaped volume
    nib.save(
        nib.Nifti1Image(new_data, img.affine, img.header),
        output_path
    )

print("\n Relabeling completed.")
print("Files saved:", OUTPUT_FOLDER)

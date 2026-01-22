import numpy as np
import nibabel as nib
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral
import matplotlib.pyplot as plt
import os

# Files
ct_path = r"C:\Users\PolDiaz\Desktop\PostProcessat\estudiN=5\imagesTs\sub-331_0000.nii.gz"
probs_path = r"C:\Users\PolDiaz\Desktop\PostProcessat\estudiN=5\Probs\sub-400.npz"
output_dir = r"C:\Users\PolDiaz\Desktop\PostProcessat\estudiN=5\Results\400\CRF"

# Load CT
ct_nii = nib.load(ct_path)
ct = ct_nii.get_fdata().astype(np.float32)
ct = (ct - np.min(ct)) / (np.max(ct) - np.min(ct) + 1e-8)

# Load probabilities
data = np.load(probs_path)
probs = data['probabilities'].astype(np.float32)
n_classes = probs.shape[0]

print("CT shape:", ct.shape)
print("Probs shape:", probs.shape)

# Transpose and flip
probs = np.transpose(probs, (0, 1, 3, 2))
#probs = np.flip(probs, axis=1)
probs = np.flip(probs, axis=3)

# Checking dimensions and orientations
z = ct.shape[2] // 2  # Central slice
save_dir = r"C:\Users\PolDiaz\Documents\GitHub\segmentation_ai_v2\postProcessing\CRFs\CheckOrientations"
os.makedirs(save_dir, exist_ok=True)  
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(ct[:, :, z], cmap='gray')
plt.title("CT slice")
plt.subplot(1,2,2)
plt.imshow(probs[22, :, :, z], cmap='hot')
plt.title("Probs slice class 1")
plt.tight_layout()
save_path = os.path.join(save_dir, f"slice_z{z}.png")
plt.savefig(save_path)
plt.close()

# Hyperparameter cases
cases = [
    {"gauss_sdims": (3,3), "gauss_compat": 4, "bilat_sdims": (5,5), "bilat_schan": 0.05, "bilat_compat": 4},
    {"gauss_sdims": (3,3), "gauss_compat": 2, "bilat_sdims": (5,5), "bilat_schan": 0.05, "bilat_compat": 8},
    {"gauss_sdims": (3,3), "gauss_compat": 1, "bilat_sdims": (5,5), "bilat_schan": 0.07, "bilat_compat": 12},
    {"gauss_sdims": (3,3), "gauss_compat": 2, "bilat_sdims": (7,7), "bilat_schan": 0.07, "bilat_compat": 14},
    {"gauss_sdims": (3,3), "gauss_compat": 1, "bilat_sdims": (5,5), "bilat_schan": 0.05, "bilat_compat": 16},
    {"gauss_sdims": (3,3), "gauss_compat": 1, "bilat_sdims": (7,7), "bilat_schan": 0.07, "bilat_compat": 16},
]

for idx, case in enumerate(cases, start=1):
    print(f"Processing case {idx}...")
    refined = np.zeros(ct.shape, dtype=np.uint8)

    for z in range(ct.shape[2]):
        img_slice = np.stack([ct[:, :, z]] * 3, axis=-1)
        probs_slice = probs[:, :, :, z] if probs.ndim == 4 else probs[:, :, z]

        d = dcrf.DenseCRF2D(img_slice.shape[1], img_slice.shape[0], n_classes)
        U = unary_from_softmax(probs_slice)
        d.setUnaryEnergy(U)

        # Pairwise Gaussian
        feats_gaussian = create_pairwise_gaussian(sdims=case["gauss_sdims"], shape=img_slice.shape[:2])
        d.addPairwiseEnergy(feats_gaussian, compat=case["gauss_compat"])

        # Pairwise Bilateral
        feats_bilateral = create_pairwise_bilateral(sdims=case["bilat_sdims"], 
                                                     schan=(case["bilat_schan"],), 
                                                     img=img_slice, 
                                                     chdim=2)
        d.addPairwiseEnergy(feats_bilateral, compat=case["bilat_compat"])

        # Inference
        Q = d.inference(5)
        refined[:, :, z] = np.argmax(Q, axis=0).reshape(img_slice.shape[:2])

    # Save output
    output_path = f"{output_dir}\\331_{idx}.nii.gz"
    refined_nii = nib.Nifti1Image(refined.astype(np.uint8), ct_nii.affine, ct_nii.header)
    nib.save(refined_nii, output_path)
    print(f"Saved case {idx} to {output_path}")

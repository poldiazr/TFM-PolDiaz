import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral

# Files
ct_path = r"C:\Users\PolDiaz\Desktop\PostProcessat\sub400\testImage.nii"
probs_path = r"C:\Users\PolDiaz\Desktop\PostProcessat\sub400\test.npz"
output_path = r"C:\Users\PolDiaz\Desktop\PostProcessat\sub400\crf_slice3D.nii.gz"

# Loading CT
ct_nii = nib.load(ct_path)
ct = ct_nii.get_fdata().astype(np.float32)

# Intensity normalization
ct = (ct - np.min(ct)) / (np.max(ct) - np.min(ct) + 1e-8)

# Loading probabilities form nnunetv2
data = np.load(probs_path)
probs = data['probabilities'].astype(np.float32) 
n_classes = probs.shape[0]

# Flopping probabilities array
probs = np.transpose(probs, (0, 2, 1, 3)) 
probs = np.flip(probs, axis=2)  # horizontal
probs = np.flip(probs, axis=1)  # vertical

print("CT shape:", ct.shape)
print("Probs shape:", probs.shape)
refined = np.zeros(ct.shape, dtype=np.uint8)

# Slice by slice
for z in range(ct.shape[2]):
    img_slice = np.stack([ct[:, :, z]] * 3, axis=-1)
    probs_slice = probs[:, :, :, z] if probs.ndim == 4 else probs[:, :, z]

    d = dcrf.DenseCRF2D(img_slice.shape[1], img_slice.shape[0], n_classes)
    U = unary_from_softmax(probs_slice)
    d.setUnaryEnergy(U)

    # Pairwise gaussian
    feats_gaussian = create_pairwise_gaussian(sdims=(3, 3), shape=img_slice.shape[:2])
    d.addPairwiseEnergy(feats_gaussian, compat=1)

    # Pairwise bilateral
    feats_bilateral = create_pairwise_bilateral(sdims=(5, 5), schan=(0.05,), img=img_slice, chdim=2)
    d.addPairwiseEnergy(feats_bilateral, compat=16)

    # Inference
    Q = d.inference(5)
    refined_slice = np.argmax(Q, axis=0).reshape(img_slice.shape[:2])

    # Saving slice
    refined[:, :, z] = refined_slice

# Saving final approach
refined_nii = nib.Nifti1Image(refined.astype(np.uint8), ct_nii.affine, ct_nii.header)
nib.save(refined_nii, output_path)
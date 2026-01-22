import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral

# Files
ct_path = r"C:\Users\PolDiaz\Desktop\PostProcessat\sub400\testImage.nii"
probs_path = r"C:\Users\PolDiaz\Desktop\PostProcessat\sub400\test.npz"  # aquí están las probabilidades guardadas

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

# Slice election
z_ref = ct.shape[2] // 2  
z_slices = list(range(max(0, z_ref-3), min(ct.shape[2], z_ref+4)))

for z in z_slices:
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
    refined[:, :, z] = refined_slice

    # Visualization
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    axs.imshow(ct[:, :, z], cmap='gray')
    axs.imshow(refined_slice, cmap='Blues', alpha=0.4)
    axs.set_title(f'Slice {z} - Refined mask')
    plt.tight_layout()
    plt.show()





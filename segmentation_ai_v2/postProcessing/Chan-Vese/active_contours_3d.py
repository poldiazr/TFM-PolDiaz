import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# === 1. Load CT and initial mask ===
ct_path   = r"C:\Users\PolDiaz\Desktop\PostProcessat\testImage.nii"
mask_path = r"C:\Users\PolDiaz\Desktop\PostProcessat\testMaskT2.nii"

ct = sitk.ReadImage(ct_path, sitk.sitkFloat32)
mask = sitk.ReadImage(mask_path, sitk.sitkFloat32)

print("Volumen cargado:", sitk.GetArrayFromImage(ct).shape)

# === 2. Normalize CT ===
ct_norm = sitk.RescaleIntensity(ct, 0.0, 1.0)

# === 3. Crear nivel inicial (signed distance) desde la máscara ===
init_ls = sitk.SignedMaurerDistanceMap(mask > 0,
                                       insideIsPositive=True,
                                       squaredDistance=False,
                                       useImageSpacing=True)

# === 4. Chan–Vese 3D ===
cv = sitk.ScalarChanAndVeseDenseLevelSetImageFilter()

cv.SetNumberOfIterations(50)
cv.SetCurvatureWeight(0.05)       # suavizado
cv.SetAreaWeight(0.0)            # bias a expansión/contracción
cv.SetLambda1(1.0)               # región interior
cv.SetLambda2(1.0)               # región exterior
#cv.SetReinitializationFrequency(10)

print("Ejecutando Chan–Vese 3D…")

levelset_out = cv.Execute(init_ls, ct_norm)

# === 5. Convert LS to binary mask ===
bw3D = levelset_out < 0  # interior = negativo

bw3D_np = sitk.GetArrayFromImage(bw3D).astype(np.uint8)

# === 6. Save result ===
out_sitk = sitk.GetImageFromArray(bw3D_np)
out_sitk.CopyInformation(ct)

out_path = r"C:\Users\PolDiaz\Desktop\PostProcessat\refined_mask_cv3D_sitk.nii.gz"
sitk.WriteImage(out_sitk, out_path)

print("✔ Chan–Vese 3D final guardado en:")
print(out_path)

# === 7. Visualizar slice central ===
z = bw3D_np.shape[0] // 2
plt.imshow(sitk.GetArrayFromImage(ct_norm)[z], cmap='gray')
plt.imshow(bw3D_np[z], cmap='Blues', alpha=0.4)
plt.title("Slice central con Chan–Vese 3D")
plt.show()

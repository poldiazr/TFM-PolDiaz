from totalsegmentator.python_api import totalsegmentator
import nibabel as nib

# Input image
img_path = r"C:\Users\PolDiaz\Desktop\DATASET\tailor\TailorSurgeryAI v2\250057\01_DCM_250057\3 Columna cervical 100 Br40 S3 Matriz 512.nii"
input_img = nib.load(img_path)

# Output directory
out_path = r"C:\Users\PolDiaz\Desktop\DATASET\tailor\TailorSurgeryAI v2\250057\02_SEG_250057\2021_vert_mask.nii"

# Inference
output_img = totalsegmentator(input_img)
nib.save(output_img, out_path)


"""
# HIP IMPLANT SPECIFIC TASK

# Input image
input_image = r"C:\Users\PolDiaz\Desktop\DATASET\tailor\TailorSurgeryAI v2\240097\01_DCM_240097\2 Columna  1.0  Bv40  3.nii"      
input_img = nib.load(input_image)

# Output directory
output_dir = r"C:\Users\PolDiaz\Desktop\DATASET\tailor\TailorSurgeryAI v2\240097\02_SEG_240104\2021_vert_mask_IMPLANT.nii"       

totalsegmentator(
    input_img,
    output_dir,
    task="hip_implant",  
    fast=False,        
    ml=False,          
    preview=False,     
)
"""


import os
import SimpleITK as sitk

# Folder path
input_folder = r"C:\Users\PolDiaz\Desktop\DATASET\csi2014"

# Iterating through the folder
for filename in os.listdir(input_folder):
    # For each .mha file
    if filename.endswith(".mha"):
        input_path = os.path.join(input_folder, filename)
        image = sitk.ReadImage(input_path)
        output_filename = filename.replace(".mha", ".nii.gz")
        output_path = os.path.join(input_folder, output_filename)
        
        # Saving NIfTI converted image
        sitk.WriteImage(image, output_path)
        
        print(f"Converted: {filename} → {output_filename}")

print("Conversion completed")

import os
import sys
from nnunetv2.run.run_training import run_training_entry as nnunet_train_entry

# Environment configuration
os.environ["nnUNet_raw"] = r"C:\Users\PolDiaz\Desktop\nnUNet\nnUNet_raw"
os.environ["nnUNet_preprocessed"] = r"C:\Users\PolDiaz\Desktop\nnUNet\nnUNet_preprocessed"
os.environ["nnUNet_results"] = r"C:\Users\PolDiaz\Desktop\nnUNet\nnUNet_results"

# Parameters
TASK_ID = 292
FOLD = 0
CONFIG = "3d_fullres"

# Running final validation only
sys.argv = [
    "nnUNetv2_train",
    str(TASK_ID),
    CONFIG,
    str(FOLD),
    "--val" 
]
nnunet_train_entry()
print("Validation finished.")

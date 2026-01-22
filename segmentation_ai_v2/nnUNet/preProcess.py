
import os
import sys

# Environment configuration
os.environ["nnUNet_raw"] = r"C:\Users\PolDiaz\Desktop\nnUNet\nnUNet_raw"
os.environ["nnUNet_preprocessed"] = r"C:\Users\PolDiaz\Desktop\nnUNet\nnUNet_preprocessed"
os.environ["nnUNet_results"] = r"C:\Users\PolDiaz\Desktop\nnUNet\nnUNet_results"

# plan_and_preprocess_entry import
from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry

# Dataset parameters definition
TASK_ID = 292  
VERIFY_DATASET = True  
NUM_THREADS = 1     

# Running
sys.argv = [
    "nnUNetv2_plan_and_preprocess",
    "-d", str(TASK_ID),
    "--verify_dataset_integrity" if VERIFY_DATASET else "",
    "-c", "3d_fullres",
    "-np", str(1)
]

os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
plan_and_preprocess_entry()
print("Preprocess done.")

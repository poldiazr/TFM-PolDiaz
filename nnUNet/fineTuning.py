import os
import torch
import numpy as np
import torch

# Environment configuration
os.environ["nnUNet_raw"] = r"C:\Users\PolDiaz\Desktop\nnUNet\nnUNet_raw"
os.environ["nnUNet_preprocessed"] = r"C:\Users\PolDiaz\Desktop\nnUNet\nnUNet_preprocessed"
os.environ["nnUNet_results"] = r"C:\Users\PolDiaz\Desktop\nnUNet\nnUNet_results"

# Intern requirements
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["nnUNet_n_proc_DA"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

# Parameters
TASK_ID = 292
FOLD = 0
DATASET = "Dataset292_Vertebrae"
CONFIGURATION = "3d_fullres"
PRETRAINED_WEIGHTS = (
    r"C:\Users\PolDiaz\.totalsegmentator\nnunet\results\Dataset292_TotalSegmentator_part2_vertebrae_1532subj"
    r"\nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres\fold_0\checkpoint_final.pth"
)
NUM_GPUS = 1 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import nnunetv2.run.load_pretrained_weights as lpw
from nnunetv2.run.run_training import run_training

# Loading pretrained weights depending on the GPUs/CPU disponibility

def load_pretrained_weights_safe(network, fname, verbose=True):
    try:
        torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    except AttributeError:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    
    checkpoint = torch.load(fname, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    network.load_state_dict(state_dict, strict=False)
    
    if verbose:
        print(f"Pretrained weights loaded correct from: {fname}")

lpw.load_pretrained_weights = load_pretrained_weights_safe

# Running run_training
try:
    run_training(
        dataset_name_or_id=DATASET,
        configuration=CONFIGURATION,
        fold=str(FOLD),
        pretrained_weights=PRETRAINED_WEIGHTS,
        num_gpus=NUM_GPUS,
        device=device
    )
except Exception as e:
    print("Error during the execution:")
    import traceback
    traceback.print_exc()

print("Fine-tune completed!")



import os
import sys
import torch

# Environment configuration
os.environ["nnUNet_raw"] = r"C:\Users\PolDiaz\Desktop\nnUNet\nnUNet_raw"
os.environ["nnUNet_preprocessed"] = r"C:\Users\PolDiaz\Desktop\nnUNet\nnUNet_preprocessed"
os.environ["nnUNet_results"] = r"C:\Users\PolDiaz\Desktop\nnUNet\nnUNet_results"

# Parameters
TASK_ID = 292
FOLDS = (0,) 
CONFIG = "3d_fullres"
CHECKPOINT_NAME = "checkpoint_final.pth"

INPUT_FOLDER = r"C:\Users\PolDiaz\Desktop\Test1\images"  # Tus imágenes de test
OUTPUT_FOLDER = r"C:\Users\PolDiaz\Desktop\Test1\TestPredictions"

# Creating predictor
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_device=True,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose=True,
    verbose_preprocessing=False,
    allow_tqdm=True
)

# Model fine-tuned specification
model_folder = os.path.join(
    os.environ["nnUNet_results"],
    f"Dataset{TASK_ID}_Vertebrae",
    f"nnUNetTrainer__nnUNetPlans__{CONFIG}"
)

predictor.initialize_from_trained_model_folder(
    model_training_output_dir=model_folder,
    use_folds=FOLDS,
    checkpoint_name=CHECKPOINT_NAME
)

from os.path import join
# Inference execution
input_files = [[join(INPUT_FOLDER, f)] for f in os.listdir(INPUT_FOLDER) if f.endswith(".nii") or f.endswith(".nii.gz")]

predictor.predict_from_files_sequential(
    list_of_lists_or_source_folder=input_files,
    output_folder_or_list_of_truncated_output_files=OUTPUT_FOLDER,
    save_probabilities=False,
    overwrite=True,
    folder_with_segs_from_prev_stage=None
)
print("Inference done")


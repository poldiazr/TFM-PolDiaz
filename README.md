# Automatic Vertebra Segmentation in CT Scans

This repository contains the **code and pipeline developed for automatic vertebra segmentation in CT scans**, aimed at accelerating and supporting 3D spine surgical planning. The pipeline combines deep learning, fine-tuning, post-processing, and clinical validation.

This repository is **public** and does **not include any confidential data**. 

## Project Overview

The main goal of this project was to develop a reliable pipeline for vertebra segmentation, including:

- Adapting a generic multi-organ **nnU-Net** model to vertebra-focused CT datasets.
- Applying **post-processing techniques** such as Chan–Vese active contours and Conditional Random Fields (CRFs) to refine boundaries and reduce false positives.
- Performing a **quantitative evaluation** at vertebral region and individual vertebra levels (Dice, IoU and Hausdorff distance).
- Validating results visually and with clinical experts to ensure anatomical consistency for surgical planning.

## Repository Structure

- **nnUNet**: Contains all files for **preprocessing**, performing the necessary **remapping**, **fine-tuning**, and **quantitative analysis**.  It includes also a `results` folder with some of the **quantitative outcomes**.
- **postProcessing**: Contains two subfolders, one for each **post-processing technique** used. Each subfolder includes a `results` folder with some of the **quantitative outcomes**.  
  *Note: The Chan–Vese method was implemented in **MATLAB**, not Python.*

  ## Author

- **Pol Díaz** – Master's Thesis – Universitat Politècnica de Catalunya
- Contact: poldiazr01@gmail.com

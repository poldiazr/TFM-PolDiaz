%% Vertebra Refinement and Metrics (3D)
% Applies Chan–Vese slice by slice
clear; close all; clc;

%% 1. Load CT and masks
ct_path = '/Volumes/TOSHIBA EXT/POL/TAILOR/PostProcessat/estudiN=5/imagesTs/sub-400_0000.nii';
mask_path = '/Users/POL/Desktop/PostProcessat/Chan-Vese/Reference Case/Predicted/C5.nii.gz';
gt_path = '/Users/POL/Desktop/PostProcessat/Chan-Vese/Reference Case/Ground Truth/C5.nii.gz';

ct = niftiread(ct_path);
mask = niftiread(mask_path) > 0;
GT = niftiread(gt_path) > 0;

info = niftiinfo(ct_path);
disp('Volume loaded.');

%% 2. Initialize variables
refined_mask = zeros(size(mask));   % 3D refined mask
num_iter = 10;                      % iterations per slice
smooth_factor = 0;
contraction_factor = -0.05;

%% 3. Iterate over slices
for z = 1:size(ct,3)
    I = mat2gray(ct(:,:,z));
    bw = mask(:,:,z);

    if nnz(bw) == 0
        refined_mask(:,:,z) = bw;
        continue;  % skip empty slices
    end

    % Chan–Vese refinement
    for i = 1:num_iter
        bw = activecontour(I, bw, 1, 'Chan-vese', ...
                           'SmoothFactor', smooth_factor, ...
                           'ContractionBias', contraction_factor);
    end

    refined_mask(:,:,z) = bw;
end

%% 4. 3D Metrics
diceScore = @(A,B) 2*nnz(A & B)/(nnz(A)+nnz(B));
iouScore = @(A,B) nnz(A & B)/nnz(A | B);

dice_initial_3D = diceScore(mask, GT);
dice_refined_3D = diceScore(refined_mask, GT);
iou_initial_3D = iouScore(mask, GT);
iou_refined_3D = iouScore(refined_mask, GT);

fprintf('Dice 3D - Initial: %.4f\n', dice_initial_3D);
fprintf('Dice 3D - Refined: %.4f\n', dice_refined_3D);
fprintf('IoU 3D - Initial: %.4f\n', iou_initial_3D);
fprintf('IoU 3D - Refined: %.4f\n', iou_refined_3D);

%% 5. Precision & Recall
TP = @(A,B) nnz(A & B);
FP = @(A,B) nnz(A & ~B);
FN = @(A,B) nnz(~A & B);

prec_initial = TP(mask, GT) / (TP(mask, GT) + FP(mask, GT));
recall_initial = TP(mask, GT) / (TP(mask, GT) + FN(mask, GT));
prec_refined = TP(refined_mask, GT) / (TP(refined_mask, GT) + FP(refined_mask, GT));
recall_refined = TP(refined_mask, GT) / (TP(refined_mask, GT) + FN(refined_mask, GT));

fprintf('Precision 3D - Initial: %.4f\n', prec_initial);
fprintf('Precision 3D - Refined: %.4f\n', prec_refined);
fprintf('Recall 3D - Initial: %.4f\n', recall_initial);
fprintf('Recall 3D - Refined: %.4f\n', recall_refined);

%% 6. Save refined mask
refined_mask_path = '/Users/POL/Desktop/TestPostPro/Chan-Vese/Refined/C4.nii.gz';
refined_mask_uint8 = uint8(refined_mask);  % binary mask

mask_info = info;
mask_info.Datatype = 'uint8';
mask_info.BitsPerPixel = 8;

niftiwrite(refined_mask_uint8, refined_mask_path, mask_info);
disp(['Refined mask saved at: ', refined_mask_path]);

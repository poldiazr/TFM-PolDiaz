% Requires: Image Processing Toolbox

clear; close all; clc;

% Paths to the files
ct_path = '/Volumes/TOSHIBA EXT/POL/TAILOR/PostProcessat/estudiN=5_2/imagesTs/sub-398_0000.nii.gz';
mask_path = '/Users/POL/Downloads/398/Predictions/L2.nii.gz';
gt_path = '/Users/POL/Downloads/398/Ground Truth/L2.nii.gz';

% Loading volumes 
ct = niftiread(ct_path);
mask = niftiread(mask_path) > 0;   
GT = niftiread(gt_path) > 0;       
info = niftiinfo(ct_path);

% Testing parameters
smooth_vals = 0:0.05:0.3;            % SmoothFactor
contraction_vals = [-0.1, -0.05, 0, 0.05, 0.1]; % ContractionBias

best_dice = 0;
best_params = struct('SmoothFactor',0,'ContractionBias',0);

% Iterating through all the combinations 
for s = smooth_vals
    for c = contraction_vals
        refined_mask_tmp = zeros(size(mask));

        % Slice by slice
        for z = 1:size(ct,3)
            I = mat2gray(ct(:,:,z));
            bw = mask(:,:,z);

            if nnz(bw) == 0
                refined_mask_tmp(:,:,z) = bw;
                continue;
            end
            num_iter = 10;
            for i = 1:num_iter
                bw = activecontour(I, bw, 1, 'Chan-vese', ...
                           'SmoothFactor', s, ...
                           'ContractionBias', c);
            end

            refined_mask_tmp(:,:,z) = bw;

        end

        % Dice Score 3D
        diceScore = @(A,B) 2 * nnz(A & B) / (nnz(A) + nnz(B));
        dice_tmp = diceScore(refined_mask_tmp, GT);

        fprintf('Smooth=%.2f, Contraction=%.2f -> Dice=%.4f\n', s, c, dice_tmp);

        % Saving if is the best
        if dice_tmp > best_dice
            best_dice = dice_tmp;
            best_params.SmoothFactor = s;
            best_params.ContractionBias = c;
        end
    end
end

% Printing best combination
fprintf('SmoothFactor = %.2f\n', best_params.SmoothFactor);
fprintf('ContractionBias = %.2f\n', best_params.ContractionBias);
fprintf('Dice Score = %.4f\n', best_dice);


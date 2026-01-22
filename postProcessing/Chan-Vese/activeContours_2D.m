%% Vertebra Refinement with Iterative Visualization (2D)
% Requires: Image Processing Toolbox

clear; close all; clc;

%% 1. Load CT and mask (.nii)
ct_path = '/Volumes/TOSHIBA EXT/POL/TAILOR/PostProcessat/estudiN=5/imagesTs/sub-400_0000.nii';
mask_path = '/Users/POL/Desktop/PostProcessat/Chan-Vese/Reference Case/Predicted/C5.nii.gz';
gt_path = '/Users/POL/Desktop/PostProcessat/Chan-Vese/Reference Case/Ground Truth/C5.nii.gz';

ct = niftiread(ct_path);
mask = niftiread(mask_path);
GT = niftiread(gt_path);

info = niftiinfo(ct_path);
disp('Volume loaded.');

%% 2. Select axial slice (adjust manually)
slice_num = 103;              % slice to analyze
I = squeeze(ct(:, :, slice_num));
M = squeeze(mask(:, :, slice_num)) > 0;
GT = squeeze(GT(:, :, slice_num)) > 0;

% Display initial slice with mask
figure;
imshow(mat2gray(I), []);
hold on;
contour(M, [0.5 0.5], 'r', 'LineWidth', 2);
title('Initial Mask vs Slice');

%% 3. Iterative Chan–Vese refinement
bw = M;               % initial mask
num_iter = 10;        % number of iterations
smooth_factor = 0;
contraction_factor = -0.05

figure('Name','Active Contour Evolution','NumberTitle','off');

for i = 1:num_iter
    bw = activecontour(I, bw, 1, 'Chan-vese', ...
                       'SmoothFactor', smooth_factor, ...
                       'ContractionBias', contraction_factor);

    imshow(I, []);
    hold on;
    contour(bw, [0.5 0.5], 'g', 'LineWidth', 2);
    contour(M, [0.5 0.5], 'r--', 'LineWidth', 1);
    title(['Iteration ', num2str(i)]);
    legend({'Refined','Initial'}, 'TextColor','w');
    drawnow;
    pause(0.05); % controls update speed
end

%% 4. Show final result
figure;
imshow(I, []);
hold on;
contour(M, [0.5 0.5], 'r--', 'LineWidth', 1.5);
contour(bw, [0.5 0.5], 'g', 'LineWidth', 2);
title('Final Result');
legend({'Initial','Refined'});

%% 5. Dice Score
diceScore = @(A,B) 2 * nnz(A & B) / (nnz(A) + nnz(B));

dice_initial = diceScore(M, GT);
dice_chanvese = diceScore(bw, GT);

fprintf('Dice score initial mask: %.4f\n', dice_initial);
fprintf('Dice score after Chan–Vese: %.4f\n', dice_chanvese);

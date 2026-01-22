%% Refinamiento de vértebra 3D con Chan-Vese y guardado compatible con Slicer
clear; close all; clc;

% === 1. Cargar CT y máscara inicial ===
ct_path   = '/Users/POL/Desktop/250086/ct_cropped.nii.gz';
mask_path = '/Users/POL/Desktop/mask_refinada_3D.nii.gz';

ct_nii   = niftiread(ct_path);
mask_nii = niftiread(mask_path) > 0;  % binaria
info     = niftiinfo(ct_path);

disp('Volumen cargado.');

% === 2. Normalizar CT ===
I = mat2gray(ct_nii);

% === 3. (Opcional) Definir ROI pequeña alrededor de la vértebra ===
I_crop = I;      % aquí podrías recortar si quieres
M_crop = mask_nii;

% === 4. Aplicar contorno activo 3D (Chan-Vese) ===
disp('Ejecutando Chan-Vese 3D... (puede tardar)');
num_iter       = 50;      % número de iteraciones
smooth_factor  = 0.3;
contraction    = 0.15;

bw3D = activecontour(I_crop, M_crop, num_iter, 'Chan-vese', ...
                     'SmoothFactor', smooth_factor, ...
                     'ContractionBias', contraction);



%% Guardar labelmap 3D compatible con Slicer
refined_mask = uint8(bw3D > 0);  % 0/1 binaria

% Crear header nuevo basado en CT
labelmap_info = info;         % copiar información
labelmap_info.Datatype = 'uint8';
labelmap_info.BitsPerPixel = 8;
labelmap_info.Description = 'Refined vertebra mask (labelmap)';

out_path = '/Users/POL/Desktop/250086/refined_mask_3D_labelmap.nii.gz';
niftiwrite(refined_mask, out_path, labelmap_info, 'Compressed', true);

disp(['✅ Refinamiento guardado como labelmap compatible: ', out_path]);

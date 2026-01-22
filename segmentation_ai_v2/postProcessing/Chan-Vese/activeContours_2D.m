%% Refinamiento de vértebra con visualización iterativa
% Requiere: Image Processing Toolbox

clear; close all; clc;

% === 1. Cargar CT y máscara (.nii) ===
ct_path = '/Users/POL/Desktop/250086/ct_cropped.nii.gz';
mask_path = '/Users/POL/Desktop/mask_refinada_3D.nii.gz';

ct = niftiread(ct_path);
mask = niftiread(mask_path);

info = niftiinfo(ct_path);
disp('Volumen cargado.');

% === 2. Seleccionar slice axial (ajustar manualmente si quieres) ===
sagital_slice = round(size(ct, 1)/2); % mitad del volumen
I = squeeze(ct(sagital_slice, :, :));  % plano sagital
M = squeeze(mask(sagital_slice, :, :)) > 0;

% Convertir a double normalizado
I = mat2gray(I);

% === 3. Mostrar el estado inicial ===
figure('Name','Evolución del contorno activo','NumberTitle','off');
imshow(I, []);
hold on;
contour(M, [0.5 0.5], 'r', 'LineWidth', 2);
title('Segmentación inicial');
pause(1);

% === 4. Iterar paso a paso para ver evolución ===
bw = M; % máscara inicial
num_iter = 50; % número de iteraciones
smooth_factor = 0.3;

for i = 1:num_iter
    bw = activecontour(I, bw, 1, 'Chan-vese', ...
                       'SmoothFactor', smooth_factor, ...
                       'ContractionBias', 0);

    imshow(I, []);
    hold on;
    contour(bw, [0.5 0.5], 'g', 'LineWidth', 2);
    contour(M, [0.5 0.5], 'r--', 'LineWidth', 1);
    title(['Iteración ', num2str(i)]);
    legend({'Refinada','Inicial'}, 'TextColor','w');
    drawnow;
    pause(0.05); % controla velocidad de actualización
end

% === 5. Mostrar resultado final ===
figure;
imshow(I, []);
hold on;
contour(M, [0.5 0.5], 'r--', 'LineWidth', 1.5);
contour(bw, [0.5 0.5], 'g', 'LineWidth', 2);
title('Resultado final del refinamiento');
legend({'Inicial','Refinada'});

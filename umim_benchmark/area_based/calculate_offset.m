clear; clc;

baseDir = 'D:\Research\Paper\BRIGHT\ESSD_submission\Data\image_matching\noto_earthquake\';

refImageName = 'Noto_Earthquake_20240101_AOI01A_pre_disaster.tif';
senImageName = 'Noto_Earthquake_20240101_AOI01A_post_disaster_before_registration.tif';
manualCPFileName = 'Noto_Earthquake_20240101_AOI01A_manual_control_point.txt';
autoCPFileName = 'Noto_Earthquake_20240101_AOI01A_detected_feature_points.txt';

refImagePath = fullfile(baseDir, refImageName);
senImagePath = fullfile(baseDir, senImageName);
% Load Reference and Sensed Images with Geo-Info
[Ref, R_ref] = readgeoraster(refImagePath);
[Sen, R_sen] = readgeoraster(senImagePath);

autoCPPath = fullfile(baseDir, autoCPFileName);

% 读取test1.pts的像素坐标（跳过前6行）
CPs = readmatrix(autoCPPath, 'NumHeaderLines', 6);

x_ref_pixel = CPs(:,1) * 2;
y_ref_pixel = CPs(:,2) * 2;
x_sen_pixel = CPs(:,3) * 2;
y_sen_pixel = CPs(:,4) * 2;

% calculate tform (pixel-level registration)
tform = fitgeotform2d([x_sen_pixel, y_sen_pixel], [x_ref_pixel, y_ref_pixel], 'projective');
% tform_2 = fitgeotform2d([x_sen_pixel, y_sen_pixel], [x_ref_pixel, y_ref_pixel], 'pwl');

% Load manually selected controal points (mapX, mapY, sourceX, sourceY)
manualCPPath = fullfile(baseDir, manualCPFileName);
opts = detectImportOptions(manualCPPath,'FileType','text');
opts.DataLines = [3 Inf];
opts.SelectedVariableNames = 1:4;
GCPs = readmatrix(manualCPPath, opts);

mapX = GCPs(:,1);
mapY = GCPs(:,2);
sourceX = GCPs(:,3);
sourceY = GCPs(:,4);

% mapX, mapY (real-wolrd) → (pixel position in image)
[pixelX_sen, pixelY_sen] = worldToIntrinsic(R_sen, sourceX, sourceY);

% transform using pixel-based tform
[pixelX_ref, pixelY_ref] = transformPointsForward(tform, pixelX_sen, pixelY_sen);

% transform back to real-world coordinates
[mapX_ref, mapY_ref] = intrinsicToWorld(R_ref, pixelX_ref, pixelY_ref);

% calculate mean spatial offset
error_meter = sqrt((mapX_ref - mapX).^2 + (mapY_ref - mapY).^2);
mean_offset_meter = mean(error_meter);

fprintf('Geo-transform Mean Offset = %.4f meters\n', mean_offset_meter)

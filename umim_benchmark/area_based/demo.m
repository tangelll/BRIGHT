clear; clc;

%% === USER CONFIGURATION ===
refImagePath = 'D:\Research\Paper\BRIGHT\ESSD_submission\Data\image_matching\noto_earthquake\Noto_Earthquake_20240101_AOI01A_pre_disaster.tif';
senImagePath = 'D:\Research\Paper\BRIGHT\ESSD_submission\Data\image_matching\noto_earthquake\Noto_Earthquake_20240101_AOI01A_post_disaster_before_registration.tif';
gcpFilePath  = 'D:\Research\Paper\BRIGHT\ESSD_submission\Data\image_matching\noto_earthquake\Noto_Earthquake_20240101_AOI01_A_manual_control_point.txt';
resizeFactor = 0.5;  % Resize factor for matching step

featureType = 'CFOG'; % Options: 'CFOG', 'FHOG', 'FLSS', 'FSURF'

%% === ADD PATHS ===
addpath('.\pixel-wise feature represatation\CFOG');
addpath('.\pixel-wise feature represatation\FHOG');
addpath('.\pixel-wise feature represatation\FLSS');
addpath('.\pixel-wise feature represatation\FSURF');

%% === LOAD & RESIZE IMAGES ===
[Ref_full, R_ref] = readgeoraster(refImagePath);
[Sen_full, R_sen] = readgeoraster(senImagePath);

im_Ref = imresize(Ref_full, resizeFactor);
im_Sen = imresize(Sen_full, resizeFactor);

%% === FEATURE MATCHING ===
fprintf('Running feature matching using %s...\n', featureType);
tic;
[CP_Ref, CP_Sen] = matchFramework(im_Ref, im_Sen, featureType);
fprintf('Feature matching completed in %.2f seconds.\n', toc);

%% === FILTER OUTLIERS ===
[corrRefPt, corrSenPt] = ErrorDect(CP_Ref, CP_Sen, 0, 10);
corrPT = [corrRefPt,corrSenPt];%correct match

path = 'D:\test2.txt'
Wenvifile1(corrPT',path);

% Upscale coordinates to full-res
x_ref_pixel = corrRefPt(:,1) / resizeFactor;
y_ref_pixel = corrRefPt(:,2) / resizeFactor;
x_sen_pixel = corrSenPt(:,1) / resizeFactor;
y_sen_pixel = corrSenPt(:,2) / resizeFactor;

%% === COMPUTE GEO TRANSFORM ===
tform = fitgeotform2d([x_sen_pixel, y_sen_pixel], [x_ref_pixel, y_ref_pixel], 'projective');

%% === VISUALIZE MATCHED POINTS ===
figure;
imshow(Ref_full); hold on;
plot(x_ref_pixel, y_ref_pixel, 'yo', 'MarkerFaceColor', 'y'); title('Reference Image');

figure;
imshow(Sen_full); hold on;
plot(x_sen_pixel, y_sen_pixel, 'yo', 'MarkerFaceColor', 'y'); title('Sensed Image');

%% === EVALUATE AGAINST GROUND TRUTH GCP ===
opts = detectImportOptions(gcpFilePath,'FileType','text');
opts.DataLines = [3 Inf];
opts.SelectedVariableNames = 1:4;
GCPs = readmatrix(gcpFilePath, opts);

mapX = GCPs(:,1); mapY = GCPs(:,2);
sourceX = GCPs(:,3); sourceY = GCPs(:,4);

[pixelX_sen, pixelY_sen] = worldToIntrinsic(R_sen, sourceX, sourceY);
[pixelX_ref, pixelY_ref] = transformPointsForward(tform, pixelX_sen, pixelY_sen);
[mapX_ref, mapY_ref] = intrinsicToWorld(R_ref, pixelX_ref, pixelY_ref);

error_meter = sqrt((mapX_ref - mapX).^2 + (mapY_ref - mapY).^2);
mean_offset_meter = mean(error_meter);

fprintf('Geo-transform Mean Offset = %.4f meters\n', mean_offset_meter)

%% === OPTIONAL: EXPORT TO CSV ===
% T = table(mapX, mapY, sourceX, sourceY, mapX_ref, mapY_ref, error_meter);
% writetable(T, 'Registered_GCPs_Using_tform.csv');

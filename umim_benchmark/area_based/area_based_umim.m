
clear;
% add the path of feature extraction functions
addpath('.\pixel-wise feature represatation\CFOG');
addpath('.\pixel-wise feature represatation\FHOG');
addpath('.\pixel-wise feature represatation\FLSS');
addpath('.\pixel-wise feature represatation\FSURF');

% optical and sar matching
baseDir = 'D:\Research\Paper\BRIGHT\ESSD_submission\Data\image_matching\noto_earthquake\';

refImageName = 'Noto_Earthquake_20240101_AOI01A_pre_disaster.tif';
senImageName = 'Noto_Earthquake_20240101_AOI01A_post_disaster_before_registration.tif';
autoCPFileName = 'Noto_Earthquake_20240101_AOI01A_detected_feature_points.txt';
featureType = 'CFOG'; % Options: 'CFOG', 'FHOG', 'FLSS', 'FSURF'

% Build full paths
refImagePath = fullfile(baseDir, refImageName);
senImagePath = fullfile(baseDir, senImageName);

% Load images
im_Ref = imread(refImagePath);
im_Sen = imread(senImagePath);
%% Resize the input
im_Ref = imresize(im_Ref, 0.5);
im_Sen = imresize(im_Sen, 0.5);

%% image matching using the proposed framework.
tic
%using CFOG
[CP_Ref,CP_Sen] = matchFramework(im_Ref, im_Sen, featureType);
%using FHOG
%[CP_Ref,CP_Sen] = matchFramework(im_Ref,im_Sen,CP_initial_file,'FHOG');
%using FLSS
%[CP_Ref,CP_Sen] = matchFramework(im_Ref,im_Sen,CP_initial_file,'FLSS');
%using FSURF
%[CP_Ref,CP_Sen] = matchFramework(im_Ref,im_Sen,CP_initial_file,'FSURF');

fprintf('the total matching time is %f\n',toc);

%% detect the error
[corrRefPt, corrSenPt] = ErrorDect(CP_Ref, CP_Sen, 0, 10);
%wirite the point in the envi format
corrPT = [corrRefPt,corrSenPt];%correct match

% tform = fitgeotform2d(corrSenPt, corrRefPt, "pwl");

outputPath = fullfile(baseDir, autoCPFileName);

Wenvifile1(corrPT',outputPath);
%diplay the tie points
figure;
imshow(im_Ref),hold on;
plot(corrRefPt(:,1),corrRefPt(:,2),'yo','MarkerEdgeColor','k','MarkerFaceColor','y','MarkerSize',5);hold on;
title('reference image');

figure;
imshow(im_Sen),hold on;
plot(corrSenPt(:,1),corrSenPt(:,2),'yo','MarkerEdgeColor','k','MarkerFaceColor','y','MarkerSize',5);hold on;
title('sensed image');
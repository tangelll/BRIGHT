clc;
clear;
close all;
addpath(genpath(pwd));

% Path to the text file containing the prefixes
prefixListFiles = {
    % 'D:/Research/GithubRepo/BRIGHT/UMCD_benchmark/splitname/evaluation_umcd.txt';
    'D:/Research/GithubRepo/BRIGHT/UMCD_benchmark/splitname/train_umcd.txt' 
    % Add more files as needed
};
% Paths to folders
datasetFolder = 'D:/Research/Dataset/BRIGHT/BRIGHT_final';


combinedPrefixes = {};

%% Loop through each prefix list file and combine the prefixes
for k = 1:length(prefixListFiles)
    prefixListFile = prefixListFiles{k};
    
    % Read the list of prefixes from the current text file
    fileID = fopen(prefixListFile, 'r');
    prefixes = textscan(fileID, '%s'); % Each line contains a prefix
    fclose(fileID);
    prefixes = prefixes{1};
    
    % Append the prefixes to the combined list
    combinedPrefixes = [combinedPrefixes; prefixes];
end
% Read the list of prefixes from the text file


%% Initialize aggregated confusion matrix
totalTP = 0;
totalFP = 0;
totalTN = 0;
totalFN = 0;
% Preprocess images
opt_type_t1 = 'optical'; % Set to 'optical' for optical images
opt_type_t2 = 'optical'; % Set to 'optical' for optical images
% Parameters for IRG-McS
opt_Ns = 5000;
opt_SupSseg = 'v2';
opt_Niter = 6;
opt_lambda = 0.1;

%% Loop through each prefix in the list
for i = 1:length(combinedPrefixes)
    prefix = combinedPrefixes{i};
    fprintf('Processing prefix: %s\n', prefix);
    
    % Construct full paths to images and labels
    labelPath = fullfile(datasetFolder, 'target', [prefix '_building_damage.tif']);
    preImagePath = fullfile(datasetFolder, 'pre-event', [prefix '_pre_disaster.tif']);
    postImagePath = fullfile(datasetFolder, 'post-event', [prefix '_post_disaster.tif']);
    
    % Load images and ground truth
    image_t1 = imread(preImagePath);
    image_t2 = imread(postImagePath);
    gt = imread(labelPath);
    Ref_gt = double(gt(:,:,1)); % Assuming the ground truth is single-channel
    
    Ref_gt(Ref_gt == 1) = 0;
    Ref_gt(Ref_gt == 2) = 0; % You can set this to 2 to exclude damaged class from evaluation
    Ref_gt(Ref_gt >= 3) = 1;
    
    % Skip the sample without any destroyed buildings, which is too
    % difficult for the UMCD approaches
    if sum(sum(Ref_gt)) == 0
        continue
    end
    % Preprocess image_t2: Convert single-band to three-channel
    % if size(image_t2, 3) == 1
    %     image_t2 = repmat(image_t2, 1, 1, 3); % Duplicate the single band into 3 channels
    % end
    
    image_t1 = image_normlized(image_t1, opt.type_t1);
    image_t2 = image_normlized(image_t2, opt.type_t2);
    
    % IRG-McS processing
    [Cosup, ~] = SLIC_Cosegmentation_v2(image_t1, image_t2, opt.Ns, 1);
    CoNs = max(Cosup(:));
    [t1_feature, t2_feature] = MSMfeature_extraction(Cosup, image_t1, image_t2);
    iter = 1;
    Kmax =round(size(t1_feature,2).^0.5);
    Kmin = round(Kmax/10);
    [Kmat] = adaptiveKmat(t1_feature,t1_feature,t2_feature,t2_feature,Kmax,Kmin);
    labels = zeros(size(t1_feature,2),1);
    % Iterative framework
    labels = zeros(size(t1_feature, 2), 1);

    while iter<=(opt.Niter)
        idex_unchange = labels == 0;
        t1_feature_lib = t1_feature(:, idex_unchange);
        t2_feature_lib = t2_feature(:, idex_unchange);
        [fx,fy] = CalculateChangeLevel(t1_feature_lib,t1_feature,t2_feature_lib,t2_feature,Kmat);% dist
        fx_result(:,iter) = remove_outlier(fx);
        fy_result(:,iter) = remove_outlier(fy);
        [CM_map, labels] = MRF_CoSegmentation(Cosup,opt.lambda,t1_feature,t2_feature,fx_result(:,iter),fy_result(:,iter));
        CM_map_result(:,:,iter) = CM_map;
        % imshow(CM_map, [])
        iter = iter+1;
    end
    
    % Update aggregated confusion matrix
    % Ref_gt = Ref_gt/max(Ref_gt(:));
    [tp, fp, tn, fn, fplv,fnlv,~,~,pcc,kappa,imw] = performance(CM_map_result(:,:,opt.Niter), Ref_gt, 2);
    totalTP = totalTP + tp;
    totalFP = totalFP + fp;
    totalTN = totalTN + tn;
    totalFN = totalFN + fn;
    kappa
    clear fx_result
    clear fy_result
    clear CM_map_result
end

%% Calculate final metrics from aggregated confusion matrix
overallAccuracy = (totalTP + totalTN) / (totalTP + totalFP + totalTN + totalFN);
precision = totalTP / (totalTP + totalFP);
recall = totalTP / (totalTP + totalFN);
IoU = totalTP / (totalTP + totalFP + totalFN);
F1_score = 2 * (precision * recall) / (precision + recall);
kappa = (overallAccuracy - ((totalTP + totalFP) * (totalTP + totalFN) + ...
         (totalTN + totalFP) * (totalTN + totalFN)) / ...
         (totalTP + totalFP + totalTN + totalFN)^2) / ...
         (1 - ((totalTP + totalFP) * (totalTP + totalFN) + ...
         (totalTN + totalFP) * (totalTN + totalFN)) / ...
         (totalTP + totalFP + totalTN + totalFN)^2);

% Display final results
fprintf('Overall Accuracy: %.4f\n', overallAccuracy);
fprintf('Precision: %.4f\n', precision);
fprintf('Recall: %.4f\n', recall);
fprintf('IoU: %.4f\n', IoU);
fprintf('F1 Score: %.4f\n', F1_score);
fprintf('Kappa: %.4f\n', kappa);

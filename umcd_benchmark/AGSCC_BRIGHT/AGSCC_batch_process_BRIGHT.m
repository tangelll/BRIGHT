%%  Image Regression with Structure Cycle Consistency for Heterogeneous Change Detection
%{
Code: AGSCC - 2022
This is a test program for the Adaptive Graph and Structure Cycle Consistency (AGSCC) for heterogeneous change detection.

If you use this code for your research, please cite our paper. Thank you!

Sun, Yuli, et al. "Image Regression with Structure Cycle Consistency for Heterogeneous Change Detection."
IEEE Transactions on Neural Networks and Learning Systems, 2022
===================================================
%}

clc;
clear;
close all;
addpath(genpath(pwd));

% Path to the text file containing the prefixes
prefixListFiles = {
    % 'D:/Research/Dataset/BRIGHT/BRIGHT_20250103/splitname/train_setlevel.txt'; 
    % 'D:/Research/Dataset/BRIGHT/BRIGHT_20250103/splitname/test_setlevel.txt' 
    'D:/Research/GithubRepo/BRIGHT/UMCD_benchmark/splitname/train_umcd.txt' 
    % 'D:/Research/Dataset/BRIGHT/BRIGHT_20250103/splitname/val_setlevel.txt' 
    % Add more files as needed
};
% Paths to folders
datasetFolder = 'D:/Research/Dataset/BRIGHT/BRIGHT_final';


combinedPrefixes = {};

% Loop through each prefix list file and combine the prefixes
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

fprintf(['\n Data loading is completed...... ' '\n'])

% Initialize aggregated confusion matrix
totalTP = 0;
totalFP = 0;
totalTN = 0;
totalFN = 0;
%% Parameter setting

% With different parameter settings, the results will be a little different
% Ns: the number of superpxiels,  A larger Ns will improve the detection granularity, but also increase the running time. 5000 <= Ns <= 10000 is recommended.
% Niter_AGL: the maximum iteration number of Adaptive Graph Learning, Niter_AGL = 5 is recommended.
% Niter_SCC: the maximum iteration number of Structure Cycle Consistencyand , Niter_SCC =10 is recommended.
% eta: 0 < eta < 1, eta = 0.5 is recommended.
% beta, gamma, lambda: regularization parametes; beta = gamma = 5, lambda = 0.1  are recommended.
% seg: balanced parameter of the MRF segmentation. The smaller the seg, the smoother the CM. 0.025<= seg <=0.1 is recommended.


%% AGSCC

t_o = clock;
fprintf(['\n AGSCC is running...... ' '\n'])

%------------- Preprocessing: Superpixel segmentation and feature extraction---------------%

t_p1 = clock;
Compactness = 1;
for i = 1:length(combinedPrefixes)
    
    prefix = combinedPrefixes{i};
    fprintf('Processing prefix: %s\n', prefix);
    
    % if prefix == "spain-volcano_00000592"
    %     continue
    % end
    % Construct full paths to images and labels
    labelPath = fullfile(datasetFolder, 'target', [prefix '_building_damage.tif']);
    preImagePath = fullfile(datasetFolder, 'pre-event', [prefix '_pre_disaster.tif']);
    postImagePath = fullfile(datasetFolder, 'post-event', [prefix '_post_disaster.tif']);
    
    % Load images and ground truth
    % Swap pre-event and post-event so that the algorithm can work well
    image_t2 = imread(preImagePath);
    image_t1 = imread(postImagePath);
    gt = imread(labelPath);
    Ref_gt = double(gt(:,:,1)); % Assuming the ground truth is single-channel
    Ref_gt(Ref_gt < 3) = 0;
    Ref_gt(Ref_gt >= 3) = 1;
    
    if sum(sum(Ref_gt)) == 0
        continue
    end
    
    opt.Ns = 5000;
    opt.Niter_AGL = 5;
    opt.Niter_SCC = 10;
    opt.eta = 0.3; % for #6-California, opt.eta = 0.3 is better.
    opt.beta = 5;
    opt.gamma  = 5;
    opt.lambda  = 0.1;
    opt.seg = 0.05;  % for #7-Texas, opt.seg = 0.01 is better.
    image_t1 = double (image_t1);
    image_t2 = double (image_t2);

    [sup_img,Ns] =  SuperpixelSegmentation(image_t1,opt.Ns,Compactness);
    [t1_feature,t2_feature, norm_par] = Feature_extraction(sup_img,image_t1,image_t2) ;% MVE;MSM
    % fprintf('\n');fprintf('The computational time of Preprocessing (t_p1) is %i \n', etime(clock, t_p1));
    % fprintf(['\n' '====================================================================== ' '\n'])
    
    %------------- Algorithm 1: Adaptive Graph Learning---------------%
    [Sx,Kmat,wm_x,wm_x_rel] = AGSCC_AdaptiveGraphLearning(t1_feature, opt.Niter_AGL, opt.eta);
    % fprintf('\n');fprintf('The computational time of Adaptive Graph Learning (t_p2) is %i \n', etime(clock, t_p2));
    % fprintf(['\n' '====================================================================== ' '\n'])
    
    %------------- Algorithm 2: Structure Cycle Consistency based Image Regression---------------%
    % image_t1 ----> image_t2
    
    sum_wmx = sum(wm_x);
    opt.beta = sum_wmx * opt.beta;
    opt.lambda = sum_wmx * opt.lambda;
    opt.gamma  =sum_wmx * opt.gamma ; 
    opt.mu = sum_wmx * 0.4;
    [Xpie,regression_t1,delt,wm_y,RelDiff] = AGSCC_StructureCycleConsistencyRegression(t1_feature,t2_feature,Sx,Kmat,wm_x,opt.Niter_SCC, opt.beta, opt.lambda, opt.gamma, opt.eta, opt.mu);  
    % fprintf('\n');fprintf('The computational time of Image Regression (t_p3) is %i \n', etime(clock, t_p3));
    % fprintf(['\n' '====================================================================== ' '\n'])
    
    %------------- Algorithm 3: MRF segmentation---------------%
    
    delt_new (1:size(image_t2,3),:) = delt(:,:,1);
    delt_new (size(image_t2,3)+1:2*size(image_t2,3),:) = delt(:,:,2);
    delt_new (2*size(image_t2,3)+1:3*size(image_t2,3),:) = delt(:,:,3);
    
    [CM_map,labels] = MRFsegmentation(sup_img, opt.seg, delt_new);
    % fprintf('\n');fprintf('The computational time of MRF segmentation (t_p4) is %i \n', etime(clock, t_p4));
    % fprintf(['\n' '====================================================================== ' '\n'])
    
    % fprintf('\n');fprintf('The total computational time of AGSCC (t_total) is %i \n', etime(clock, t_o));
    %% Displaying results
    
    % fprintf(['\n' '====================================================================== ' '\n'])
    % fprintf(['\n Displaying the results...... ' '\n'])

    %---------------------AUC PCC F1 KC ----------------------%
    
    % Ref_gt = Ref_gt/max(Ref_gt(:));
    [tp,fp,tn,fn,fplv,fnlv,~,~,pcc,kappa,imw]=performance(CM_map,Ref_gt);
    F1 = 2*tp/(2*tp + fp + fn);
    result = 'PCC is %4.3f; F1 is %4.3f; KC is %4.3f \n';
    fprintf(result,pcc,F1,kappa)
    totalTP = totalTP + tp;
    totalFP = totalFP + fp;
    totalTN = totalTN + tn;
    totalFN = totalFN + fn;

    clear delt_new image_t1 image_t2 sup_img Ns t1_feature t2_feature norm_par Sx Kmat wm_x wm_x_rel Xpie regression_t1 delt wm_y RelDiff CM_map labels
end


% Calculate final metrics from aggregated confusion matrix
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

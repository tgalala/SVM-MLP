%% Loading Dataset
% *****************

%Clear all workspace
clear all; clc; close all;

rng('default'); % For reproducibility

% Load training set where smote was applied to balance the classes
data = readtable('../data/trainSmote.csv', 'ReadVariableNames',true);

xTrain = data(:,2:17);
yTrain = data(:,18);

%% Basic SVM Model Training
% **************************

% Training Basic SVM
tic %Time the model % 
svmBASIC = fitcsvm(xTrain, yTrain,'Standardize',true);
timeSVM = toc; %store the time

% SVM prediction
yPredSVM = predict(svmBASIC, xTrain);

cm2 = confusionmat(table2array(yTrain),yPredSVM);
SVM_TP = cm2(1,1); % For percentage cm(1,1)/sum(cm(:,1));
SVM_TN = cm2(2,2); % For percentage cm(2,2)/sum(cm(:,2));
SVM_FN = cm2(1,2); % For percentage cm(1,2)/sum(cm(:,2));
SVM_FP = cm2(2,1); % For percentage cm(2,1)/sum(cm(:,1)); 

SVM_Accuracy = ((SVM_TP + SVM_TN)/(SVM_TP + SVM_TN + SVM_FP + SVM_FN))*100;

% Precision, Recall, Fscore, AUC
SVM_Precision = cm2(2,2) / (cm2(2,2) + cm2(1,2));
SVM_Recall = cm2(2,2) / (cm2(2,2) + cm2(2,1));
SVM_Fscore = 2*SVM_TP /(2*SVM_TP + SVM_FP + SVM_FN);

[label, score_svm] = predict(svmBASIC, xTrain); % optimized
[xSVM, ySVM, tSVM, aucSVM] =perfcurve(table2array(yTrain),score_svm(:,2),1); % score vector for positive '1' outcome

%% Display basic SVM performance
% *******************************

fprintf('\n****************************************************************************')
fprintf('\n                            Basic SVM Scores                               ')
fprintf('\n****************************************************************************')
finalscore = {'Basic Support Vector Machines'};
Accuracy = [SVM_Accuracy];
Fscore = [SVM_Fscore];
AUC = [aucSVM];
Time = [timeSVM];
T = table(Accuracy, Fscore, AUC, Time, 'RowNames',finalscore);
head(T)
 
%% Hyperparameters lookup
%************************
  
% VariableDescriptions = hyperparameters('fitcsvm',xTrain, yTrain);
% % Examine all the hyperparameters.
% 
% for ii = 1:length(VariableDescriptions)
%      disp(ii),disp(VariableDescriptions(ii))
% end

%% SVM Model Training using Grid Search
% *************************************

%Creating an an array to capture the results
results_kernel = zeros(1,10);
results_pol = zeros(1,20);
results_scale= zeros(1,5);
results_learnRate = zeros(1,5);
results_box = zeros(1,5);
Scores_Grid = [];
result_AUC =[];

% Choice of Hyper Parameters 

% Note: 
% 1- Choose the following paramters
% combinations that works with the corresponding kernel
% 2- Polynomial training takes a long time)
% 3- Polynomial training remove scale & RBF training remove pol 

kernel = "polynomial"; % "RBF", "polynomial", "linear"; 
pol = [2 3 4]; %  Polynomial Order
scale = [0.1 0.5 1]; % Kernel scale, can be set to auto
box = [0.5 0.75 0.8 5 10 50 100]; % Box Constraint, large box flexible model. small = rigid model, less sensitive to overfitting

% Grid Search
% (Note: Choose the following paramters
% combinations in grid search that works with the coresponsing kernel)

for i=1:length(kernel)

    for n=1:length(pol)

        for m=1:length(scale)

            for k=1:length(box)
                    
                    % SVM training parameters 
                    Mdl_Optimized = fitcsvm(xTrain,yTrain,'Standardize', true,...
                                           'KernelFunction', kernel(i),...
                                           'PolynomialOrder' ,pol(n),... 
                                           'KernelScale', scale(m),...
                                           'BoxConstraint',box(k));                        

                    tic % time each cross validated model
                    %Cross-validating the model & active
                    %parrallel computing
                    
                    cvMdl_Optimized = crossval(Mdl_Optimized, 'KFold',2); % Return to 10 folds
                    toc

                    Grid_time = toc; % store times

                    % Predicting model with kfoldPredict
                    [y_pred ,score_svm]  = kfoldPredict(cvMdl_Optimized);

                    % Calculating Loss on
                    % training set with kfold
                    loss = kfoldLoss(cvMdl_Optimized);
                    
                    cm2 = confusionmat(table2array(yTrain),y_pred);
                    SVM_TP = cm2(1,1); % For percentage cm(1,1)/sum(cm(:,1));
                    SVM_TN = cm2(2,2); % For percentage cm(2,2)/sum(cm(:,2));
                    SVM_FN = cm2(1,2); % For percentage cm(1,2)/sum(cm(:,2));
                    SVM_FP = cm2(2,1); % For percentage cm(2,1)/sum(cm(:,1)); 

                    SVM_Accuracy = ((SVM_TP + SVM_TN)/(SVM_TP + SVM_TN + SVM_FP + SVM_FN))*100;
                    
                    % Precision, Recall, Fscore, AUC
                    SVM_Precision = cm2(2,2) / (cm2(2,2) + cm2(1,2));
                    SVM_Recall = cm2(2,2) / (cm2(2,2) + cm2(2,1));
                    SVM_Fscore = 2*SVM_TP /(2*SVM_TP + SVM_FP + SVM_FN);
                    % [xSVM, ySVM, tSVM, aucSVM] = perfcurve(yTest,yPredSVM, 1);
                    [xSVM, ySVM, tSVM, aucSVM] =perfcurve(table2array(yTrain),score_svm(:,2),1); % score vector for positive '1' outcome;

                    % AUC score results
                    result_AUC=[result_AUC, aucSVM];

                    % Display performance
                    % metrics for each model
                    fprintf('\n****************************************************************************************************')
                    fprintf('\n                            SVM Hyper-parameters Grid Search                                       ')
                    fprintf('\n****************************************************************************************************\n')
                    fprintf('     Kernel    Polynomial   Scale    Box      Accuracy     F-score      AUC        Time\n')
                    fprintf('****************************************************************************************************\n')
                    Scores_Grid = [Scores_Grid; kernel(i),  pol(n), scale(m), box(k), SVM_Accuracy, SVM_Fscore, aucSVM, Grid_time]
                    
            end

        end

   end

end
    

% Return highest F-score in matrix and the placement
fprintf('\n***************************************************************************************************')
fprintf('\n                                SVM Best Hyper-parameters                                          ')
fprintf('\n***************************************************************************************************\n')
fprintf('  Kernel     Polynomial   Scale    Box      Accuracy     F-score      AUC        Time\n')
fprintf('***************************************************************************************************')

% Printing Max Fscore Performance Metrics
[maxValue, index] = max(result_AUC);
Best = Scores_Grid(index,:) 

plot(xSVM,ySVM)
legend('SVM AUC = 0.93','Location','best')
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curve')

figure(4);
plotconfusion(categorical(table2array(yTrain)),categorical(y_pred), "SVM")

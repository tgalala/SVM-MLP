%% Loading Dataset
% ****************

%Clear all workspace
clear all; clc; close all;

rng('default'); % For reproducibility

% Load training set where smote was applied to balance the classes
data = readtable('../data/trainSmote.csv', 'ReadVariableNames',true);
data = table2array(data);

xTrain = data(:,2:17);
yTrain = data(:,18);

%% SVM Bayesian Optimization
% **************************

c = cvpartition(5598, 'KFold',3);
opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
    'AcquisitionFunctionName','expected-improvement-plus', 'MaxObjectiveEvaluations', 100);
svmmod = fitcsvm(xTrain,yTrain,'KernelFunction','rbf',...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts)

%% Best Hyperparameters
% *********************

disp(svmmod.HyperparameterOptimizationResults);
bestParam= svmmod.HyperparameterOptimizationResults.XAtMinObjective;
loss = svmmod.HyperparameterOptimizationResults.MinObjective;
bestBox = table2array(bestParam(1,1));
bestScale = table2array(bestParam(1,2));

%% Final Model Training with best hyperparameters
% ***********************************************

% Array to capture the results
Scores_SVM_Bayes = [];

tic

% SVM training parameters 
ModelBayesSVM = fitcsvm(xTrain,yTrain,'Standardize', true,...
                       'KernelScale', bestScale,...
                       'BoxConstraint', bestBox);                        

Bayes_time = toc;

[label, score_svm] = predict(ModelBayesSVM, xTrain);

% Metrics and confusion to compare

cm = confusionmat(yTrain,label);
Bayes_TP = cm(1,1); % For percentage cm(1,1)/sum(cm(:,1));
Bayes_TN = cm(2,2); % For percentage cm(2,2)/sum(cm(:,2));
Bayes_FN = cm(1,2); % For percentage cm(1,2)/sum(cm(:,2));
Bayes_FP = cm(2,1); % For percentage cm(2,1)/sum(cm(:,1)); 

Sens = cm(1,1)/sum(cm(1,:));
Spec = cm(2,2)/sum(cm(2,:));
Bayes_Accuracy = ((Bayes_TP + Bayes_TN)/(Bayes_TP + Bayes_TN + Bayes_FP + Bayes_FN))*100;

% Precision, Recall, Fscore, AUC
Bayes_Precision = cm(2,2) / (cm(2,2) + cm(1,2));
Bayes_Recall = cm(2,2) / (cm(2,2) + cm(2,1)); % Sensitivity
Bayes_Fscore = 2 * Bayes_Precision * Bayes_Recall / (Bayes_Precision + Bayes_Recall);
%[xBayes, yBayes, TBayes, aucBayes] = perfcurve(yTrain, yPredSVM, 1);
[xBayes, yBayes, TBayes, aucBayes] = perfcurve(yTrain,score_svm(:,2),1); % score vector for positive '1' outcome;


%% Display Bayes SVM performance
% ******************************

fprintf('\n*******************************************************************************')
fprintf('\n                                Bayes SVM Scores                               ')
fprintf('\n*******************************************************************************')
finalscore = {'Bayes Support Vector Machines'};
Accuracy = [Bayes_Accuracy];
Fscore = [Bayes_Fscore];
AUC = [aucBayes];
Time = [Bayes_time];
T = table(Accuracy, Fscore, AUC, Time, 'RowNames',finalscore);
head(T)

%ROC curve MLP with Bayes Optimization
figure();
plot(xBayes,yBayes);
legend('SVM AUC = 0.92','Location','best');
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC curve SVM with Bayes Optimization')


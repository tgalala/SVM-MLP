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

%% MLP Bayesian Optimization
% ***************************

% Array to capture the results
Scores_MLP_Bayes = [];

% Train/Vaidation split 20%
cv = cvpartition(numel(yTrain), 'Holdout', 0.2); 

% Selected Hyperparameters to be optimized
vars = [optimizableVariable('hiddenLayerSize', [5 40], 'Type', 'integer');
        optimizableVariable('lr', [1e-3 1], 'Transform', 'log')
        optimizableVariable('mc', [0.2 0.8]);];

% Optimize using kfoldloss function with 10 folds
fun = @(best)kFoldLoss(xTrain', yTrain', cv, best.hiddenLayerSize, best.lr, best.mc);


results = bayesopt(fun, vars,'IsObjectiveDeterministic', true,...
                    'AcquisitionFunctionName', 'expected-improvement-plus',... 
                    'MaxObjectiveEvaluations', 100, 'UseParallel',true);

%% Best Hyperparameters
% *********************

[best,CriterionValue,iteration] = bestPoint(results,'Criterion','min-observed');

%% Final Model Training with best hyperparameters
% ***********************************************

tic;
Bayes_net = patternnet(best.hiddenLayerSize, 'trainrp');
Bayes_net.trainParam.lr = best.lr;
Bayes_net.trainParam.mc = best.mc;
[Bayes_net tr y e] = train(Bayes_net, xTrain', yTrain', 'useParallel','yes');
toc;
Bayes_time = toc;

outputs = Bayes_net(xTrain');
errors = gsubtract(yTrain',outputs);
performance1 = perform(Bayes_net,yTrain',outputs);
perf1 = crossentropy(Bayes_net,yTrain',outputs);

% Cross Entropy for Training, validation and test sets
tr = tr;
Training_cross_entropy_best = tr.best_perf;
Validaton_cross_entropy_best = tr.best_vperf;
Test_cross_entropy_best = tr.best_tperf;

% Metrics and confusion to compare
[c,cm,ind,per] = confusion(yTrain',outputs);
Bayes_TP = cm(1,1)/sum(cm(:,1)); % Division is  by elements in the predicted true column
Bayes_TN = cm(2,2)/sum(cm(:,2)); % Same issues:
Bayes_FN = cm(1,2)/sum(cm(:,2));
Bayes_FP = cm(2,1)/sum(cm(:,1));
Sens = cm(1,1)/sum(cm(1,:));
Spec = cm(2,2)/sum(cm(2,:));
Bayes_Accuracy = ((Bayes_TP + Bayes_TN)/(Bayes_TP + Bayes_TN + Bayes_FP + Bayes_FN))*100;

% Precision, Recall, Fscore, AUC
Bayes_Precision = cm(2,2) / (cm(2,2) + cm(1,2));
Bayes_Recall = cm(2,2) / (cm(2,2) + cm(2,1)); % Sensitivity
Bayes_Fscore = 2 * Bayes_Precision * Bayes_Recall / (Bayes_Precision + Bayes_Recall);
[xBayes, yBayes, TBayes, aucBayes] = perfcurve(yTrain', outputs, 1);

%% Display Bayes MLP performance
% ******************************

fprintf('\n****************************************************************************')
fprintf('\n                            Bayes MLP Scores                               ')
fprintf('\n****************************************************************************')
finalscore = {'Bayes Multilayer Perceptron'};
Accuracy = [Bayes_Accuracy];
Fscore = [Bayes_Fscore];
AUC = [aucBayes];
Time = [Bayes_time];
T = table(Accuracy, Fscore, AUC, Time, 'RowNames',finalscore);
head(T)

% fprintf('\n**************************\n')
% fprintf('\n   MLP Bayes Best Score   \n')
% fprintf('\n**************************\n\n')
% fprintf('   Accuracy   Fscore   AUC  \n')
% fprintf('****************************\n')
% Scores_MLP_Bayes = [Scores_MLP_Bayes; Bayes_Accuracy,Bayes_Fscore, aucBayes]

%Confusion matrix that matches the tool matrices whole
plotconfusion(yTrain',outputs);

%ROC curve MLP with Bayes Optimization
figure();
plot(xBayes,yBayes);
legend('MLP AUC = 0.94','Location','best');
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC curve MLP with Bayes Optimization')


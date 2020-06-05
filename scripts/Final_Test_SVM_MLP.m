%**************************************************************************
%                   MLP VS SVM - Model Testing 
%**************************************************************************

%% Loading Dataset
% *****************

%Clear all workspace
clear all; clc; close all;

rng('default'); % For reproducibility

% Load training set where smote was applied to balance the classes
data = readtable('../data/trainSmote.csv', 'ReadVariableNames',true);
data = table2array(data);
xTrain = data(:,2:17);
yTrain = data(:,18);

test = readtable('../data/test.csv', 'ReadVariableNames',true);
test = table2array(test);
xTest = test(:,2:17);
yTest = test(:,18);

%create arrays to store all values
AUC =[];
Fscore = [];
Time = [];
Precision =[];
Recall = [];
score = [];

%%  MODEL 1 Support Vector Machines 
%**************************************************************************
%% Train final model on full training set using the best hyperparameters
%**************************************************************************

% SVM Optimal hyperparameters
kernel =  "polynomial"; % "gaussian", "linear", "RBF";
pol = 2; %2;
scale =  0.65 ;%'auto';
box = 0.8;

% Parallel computing
options = statset('UseParallel',true);

tic %Time the model % 
svm_optimal = fitcsvm(xTrain,yTrain, 'Standardize', true,...
                                     'KernelFunction',kernel,...
                                     'PolynomialOrder' ,pol,...
                                     'BoxConstraint',box);
timeSVM = toc; %store the time

%% Saving our SVM model to a mat file
%*************************************

SVMModelFinal = svm_optimal;
save SVMModelFinal;

%% SVM Evaluation on Test Set
%****************************
 
% Load the saved model
load SVMModelFinal;

% SVM prediction on test set
yPredSVM = predict(SVMModelFinal, xTest);

% SVM confusion chart (test set)
figure(2);
plotconfusion(categorical(yTest),categorical(yPredSVM), "SVM")

cm2 = confusionmat(yTest,yPredSVM);
SVM_TP = cm2(1,1); % For percentage cm(1,1)/sum(cm(:,1));
SVM_TN = cm2(2,2); % For percentage cm(2,2)/sum(cm(:,2));
SVM_FN = cm2(1,2); % For percentage cm(1,2)/sum(cm(:,2));
SVM_FP = cm2(2,1); % For percentage cm(2,1)/sum(cm(:,1)); 

SVM_Accuracy = ((SVM_TP + SVM_TN)/(SVM_TP + SVM_TN + SVM_FP + SVM_FN))*100;
% Precision, Recall, Fscore, AUC
SVM_Precision = cm2(2,2) / (cm2(2,2) + cm2(1,2));
SVM_Recall = cm2(2,2) / (cm2(2,2) + cm2(2,1));
SVM_Fscore = 2*SVM_TP /(2*SVM_TP + SVM_FP + SVM_FN);
%[xSVM, ySVM, tSVM, aucSVM] = perfcurve(yTest,yPredSVM, 1);
[label, score_svm] = predict(svm_optimal, xTest); % optimized
[xSVM, ySVM, tSVM, aucSVM] =perfcurve(yTest,score_svm(:,2),1); % score vector for positive '1' outcome

%% MODEL 2: MLP
%**************************************************************************
%% Train final model on full training set using the best hyperparameters
%**************************************************************************

% MLP Hyper-parameters Obtained from Grid Search
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 85/100;
net.divideParam.valRatio = 15/100;
net = patternnet(35, 'trainrp'); 
net.trainParam.lr = 0.005;
net.trainParam.mc = 0.35;
% net.layers{1}.transferFcn = 'transig';
net.trainParam.epochs=200;

tic %time the model
[net tr y e] = train(net, xTrain', yTrain'); %, 'useParallel','yes');
timeMLP = toc; %store the time

%% Saving our MLP model into a mat file
%**************************************

MLPModelFinal = net;
save MLPModelFinal;

%% MLP Evaluation on Test Set
%****************************

% Load the saved model
load MLPModelFinal;

ypredMLP = MLPModelFinal(xTest');

% MLP confusion chart (test set)
figure(3);
plotconfusion(yTest',ypredMLP, "MLP")

[c,cm,ind,per] = confusion(yTest',ypredMLP);
MLP_TP = cm(1,1); % For percentage cm(1,1)/sum(cm(:,1));
MLP_TN = cm(2,2); % For percentage cm(2,2)/sum(cm(:,2));
MLP_FN = cm(1,2); % For percentage cm(1,2)/sum(cm(:,2));
MLP_FP = cm(2,1); % For percentage cm(2,1)/sum(cm(:,1)); 

MLP_Accuracy = ((MLP_TP + MLP_TN)/(MLP_TP + MLP_TN + MLP_FP + MLP_FN))*100;
% Precision, Recall, Fscore, AUC
MLP_Precision = cm(2,2) / (cm(2,2) + cm(1,2));
MLP_Recall = cm(2,2) / (cm(2,2) + cm(2,1));
MLP_Fscore = 2*MLP_TP /(2*MLP_TP + MLP_FP + MLP_FN);
[xMLP, yMLP, TMLP, aucMLP] = perfcurve(yTest',ypredMLP, 1);

%% Display performance
%*********************

fprintf('\n*******************************************************************************')
fprintf('\n                            SVM VS MLP Scores on Test Set                               ')
fprintf('\n*******************************************************************************')
finalscore = {'Support Vector Machines Scores'; 'Multilayer Perceptron Scores'};
Accuracy = [SVM_Accuracy; MLP_Accuracy];
Fscore = [SVM_Fscore; MLP_Fscore];
AUC = [aucSVM; aucMLP];
Time = [timeSVM; timeMLP];
T = table(Accuracy, Fscore, AUC, Time, 'RowNames',finalscore);
head(T)
%% Plot ROC, Error & AUC Bar  
%***************************

%% Plotting ROC Curves for the two models together for comparision
%*****************************************************************

figure(4)
plot(xSVM,ySVM)
hold on
plot(xMLP,yMLP)
legend('SVM AUC = 0.87','MLP AUC = 0.89','Location','best')
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves Comparison')
hold off

%% Error plots
%*************

% plt=0;
% plt = plt+1;, figure(plt), hold on;
% plot(log(tr.perf),'b', 'LineWidth', 3)
% plot(log(tr.vperf),'r', 'LineWidth', 3)
% %plot(log(tr.tperf),'r', 'LineWidth', 2)
% xlabel('Epochs'); ylabel('Cross Entropy');
% title('MLP Train & Validation Learning Curves');
% legend('Train','Validation','Location','best');

%% AUC Bar Chart Comparison
%**************************

names = {'SVM','MLP'};
AUC(1) = aucSVM;
AUC(2) = aucMLP;
figure
bar(AUC);
title('Area Under the Curve');
xlabel('Model');
ylabel('AUC');
xticklabels(names);
ylim([0,1]);

s = rng;
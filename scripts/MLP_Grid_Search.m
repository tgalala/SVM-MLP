%% Loading Dataset
%*****************

%Clear all workspace
clear all; clc; close all;

% Load training set where smote was applied to balance the classes
data = readtable('../data/trainSmote.csv', 'ReadVariableNames',true);
data = table2array(data);

xTrain = data(:,2:17);
yTrain = data(:,18);
%% Basic MLP Model Training
%**************************

% Array to capture the results
Scores_MLP_Basic = [];

tic;
net = patternnet();
[net_basic tr y e] = train(net, xTrain', yTrain', 'useParallel','yes');
toc;
Basic_time = toc; % store times

outputs = net_basic(xTrain');

[c,cm,ind,per] = confusion(yTrain',outputs);
Basic_TP = cm(1,1)/sum(cm(:,1)); Basic_TP = round(Basic_TP,3); % Division is  by elements in the predicted true column
Basic_TN = cm(2,2)/sum(cm(:,2)); Basic_TN = round(Basic_TN,3);% Same issues:
Basic_FN = cm(1,2)/sum(cm(:,2)); Basic_FN = round(Basic_FN,3);
Basic_FP = cm(2,1)/sum(cm(:,1)); Basic_FP = round(Basic_FP,3);
Sens = cm(1,1)/sum(cm(1,:));
Spec = cm(2,2)/sum(cm(2,:));

Basic_Accuracy = ((Basic_TP + Basic_TN)/(Basic_TP + Basic_TN + Basic_FP + Basic_FN))*100;

% Precision, Recall, Fscore, AUC
Basic_Precision = cm(2,2) / (cm(2,2) + cm(1,2));
Basic_Recall = cm(2,2) / (cm(2,2) + cm(2,1)); % Sensitivity
Basic_Fscore = 2 * Basic_Precision * Basic_Recall / (Basic_Precision + Basic_Recall);
[xBasic, yBasic, tBasic, aucBasic] = perfcurve(yTrain', outputs, 1);

% Display basic MLP performance

fprintf('\n****************************************************************************')
fprintf('\n                            Basic MLP Scores                               ')
fprintf('\n****************************************************************************')
finalscore = {'Basic Multilayer Perceptron'};
Accuracy = [Basic_Accuracy];
Fscore = [Basic_Fscore];
AUC = [aucBasic];
Time = [Basic_time];
T = table(Accuracy, Fscore, AUC, Time, 'RowNames',finalscore);
head(T)

%% MLP Model Training using Grid Search
%**************************************

rng('default'); % For reproducibility

% Creating an an array to capture the results
results_HiddenLayerSize = zeros(1,50);
results_learn = zeros(1,50);
results_memento = zeros(1,50);
results_regularization = zeros(1,50);
Scores_Grid = [];
result_AUC =[];

% Choice of Hyper Parameters
func = ["trainrp","trainbfg", "trainscg", "traincgp", "traincgb"]; % training function 
HiddenLayerSize = [25 35 45]; % Number of Nodes in a hidden layer
learn = [0.015 0.005 0.15 0.5 0.75 0.90]; %  Learning rate
momentum = [0.25 0.35 0.45 0.55 0.75]; % momentum
regularization = [0 0.15 0.25]; % Cross-Entropy regularization
% function = ["logsig", "tansig"]; % activation function to be changed for hidden layer

% 10 kfold 
kfold=10;
groups=[0 1];
cvFolds = crossvalind('Kfold', groups, kfold);

% Grid Search

for g=1:length(func)
    
    for i=1:length(HiddenLayerSize)
           
        for k=1:length(learn)
               
            for m=1:length(momentum)
            
                for n=1:length(regularization)
                                        
                    for z = 1:kfold                                
                      
                        testIdx = (cvFolds == z);               % get indices of test instances
                        trainIdx = ~testIdx  ;                  % get indices training instances
                        trInd=find(trainIdx);
                        tstInd=find(testIdx);
                        net.divideFcn = 'divideind'; 
                        net.divideParam.trainInd=trInd;
                        net.divideParam.testInd=tstInd;

                        %Grid Search to find optimal hyper paramters
                        net = patternnet(HiddenLayerSize(i), func(g));
                        net.trainParam.lr = learn(k);
                        net.trainParam.mc = momentum(m);
                        net.performParam.regularization = regularization(n);
                        tic;
                        [Grid_net tr y e] = train(net, xTrain', yTrain', 'useParallel','yes');
                        toc;
                        Grid_time = toc; % store times
                        
                        outputs = Grid_net(xTrain');
                        errors = gsubtract(yTrain',outputs);
                        performance = perform(Grid_net,yTrain',outputs);
                        trainTargets = yTrain' .* tr.trainMask{1};
                        testTargets = yTrain'  .* tr.testMask{1};
                        trainPerformance = perform(Grid_net,trainTargets,outputs);
                        testPerformance = perform(Grid_net,testTargets,outputs);
                        test(kfold)=testPerformance;
                        
                        [c,cm,ind,per] = confusion(yTrain',outputs);
                        Grid_TP = cm(1,1); % For percentage cm(1,1)/sum(cm(:,1));
                        Grid_TN = cm(2,2); % For percentage cm(2,2)/sum(cm(:,2));
                        Grid_FN = cm(1,2); % For percentage cm(1,2)/sum(cm(:,2));
                        Grid_FP = cm(2,1); % For percentage cm(2,1)/sum(cm(:,1)); 
                        Sens = cm(1,1)/sum(cm(1,:));
                        Spec = cm(2,2)/sum(cm(2,:));
                        
                        Grid_Accuracy = ((Grid_TP + Grid_TN)/(Grid_TP + Grid_TN + Grid_FP + Grid_FN))*100;
                        % Precision, Recall, Fscore, AUC
                        Grid_Precision = cm(2,2) / (cm(2,2) + cm(1,2));
                        Grid_Recall = cm(2,2) / (cm(2,2) + cm(2,1)); % Sensitivity
                        %Grid_Fscore = 2 * Grid_Precision * Grid_Recall / (Grid_Precision + Grid_Recall);
                        Grid_Fscore = 2*Grid_TP /(2*Grid_TP + Grid_FP + Grid_FN);
                        [xGrid, yGrid, TGrid, aucGrid] = perfcurve(yTrain', outputs, 1);

                        % AUC score results
                        result_AUC=[result_AUC, aucGrid];

                        %Display performance
                        %metrics for each model
                        fprintf('\n****************************************************************************************************')
                        fprintf('\n                            MLP Hyper-parameters Grid Search                                       ')
                        fprintf('\n****************************************************************************************************\n')
                        fprintf('     Function    Hidden   learn    momentum   Reg      Accuracy     F-score        AUC          Time\n')
                        fprintf('****************************************************************************************************\n')
                        Scores_Grid = [Scores_Grid; func(g), HiddenLayerSize(i), learn(k) ,momentum(m), regularization(n),Grid_Accuracy,Grid_Fscore, aucGrid, Grid_time]
                
                     end
                 
                 end
                
            end
            
        end
        
    end
    
end 

% Return highest F-score in matrix and the placement
fprintf('\n***************************************************************************************************')
fprintf('\n                                MLP Best Hyper-parameters                                          ')
fprintf('\n***************************************************************************************************\n')
fprintf('   Function     Hidden    learn    memento     Reg     Accuracy     Fscore        AUC        Time  \n')
fprintf('***************************************************************************************************')

% Printing Max Fscore Performance Metrics
[maxValue, index] = max(result_AUC);
Best = Scores_Grid(index,:)
s=rng;

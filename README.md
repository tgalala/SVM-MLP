
# A Comparative Study of Support Vector Machines and Multilayer Perceptron for Predicting Direct Marketing Response in Banking

Language
MATLAB

Introduction
This repository presents a comparative study between two algorithms to predict whether customers will sign to a long-term deposit with the bank with Support Vector Machine (SVM) and Multilayer Perception Neural Network (MLP). A grid search was conducted for both algorithms to optimize the hyperparameters. The best models were evaluated against a test set and critically evaluated by their F-Measure and ROC Area values (AUC).


Dataset
The data is related with direct marketing campaigns of a Portuguese banking institution retrieved from the University of California Irvine (UCI) Machine Learning Repository. We choose the dataset containing (4521) instances with (17) attributes without missing values. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.


Scripts Folder

1. Final_Test_SVM_MLP <br>
This is the main file to run which contains both models. Full training on train set then prediction
on the unseen test set. Here most figures and charts are produced.

2. MLP_Grid_Search <br>
MLP grid search on training set.

3. MLP_Bayes_Optimization_Kfold <br>
MLP Bayes optinmzation on training set.

4. kFoldLoss <br>
This file in needed for MLP_Bayes_Optimization_Kfold to run

5. MLPModelFinal.mat <br>
MLP best model in mat format saved.

6. SVM_Grid_Search <br>
SVM grid search on training set.

7. SVM_Bayes_Optimization_kfold <br>
SVM Bayes optinmzation on training set.

8. SVMModelFinal.mat <br>
MLP best model in mat format saved.


Data Folder:

1. test <br>
test dataset

2. trainSmote <br>
Training set after SMOTE is applied.

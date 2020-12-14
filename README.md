# Project: Machine Learning Engineer with Microsoft Azure Capstone

# Table of Contents
<!--ts-->
- [Dataset](#dataset)
  * [Overview](#overview)
  * [Task](#task)
  * [Access](#access)
- [Architectural Diagram](#architectural-diagram)
- [Automated ML](#automated-ml)
  * [Overview of AutoML Settings](#overview-of-automl-settings)
  * [Results](#results)
  * [RunDetails Widget](#rundetails-widget)
  * [Best Model](#best-model)
- [Hyperparameter Tuning](#hyperparameter-tuning)
  * [Overview of Hyperparameter Tuning Settings](#overview-of-hyperparameter-tuning-settings)
  * [Results](#results)
  * [RunDetails Widget](#rundetails-widget)
  * [Best Model](#best-model)
- [Model Deployment](#model-deployment)
  * [Overview of Deployed Model](#overview-of-deployed-model)
  * [Endpoint Query](#endpoint-query)  
- [Screen Recording](#screen-recording)
- [Suggestions to Improve](#suggestions-to-improve)
 
<!--te-->  

## Dataset
### Overview
### Task
### Access

## Architectural Diagram

The architechtural diagram is illustrated in the chart as below:

![architechtural diagram](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/Architectural-Diagram.png?raw=true)

For the real-time endpoint: 
With the registered bank marketing dataset, a classification automated machine learning training was performed with a criteria of "accuracy". 
After training, the best model was generated as of "VotingEnsemble" and was deployed as a real-time endpoint. 
The prediction was made by running the provided "endpoint.py" file. 

For the pipeline endpoint: 
The same registered bank marketing dataset was used. The automated machine learning training was performed with the use of a Jupyter Notebook. The primary metric used was "AUC_weighted". 
Afterwards, the pipeline with AutoMLStep was created. During the training, the AutoMLStep could be visualized with the use of "RunDetails" widget. 
The best model was retrieved and tested. 
The pipeline was publishd and the REST endpoint was generated to use for predictions. 

## Automated ML

### Overview of AutoML Settings 

### Results


### RunDetails Widget
![RunDetails Widget-1](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/RunDetails-Widget-1_new.PNG?raw=true)
![RunDetails Widget-2](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/RunDetails-Widget-2_new.PNG?raw=true)

### Best Model
![best model 1](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/best%20model%20-%201_new.PNG?raw=true)
![best model 2](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/best%20model%20-%202_new.PNG?raw=true) model


## Hyperparameter Tuning

### Overview of Hyperparameter Tuning Settings 

### Results


### RunDetails Widget
![RunDetails Widget-1](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/RunDetails-Widget-1_new.PNG?raw=true)
![RunDetails Widget-2](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/RunDetails-Widget-2_new.PNG?raw=true)

### Best Model
![best model 1](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/best%20model%20-%201_new.PNG?raw=true)
![best model 2](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/best%20model%20-%202_new.PNG?raw=true) model

## Model Deployment

### Overview of Deployed Model 

### Endpoint Query

## Screen Recording

A [screen recording](https://youtu.be/f7VzVPqbxpY) of the project is provided to demonstrate the following steps: 

* a working model
* demo of the deployed model
* demo of a sample request sent to the endpont and its response 

## Suggestions to Improve

* To perform feature engineering, for example, dimension reduction using PCA. PCA enables to represent a multivariate data (i.e., high dimension) as smaller set of variables (i.e., small dimenstion) in order to observe trends, clusters, and outliers. This could uncover the relationships between observations and variables and among the variables.

* To fix data imbalance. The dataset is highly imbalanced and about 2473 / (2473 + 22076) * 100 = 10.1 % of the clients actually subscribed to a term deposit. Imbalanced data can lead to a falsely perceived positive effect of a model's accuracy because the input data has bias towards one class. Therefore, an data imbalance issue should be considered to fix for future experiments.We may try: 1) random undersampling, 2) oversampling with SMOTE, 3) a combination of under- and oversampling method using pipeline.

* To try to use other metrics such as 'AUC_weighted' to get better automated ML training . As for a highly unbalanced problem like this, AUC metric is very popular. AUC is acturally preferred over accuracy for binary classification therefore this metric is worth a try.


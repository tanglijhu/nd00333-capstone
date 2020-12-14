# Project Title: Operationalizing Machine Learning Pipeline and Deployed Endpoint

The project's goal is to operationalize a machine learning pipeline with a deployed REST endpoint. 

The dataset used contains the bank marketing information including age, job, marital status, education, housing, loan, poutcome, etc. It is a classification problem where the goal is to predict whether the client will subscribe to a term deposit. After Azure Studio automated machine learning training, the best model was generated based on the criteria of "accuracy". The best model was deployed as the endpoint, which serves out predictions about whether a client's tendency to subscribe to a term deposit. 

# Table of Contents
<!--ts-->
- [Architectural Diagram](#architectural-diagram)
- [Screenshots](#screenshots)
  * [registered dataset](#registered-dataset)
  * [completed automated machine learning ](#completed-automated-machine-learning )
  * [best model](#best-model)
  * [Application Insights enabled](#Application-Insights-enabled)
  * [running logs file](#running-logs-file)
  * [running swagger on localhost showing the HTTP API methods and responses from the model](#running-swagger-on-localhost-showing-the-HTTP-API-methods-and-responses-from-the-model)
  * [running endpoint file against the API producing JSON output from the model](#running-endpoint-file-against-the-API-producing-JSON-output-from-the-model)
  * [active REST pipeline endpoint](#active-REST-pipeline-endpoint)
  * [use RunDetails widget in Jupyter Notebook](#use-rundetails-widget-in-jupyter-notebook)
  * [scheduled run](#scheduled-run)
- [Key Steps](#key-steps)
- [Screen Recording](#screen-recording)
- [Suggestions to Improve](#suggestions-to-improve)
 
<!--te-->  

    
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

## Screenshots

### registered dataset
![registered dataset](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/registered%20datasets_new.PNG?raw=true)

### completed automated machine learning 
![completed automated machine learning ](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/auto-ml-completed_new.PNG?raw=true)

### best model
![best model 1](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/best%20model%20-%201_new.PNG?raw=true)
![best model 2](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/best%20model%20-%202_new.PNG?raw=true)

### Application Insights enabled 
![Application Insights enabled](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/endpoint-after-running-log-file_new.PNG?raw=true)

### running logs file
![running "logs.py"](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/running-logs-file_new.PNG?raw=true)

### running swagger on localhost showing the HTTP API methods and responses from the model
![running swagger 1](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/swagger-running-get_new.PNG?raw=true)
![running swagger 2](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/swagger-running-get-1_new.PNG?raw=true)
![running swagger 3](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/swagger-running-post_new.PNG?raw=true)
![running swagger 4](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/swagger-running-post-1_new.PNG?raw=true)

### running endpoint file against the API producing JSON output from the model
![running "endpoint.py"](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/running-endpoint_new.PNG?raw=true)

### created pipeline
![created pipeline](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/pipeline-created_new.PNG?raw=true)

### active REST pipeline endpoint
![pipeline endpoint](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/active-rest-pipeline-endpoint_new.PNG?raw=true)

### use RunDetails widget in Jupyter Notebook
![RunDetails Widget-1](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/RunDetails-Widget-1_new.PNG?raw=true)
![RunDetails Widget-2](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/RunDetails-Widget-2_new.PNG?raw=true)

### scheduled run
![scheduled run](https://github.com/tanglijhu/nd00333_AZMLND_operationalizing_ML_project/blob/main/img/scheduled-run-pipeline-rest-endpoint_new.PNG?raw=true)


## Key-Steps

* Step 1: bank marketing dataset registration

* Step 2: Azure automated machine training as a classification problem with a creteria of "accuracy"

* Step 3: best model generation as of "VotingEnsemble" (accuracy as ~ 0.92) and registration

* Step 4: an real-time endpoint deployment from the best model:

   + running "logs.py" to enable the Application Insights" 
   + running swagger on localhost showing the HTTP API methods (both of GET and POST) and responses from the model

* Step 5: real time predictions with the provided "endpoint.py" file

* Step 6: creating, publishing, and consuming a REST pipeline endpoint with schduled predictions using the provided Jupyter Notebook "aml-pipelines-with-automated-machine-learning-step.ipynb"

## Screen-Recording

A [screen recording](https://youtu.be/f7VzVPqbxpY) of the project is provided to demonstrate the following steps: 

* a working deployed ML model endpoint
* the deployed endpoint
* the best AutoML model
* successful API requests to the endpoint with a JSON payload 

## Suggestions-to-Improve

* To perform feature engineering, for example, dimension reduction using PCA. PCA enables to represent a multivariate data (i.e., high dimension) as smaller set of variables (i.e., small dimenstion) in order to observe trends, clusters, and outliers. This could uncover the relationships between observations and variables and among the variables.

* To fix data imbalance. The dataset is highly imbalanced and about 2473 / (2473 + 22076) * 100 = 10.1 % of the clients actually subscribed to a term deposit. Imbalanced data can lead to a falsely perceived positive effect of a model's accuracy because the input data has bias towards one class. Therefore, an data imbalance issue should be considered to fix for future experiments.We may try: 1) random undersampling, 2) oversampling with SMOTE, 3) a combination of under- and oversampling method using pipeline.

* To try to use other metrics such as 'AUC_weighted' to get better automated ML training . As for a highly unbalanced problem like this, AUC metric is very popular. AUC is acturally preferred over accuracy for binary classification therefore this metric is worth a try.


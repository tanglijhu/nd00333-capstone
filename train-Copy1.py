from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

# +
from azureml.core.dataset import Dataset
#url = 'https://mlstrg130254.blob.core.windows.net/azureml-blobstore-01848383-aa83-4b0d-a1e9-2957a2018b7b/12-12-2020_053200_UTC/heart_failure_clinical_records_dataset.csv'
###dataset = TabularDatasetFactory.from_delimited_files(path = url)
###ds = dataset.to_pandas_dataframe() ### YOUR CODE HERE ###
##dataset=Dataset.Tabular.from_delimited_files(path=url)

# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = 'a0f586ec-8016-4ea9-8248-9bf2299ad437'
resource_group = 'aml-quickstarts-130254'
workspace_name = 'quick-starts-ws-130254'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='heart-failure')
# -



# TODO: Split data into train and test sets.

### YOUR CODE HERE ###a
run = Run.get_context()

def clean_data(data):
  
    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    
    y_df = x_df.pop("DEATH_EVENT")
    return x_df , y_df

x, y = clean_data(dataset)

x.head()

y.head()

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)




def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))




if __name__ == '__main__':
    main()

# +
import numpy as np
import argparse
import os
import pandas as pd

from azureml.core import Run

from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.core.dataset import Dataset

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

# +
train=pd.read_pickle("train.pkl")
test=pd.read_pickle("test.pkl")
target="DEATH_EVENT"
x_train=train.drop(target,axis=1)
y_train=train[target]
x_test=test.drop(target,axis=1)
y_test=test[target]

x_train_values=x_train.values
x_test_values=x_test.values
y_train_values=y_train.values
y_test_values=y_test.values

# -

### YOUR CODE HERE ###a
run = Run.get_context()

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

    os.makedirs('outputs', exist_ok=True)
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(LogisticRegression, 'outputs/model.joblib')


if __name__ == '__main__':
    main()



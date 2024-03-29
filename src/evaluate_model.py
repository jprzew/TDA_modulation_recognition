from dvc.api import params_show
from spocheck import get_models, make_pipeline, load_dataset
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import pandas as pd
import json
from datasets.radioml import index_to_class
from utils import get_repo_path
import config as cfg


def save_results(y_hat, y_test, path):
    """Saves prediction and test-labels to file"""
    y_hat = pd.Series(y_hat).map(index_to_class)
    y_test = pd.Series(y_test).map(index_to_class)

    y_hat.name = 'y_hat'
    y_test.name = 'y_test'

    results = pd.concat([y_hat, y_test], axis=1)
    results.to_csv(path, index=False)


def save_metrics(accuracy, path):
    """Saves evaluation metrics to file"""
    with open(path, 'w') as f:
        json.dump({'accuracy': accuracy}, f)


def main():

    # Read parameters
    params = params_show()['evaluate_model']
    model_name = params['model']
    seed = params['random_seed']
    test_size = params['test_size']

    # Prepare model
    model = get_models()[model_name]
    model = make_pipeline(model)

    # Fit model
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = model_selection \
        .train_test_split(X, y, test_size=test_size, random_state=seed)

    model.fit(X_train, y_train)

    # Test model
    y_hat = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_hat)

    # Save results
    save_metrics(accuracy, get_repo_path() / cfg.TestModel.accuracy_file)
    save_results(y_hat, y_test, get_repo_path() / cfg.TestModel.results_file)


if __name__ == '__main__':
    main()

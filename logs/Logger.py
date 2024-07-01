import neptune
from dotenv import load_dotenv
import os
from typing import Union, List


class Logger():
    def __init__(self, env_path, project, experiment_name=None):

        load_dotenv(env_path)
        self.run = neptune.init_run(
            project=project,
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1NTBhZWU4ZS0wODJiLTQ0MTctOTE3OC00MTA2OTBmMTljNTUifQ==",#os.environ['NEPTUNE_API_TOKEN'],
            name=experiment_name
        )

    def log_hyperparameters(self, params: dict):
        for param, value in params.items():
            self.run[f'hyperparameters/{param}'] = value

    def save_param(self, type_set, metric_name: Union[List[str], str], metric_value: Union[List[float], float]):
        if isinstance(metric_name, List):
            for p_n, p_v in zip(metric_name, metric_value):
                self.run[f"{type_set}/{p_n}"].append(p_v)
        else:
            self.run[f"{type_set}/{metric_name}"].append(metric_value)

    def log_final_val_accuracy(self, value: float):
        self.run['final_metrics/validation_accuracy'] = value

    def log_final_val_target(self, value: float):
        self.run['final_metrics/validation_target'] = value

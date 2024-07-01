from abc import ABC, abstractmethod

import numpy
import numpy as np

from enum import Enum


class PreprocessType(Enum):
    Normalization = 1,
    Standartization = 2


class BaseDataset(ABC):

    def __init__(self, train_set_percent, valid_set_percent):
        self.train_set_percent = train_set_percent
        self.valid_set_percent = valid_set_percent

        self.means = None
        self.stds = None

    @property
    @abstractmethod
    def targets(self):
        # targets variables
        pass

    @property
    @abstractmethod
    def inputs(self):
        # inputs variables
        pass

    def _divide_into_sets(self):
        inputs_length = len(self.inputs)
        indexes = np.random.permutation(inputs_length)

        train_set_size = int(self.train_set_percent * inputs_length)
        valid_set_size = int(self.valid_set_percent * inputs_length)

        randomized_inputs = self.inputs[indexes]
        randomized_targets = self.targets[indexes]

        self.inputs_train = randomized_inputs[:train_set_size]
        self.inputs_valid = randomized_inputs[train_set_size:train_set_size + valid_set_size]
        self.inputs_test = randomized_inputs[train_set_size + valid_set_size:]

        self.targets_train = randomized_targets[:train_set_size]
        self.targets_valid = randomized_targets[train_set_size:train_set_size + valid_set_size]
        self.targets_test = randomized_targets[train_set_size + valid_set_size:]

    def normalization(self): # -> [0 , 1]
        # BONUS TASK
        inputs_min = np.min(self.inputs, axis=0).reshape(self.inputs.shape[1], 1)
        inputs_max = np.max(self.inputs, axis=0).reshape(self.inputs.shape[1], 1)
        difference = inputs_max - inputs_min
        difference[difference == 0] = 1
        self.inputs_train = ((self.inputs_train.T - inputs_min) / difference).T
        self.inputs_valid = ((self.inputs_valid.T - inputs_min) / difference).T
        self.inputs_test = ((self.inputs_test.T - inputs_min) / difference).T

    def standardization(self):
        mean, std = self.__get_data_stats()
        self.inputs_train = ((self.inputs_train.T - mean) / std).T
        self.inputs_valid = ((self.inputs_valid.T - mean) / std).T
        self.inputs_test = ((self.inputs_test.T - mean) / std).T
        return mean, std
    
    def __get_data_stats(self):
        mean = np.mean(self.inputs_train, axis=0).reshape(self.inputs_train.shape[1], 1)
        std = np.std(self.inputs_train, axis=0).reshape(self.inputs_train.shape[1], 1)
        std[std == 0] = 1
        return mean, std




class BaseClassificationDataset(BaseDataset):

    @property
    @abstractmethod
    def k(self):
        # number of classes
        pass

    # @staticmethod
    # def onehotencoding(targets: np.ndarray, number_of_classes: int) -> np.ndarray:
    #     matr = np.zeros((len(targets), number_of_classes))
    #     matr[np.arange(len(targets)), targets] = 1

    #     return matr
    @staticmethod
    def onehotencoding(targets: np.ndarray, number_classes: int) -> np.ndarray:
        targets = np.array(targets).reshape(-1)
        return np.eye(number_classes)[targets]

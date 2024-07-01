import math
from typing import Union
from enum import Enum
from easydict import EasyDict

from ft_datasets.base_dataset_classes import BaseClassificationDataset
from utils import metrics
from utils.common_functions import *
from logs.Logger import Logger

import cloudpickle
import numpy as np


class LogRegStoppingCriteria(Enum):
    Epoch = 1,


class LogRegWeightsInitType(Enum):
    Normal = 1,
    Uniform = 2,
    XavierNormal = 3,
    He = 4,
    He_Uniform = 5


class LogReg:

    def __init__(self, cfg: EasyDict, number_classes: int, input_vector_dimension: int, logging: bool = False,
                env_path: str = None):
        self.k = number_classes
        self.d = input_vector_dimension

        self.cfg = cfg

        getattr(self, f'weights_init_{cfg.weights_init_type.name}')(**cfg.weights_init_kwargs)

        self.logging = logging
        if logging:
            self.experiment_name = generate_experiment_name(self.cfg.reg_coefficient, self.cfg.gamma)

            self.logger = Logger(env_path, cfg.project_name, self.experiment_name)
            self.logger.log_hyperparameters(params=cfg)
        
            

        else:
            self.logger = None

        self.data_for_plots = {'epochs': [],
                               'target_function_value_train': [],
                               'accuracy_train': [],
                               'target_function_value_valid':[],
                               'accuracy_valid': []}


    @property
    def weights(self):
        return self.weights_matrix

    @property
    def bias(self):
        return self.__bias

    def looging_params(self, epoch, target_function_value_train, accuracy_train, target_function_value_valid, accuracy_valid):

        self.data_for_plots['epochs'].append(epoch)
        self.data_for_plots['target_function_value_train'].append(target_function_value_train)
        self.data_for_plots['accuracy_train'].append(accuracy_train)
        self.data_for_plots['target_function_value_valid'].append(target_function_value_valid)
        self.data_for_plots['accuracy_valid'].append(accuracy_valid)

        if self.logging:
            self.logger.save_param('train', 'accuracy', accuracy_train)
            self.logger.save_param('train', 'target_value', target_function_value_train)

            self.logger.save_param('valid', 'accuracy', accuracy_valid)
            self.logger.save_param('valid', 'target_value', target_function_value_valid)


    def weights_init_normal(self, sigma):
        self.weights_matrix = 0.0 + sigma * np.random.randn(self.k, self.d)
        self.__bias = 0.0 + sigma * np.random.randn(self.k, 1)

    def weights_init_uniform(self, epsilon):
        # BONUS TASK
        self.weights_matrix = np.random.uniform(0, epsilon, size=(self.k, self.d))
        self.__bias = np.random.uniform(0, epsilon, size=(self.k, 1))

    def weights_init_xavier(self, n_in, n_out):
        # BONUS TASK
        limit = math.sqrt(6.0 / (n_in + n_out))
        self.weights_matrix = np.random.uniform(-limit, limit, size=(self.k, self.d))
        self.__bias = np.random.uniform(-limit, limit, size=(self.k, 1))

    def weights_init_he(self, n_in):
        # BONUS TASK
        std = math.sqrt(2.0 / n_in)
        self.weights_matrix = 0.0 + std * np.random.randn(self.k, self.d)
        self.__bias = 0.0 + std * np.random.randn(self.k, 1)

    def __softmax(self, model_output: np.ndarray) -> np.ndarray:
        """
               Computes the softmax function on the model output.

               The formula for softmax function is:
               y_j = e^(z_j) / Σ(i=0 to K-1) e^(z_i)

               where:
               - y_j is the softmax probability of class j,
               - z_j is the model output for class j before softmax,
               - K is the total number of classes,
               - Σ denotes summation.

               For numerical stability, subtract the max value of model_output before exponentiation:
               z_j = z_j - max(model_output)

               Parameters:
               model_output (np.ndarray): The model output before softmax.

               Returns:
               np.ndarray: The softmax probabilities.
            TODO implement this function
               """
        # z_exp = np.exp(model_output - np.max(model_output, axis=1, keepdims=True)) ## Нормализуем значения чтобы не было переполнения
        # return z_exp / np.sum(z_exp, axis=1, keepdims=True)
        model_output = model_output - np.max(model_output)
        return np.exp(model_output) / np.sum(np.exp(model_output), axis=0)

    def get_model_confidence(self, inputs: np.ndarray) -> np.ndarray:
        """
                Calculates model confidence using the formula:
                y(x, b, W) = Softmax(Wx + b) = Softmax(z)

                Parameters:
                inputs (np.ndarray): The input data.

                Returns:
                np.ndarray: The model confidence.
        """
        z = self.__get_model_output(inputs)
        y = self.__softmax(z)
        return y

    def __get_model_output(self, inputs: np.ndarray) -> np.ndarray:
        """
        This function computes the model output by applying a linear transformation
        to the input data.

        The linear transformation is defined by the equation:
        z = W * x + b

        where:
        - W (a KxD matrix) represents the weight matrix,
        - x (a DxN matrix, also known as 'inputs') represents the input data,
        - b (a vector of length K) represents the bias vector,
        - z represents the model output before activation.

        Returns:
        np.ndarray: The model output before softmax.

        """    
        return np.dot(self.weights_matrix, inputs.T) + self.__bias
        return np.dot(self.weights, inputs.T) + self.b

    

    def __get_gradient_w(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: np.ndarray) -> np.ndarray:
        gradient_w = np.dot(model_confidence - targets.T, inputs)
        gradient_w_with_regularization = gradient_w + self.cfg.reg_coefficient * self.weights_matrix
        return gradient_w_with_regularization

    def __get_gradient_b(self, targets: np.ndarray, model_confidence: np.ndarray) -> np.ndarray:
        u = np.array([np.ones(targets.shape[0])])
        return np.dot(model_confidence - targets.T, u.T)

    def __weights_update(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: np.ndarray):
        self.weights_matrix = self.weights_matrix - self.cfg.gamma * self.__get_gradient_w(inputs, targets, model_confidence)
        self.__bias = self.__bias - self.cfg.gamma * self.__get_gradient_b(targets, model_confidence)

    def __gradient_descent_step(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                epoch: int, inputs_valid: Union[np.ndarray, None] = None,
                                targets_valid: Union[np.ndarray, None] = None):
        #  one step in Gradient descent:
        #  calculate model confidence;
        #  target function value calculation;
        #
        #  update weights
        #   you can add some other steps if you need
        # log your results in Neptune
        """
        :param targets_train: onehot-encoding
        :param epoch: number of loop iteration
        """
        targets_train_onehot = BaseClassificationDataset.onehotencoding(targets_train, self.k) ## Матрица ответов [0, 1] -> [1, 0][0,1]

        y = self.get_model_confidence(inputs_train) ## v
        target_function_value = self.__target_function_value(inputs_train, targets_train_onehot, y) ## сумарный лосс по трейну
        
        self.__validate(inputs_train, targets_train, y)

        accuracy_train = self.accuracy
        

        accuracy_train = self.accuracy
        confusion_matrix_train = self.confusion_matrix
        
        if inputs_valid is not None and targets_valid is not None:
            
            targets_valid_onehot = BaseClassificationDataset.onehotencoding(targets_valid, self.k)
            y_valid = self.get_model_confidence(inputs_valid) ## v
            target_function_value_valid = self.__target_function_value(inputs_valid, targets_valid_onehot, y_valid)
            
            accuracy_valid = self.__validate(inputs_valid, targets_valid)
            self.looging_params(epoch, target_function_value, accuracy_train, target_function_value_valid, accuracy_valid)

        if epoch % 10 == 0:
            print(f'Target function value on train set: {target_function_value}')
            print('Confusion matrix on train set:')
            print(confusion_matrix_train)
            print(f'Accuracy on train set: {accuracy_train}')
            if inputs_valid is not None and targets_valid is not None:
                print('Confusion matrix on validation set:')
                print(self.confusion_matrix)
                print(f'Accuracy on validation set: {self.accuracy}')
            print()



        self.__weights_update(inputs_train, targets_train_onehot, y)

    def gradient_descent_epoch(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                               inputs_valid: Union[np.ndarray, None] = None,
                               targets_valid: Union[np.ndarray, None] = None):
        # loop stopping criteria - number of iterations of gradient_descent
        # while not stopping criteria
        #   self.__gradient_descent_step(inputs, targets)

        for epoch in range(self.cfg.nb_epoch):
            print(f'epoch = {epoch}')
            self.__gradient_descent_step(inputs_train, targets_train, epoch, inputs_valid, targets_valid)
            

    def gradient_descent_gradient_norm(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                       inputs_valid: Union[np.ndarray, None] = None,
                                       targets_valid: Union[np.ndarray, None] = None):
        #  gradient_descent with gradient norm stopping criteria BONUS TASK

        targets_train_onehot = BaseClassificationDataset.onehotencoding(targets_train, self.k)
        epoch = 0
        while True:
            y = self.get_model_confidence(inputs_train)
            gradient = np.append(self.__get_gradient_w(inputs_train, targets_train_onehot, y),
                                 self.__get_gradient_b(targets_train_onehot, y))
            gradient_norm = np.linalg.norm(gradient)
            print(f'epoch = {epoch}, gradient_norm = {gradient_norm}')
            self.__gradient_descent_step(inputs_train, targets_train, epoch, inputs_valid, targets_valid)
            if gradient_norm < self.cfg.gradient_norm_threshold:
                break
            epoch += 1

    def gradient_descent_difference_norm(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                         inputs_valid: Union[np.ndarray, None] = None,
                                         targets_valid: Union[np.ndarray, None] = None):
        #  gradient_descent with stopping criteria - norm of difference between ￼w_k-1 and w_k;￼BONUS TASK
        epoch = 0
        w_prev = self.weights_matrix
        print(f'epoch = {epoch}')
        while True:
            self.__gradient_descent_step(inputs_train, targets_train, epoch, inputs_valid, targets_valid)
            w_current = self.weights_matrix
            difference_norm = np.linalg.norm(w_current - w_prev)
            epoch += 1
            print(f'epoch = {epoch}, difference_norm = {difference_norm}')
            if difference_norm < self.cfg.difference_norm_threshold:
                break
            w_prev = w_current

    def gradient_descent_metric_value(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                      inputs_valid: Union[np.ndarray, None] = None,
                                      targets_valid: Union[np.ndarray, None] = None):
        #  gradient_descent with stopping criteria - metric (accuracy, f1 score or other) value on validation set is not growing;￼
        self.__validate(inputs_valid, targets_valid)
        accuracy_prev = self.accuracy
        epoch = 0
        count = 0
        print(f'epoch = {epoch}, accuracy_valid = {self.accuracy} ({count})')
        while True:
            self.__gradient_descent_step(inputs_train, targets_train, epoch, inputs_valid, targets_valid)
            self.__validate(inputs_valid, targets_valid)
            accuracy_current = self.accuracy
            count = count + 1 if accuracy_current == accuracy_prev else 0
            epoch += 1
            print(f'epoch = {epoch}, accuracy_valid = {accuracy_current} ({count})')
            if count >= self.cfg.nb_repeats:
                break
            accuracy_prev = accuracy_current

    def batch_gradient_descent(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                               inputs_valid: Union[np.ndarray, None] = None,
                               targets_valid: Union[np.ndarray, None] = None):
        # BONUS TASK
        number_of_bathes = inputs_train.shape[0] // self.cfg.batch_size
        split_intervals = []
        
        for i in range(1, number_of_bathes + 1):
            split_intervals.append(int(self.cfg.batch_size * i)) ##32 65 96 128..
        
        for epoch in range(self.cfg.nb_epoch):
            randomize = np.arange(inputs_train.shape[0])
            np.random.shuffle(randomize)
            inputs_train = inputs_train[randomize]
            targets_train = targets_train[randomize]
            
            batches_input = np.split(inputs_train, split_intervals)
            batches_target = np.split(targets_train, split_intervals)


            for i in range(number_of_bathes - 1):
                print(f'epoch = {epoch}, batch = {i}')
                targets_train_onehot = BaseClassificationDataset.onehotencoding(batches_target[i], self.k)
                y = self.get_model_confidence(batches_input[i])
                self.__weights_update(batches_input[i], targets_train_onehot, y)
                if epoch == self.cfg.nb_epoch - 1 and i == number_of_bathes - 2:
                    target_function_value = self.__target_function_value(batches_input[i], targets_train_onehot, y)
                    self.__validate(batches_input[i], batches_target[i], y)
                    print(f'Target function value on train set: {target_function_value}')
                    print('Confusion matrix on train set:')
                    print(self.confusion_matrix)
                    print(f'Accuracy on train set: {self.accuracy}')
                    if inputs_valid is not None and targets_valid is not None:
                        self.__validate(inputs_valid, targets_valid)
                        print('Confusion matrix on validation set:')
                        print(self.confusion_matrix)
                        print(f'Accuracy on validation set: {self.accuracy}')


    

    def train(self, inputs_train: np.ndarray, targets_train: np.ndarray, inputs_valid: Union[np.ndarray, None] = None, targets_valid: Union[np.ndarray, None] = None):
        print(f'gradient_descent_{self.cfg.gd_stopping_criteria.name}')
        getattr(self, f'gradient_descent_{self.cfg.gd_stopping_criteria.name}')(inputs_train, targets_train, inputs_valid, targets_valid)

    def __target_function_value(self, inputs: np.ndarray, targets: np.ndarray,
                                model_confidence: Union[np.ndarray, None] = None) -> float:

        """
        This function computes the target function value based on the formula:

        Σ(i=0 to N-1) Σ(k=0 to K-1) t_ik * (ln(Σ(l=0 to K-1) e^(z_il)) - z_ik)
        where:
        - N is the size of the data set,
        - K is the number of classes,
        - t_{ik} is the target value for data point i and class k,
        - z_{il} is the model output before softmax for data point i and class l,
        - z is the model output before softmax (matrix z).

        Parameters:
        inputs (np.ndarray): The input data.
        targets (np.ndarray): The target data.
        z (Union[np.ndarray, None]): The model output before softmax. If None, it will be computed.

        Returns:
        float: The value of the target function.
        """
        if model_confidence is None:
            model_confidence = self.get_model_confidence(inputs)
        value = 0.0
        for i in range(inputs.shape[0]):
            for k in range(self.k):
                value += (targets[i][k] * (np.log(np.sum(np.exp(model_confidence), axis=0))[i]
                                           - model_confidence[k][i]))
        value += (np.sum(np.square(self.weights_matrix)) * self.cfg.reg_coefficient / 2)
        return value

    def __validate(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: Union[np.ndarray, None] = None):
        if model_confidence is None:
            model_confidence = self.get_model_confidence(inputs)
        predictions = np.argmax(model_confidence, axis=0)
        self.confusion_matrix = metrics.confusion_matrix(predictions, targets)
        self.accuracy = metrics.accuracy_mat(self.confusion_matrix)
        return self.accuracy
    
    def __call__(self, inputs: np.ndarray):
        model_confidence = self.get_model_confidence(inputs)
        predictions = np.argmax(model_confidence, axis=0)
        return predictions
    

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            return cloudpickle.load(f)
import numpy as np

from configs.logistic_regression_config import cfg
from models.logistic_regression_model import LogReg
from models.logistic_regression_model import LogRegWeightsInitType
from models.logistic_regression_model import LogRegStoppingCriteria

#from models.LogReg import LogReg
from ft_datasets.digits_dataset import Digits
from ft_datasets.base_dataset_classes import PreprocessType
from utils.visualisation import Visualisation
from utils.enums import SetType

import pickle

def experiment1():
    digits = Digits(cfg)

    model = LogReg(cfg, digits.k, digits.d,
                logging=False,
                env_path='../../env.env'
                )
    
    print('---------------------------- TRAIN ----------------------------')
    model.train(digits(SetType.train)['inputs'], digits(SetType.train)['targets'],
                        digits(SetType.valid)['inputs'], digits(SetType.valid)['targets'])
    #model.batch_gradient_descent(digits(SetType.train)['inputs'], digits(SetType.train)['targets'],
    #                                                 digits(SetType.valid)['inputs'], digits(SetType.valid)['targets'])

    Visualisation.visualise_metrics(epochs=model.data_for_plots['epochs'],
                                    metrics=model.data_for_plots['target_function_value_train'],
                                    plot_title='Target function values on training set',
                                    y_title='Target function value')
    Visualisation.visualise_metrics(epochs=model.data_for_plots['epochs'],
                                    metrics=model.data_for_plots['accuracy_train'],
                                    plot_title='Accuracy values on training set',
                                    y_title='Accuracy')
    Visualisation.visualise_metrics(epochs=model.data_for_plots['epochs'],
                                    metrics=model.data_for_plots['accuracy_valid'],
                                    plot_title='Accuracy values on validation set',
                                    y_title='Accuracy')
    
    pickled_representation_of_log_reg_model = pickle.dumps(model)
    return pickled_representation_of_log_reg_model





def experiment_2(digits_dataset, pickled_representation_of_log_reg_model: bytes):
    # BONUS TASK
    log_reg_model = pickle.loads(pickled_representation_of_log_reg_model)
    # BONUS TASK
    inputs = digits_dataset(SetType.valid)['inputs']
    inputs = (inputs.T * digits_dataset.std + digits_dataset.mean).T
    targets = digits_dataset(SetType.valid)['targets']
    y = log_reg_model.get_model_confidence(inputs)
    predictions = np.argmax(y, axis=0)
    maximums = np.max(y, axis=0)
    indices = maximums.argsort()[::-1]
    predictions = predictions[indices]
    inputs = inputs[indices]
    targets = targets[indices]
    right = []
    right_predictions = []
    wrong = []
    wrong_predictions = []
    i = 0
    while (len(right) < 3 or len(wrong) < 3) and i < len(predictions):
        if len(right) < 3 and predictions[i] == targets[i]:
            right.append(inputs[i])
            right_predictions.append(predictions[i])
        if len(wrong) < 3 and predictions[i] != targets[i]:
            wrong.append(inputs[i])
            wrong_predictions.append(predictions[i])
        i += 1
    Visualisation.visualise_images(images=right,
                                   predictions=right_predictions,
                                   plot_title='3 images for which classifier saved most confidence class prediction '
                                              'and was right')
    Visualisation.visualise_images(images=wrong,
                                   predictions=wrong_predictions,
                                   plot_title='3 images for which classifier saved most confidence class prediction '
                                              'and was wrong')

if __name__ == "__main__":
    digits = Digits(cfg)

    pickled_representation_of_log_reg_model = experiment1()    

    experiment_2(digits,pickled_representation_of_log_reg_model)

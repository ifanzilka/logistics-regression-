from easydict import EasyDict

from utils.enums import DataProcessTypes, WeightsInitType, GDStoppingCriteria

cfg = EasyDict()


cfg.project_name = "DigitsLogReg"

# data
cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1
cfg.data_preprocess_type = DataProcessTypes.standardization
#cfg.data_preprocess_type = DataProcessTypes.normalization
#cfg.data_preprocess_type = DataProcessTypes.nothing

# training
cfg.weights_init_type = WeightsInitType.normal
cfg.weights_init_kwargs = {'sigma': 1}
# cfg.weights_init_type = WeightsInitType.uniform
# cfg.weights_init_kwargs = {'epsilon': 1}
# cfg.weights_init_type = WeightsInitType.xavier
# cfg.weights_init_kwargs = {'n_in': 1, 'n_out': 1}
# cfg.weights_init_type = WeightsInitType.he
# cfg.weights_init_kwargs = {'n_in': 1}

# gradient descent
cfg.gamma = 0.01

# regularization
cfg.reg_coefficient = 0.01

# stopping criteria
cfg.gd_stopping_criteria = GDStoppingCriteria.epoch
#cfg.gd_stopping_criteria = GDStoppingCriteria.gradient_norm
# cfg.gd_stopping_criteria = GDStoppingCriteria.difference_norm
# cfg.gd_stopping_criteria = GDStoppingCriteria.metric_value
cfg.nb_epoch = 12
cfg.gradient_norm_threshold = 3
cfg.difference_norm_threshold = 0.02
cfg.nb_repeats = 20

# batch
cfg.batch_size = 32

cfg.visualize = True 


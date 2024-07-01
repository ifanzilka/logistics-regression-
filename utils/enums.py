from enum import IntEnum

DataProcessTypes = IntEnum('DataProcessTypes', ('standardization', 'normalization','nothing'   ))
SetType = IntEnum('SetType', ('train', 'valid', 'test'))
WeightsInitType = IntEnum('WeightsInitType', ('normal', 'uniform', 'xavier', 'he'))
GDStoppingCriteria = IntEnum('GDStoppingCriteria', ('epoch', 'gradient_norm', 'difference_norm', 'metric_value'))

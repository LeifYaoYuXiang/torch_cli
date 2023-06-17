from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

IMPLEMENTED_CLASSIFICATION_METRIC = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
IMPLEMENTED_REGRESSION_METRIC = ['mse', 'rmse', 'mae', 'r2', 'evs']


def accuracy(y_true, y_pred, binary=True):
    return accuracy_score(y_true, y_pred)


def precision(y_true, y_pred, binary=True):
    if binary:
        return precision_score(y_true, y_pred)
    else:
        return precision_score(y_true, y_pred, average='weighted')


def recall(y_true, y_pred, binary=True):
    if binary:
        return recall_score(y_true, y_pred)
    else:
        return recall_score(y_true, y_pred, average='weighted')


def f1(y_true, y_pred, binary=True):
    if binary:
        return f1_score(y_true, y_pred)
    else:
        return f1_score(y_true, y_pred, average='weighted')


def roc_auc(y_true, y_pred, binary=True):
    if binary:
        return roc_auc_score(y_true, y_pred)
    else:
        return roc_auc_score(y_true, y_pred, average='weighted')


def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def evs(y_true, y_pred):
    return explained_variance_score(y_true, y_pred)


# class for classification metric: 分类评价指标
class ClassificationMetric:
    def __init__(self, decimal: int = 4, binary: bool = True):
        self.decimal = decimal
        self.binary = binary

    def metric(self, y_true, y_pred, eval_metric: list):
        for each_metric in eval_metric:
            assert each_metric in IMPLEMENTED_CLASSIFICATION_METRIC
        metric_dict = {}
        for each_metric in eval_metric:
            metric_dict[each_metric] = round(globals()[each_metric](y_true, y_pred, self.binary), self.decimal)
        return metric_dict


# class for regression metric: 回归评价指标
class RegressionMetric:
    def __init__(self, decimal: int = 4):
        self.decimal = decimal

    def metric(self, y_true, y_pred, eval_metric: list):
        for each_metric in eval_metric:
            assert each_metric in IMPLEMENTED_REGRESSION_METRIC
        metric_dict = {}
        for each_metric in eval_metric:
            metric_dict[each_metric] = round(globals()[each_metric](y_true, y_pred), self.decimal)
        return metric_dict

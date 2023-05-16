IMPLEMENTED_CLASSIFICATION_METRIC = []
IMPLEMENTED_REGRESSION_METRIC = []


class ClassificationMetric:
    def __init__(self, decimal: int, binary: True):
        self.decimal = decimal
        self.binary = binary
        self.build_metric()

    def build_metric(self):
        pass

    def metric(self, y_true, y_pred, eval_metric: list):
        for each_metric in eval_metric:
            assert each_metric in IMPLEMENTED_CLASSIFICATION_METRIC
        metric_dict = {}
        for each_metric in eval_metric:
            pass

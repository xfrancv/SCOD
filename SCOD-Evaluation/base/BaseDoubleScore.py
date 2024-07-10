import numpy.typing as npt
from helpers.metrics import Metrics

class BaseDoubleScore:
    def __init__(self,
                y_true: npt.NDArray[int],
                 y_pred: npt.NDArray[int],
                 score_1: npt.NDArray[float],
                 score_2: npt.NDArray[float]):
        self.y_true = y_true
        self.y_pred = y_pred
        self.score_1 = score_1
        self.score_2 = score_2
        self.is_fit = False

    def get_score_1_metrics(self) -> Metrics:
        return Metrics(y_pred=self.y_pred, y_true=self.y_true, score=self.score_1)

    def get_score_2_metrics(self) -> Metrics:
        return Metrics(y_pred=self.y_pred, y_true=self.y_true, score=self.score_2)
    
    def self_evaluate(self):
        return self.evaluate(y_true = self.y_true,
                        y_pred = self.y_pred,
                        score_1 = self.score_1,
                        score_2 = self.score_2)
        
    def evaluate(self):
        raise NotImplementedError
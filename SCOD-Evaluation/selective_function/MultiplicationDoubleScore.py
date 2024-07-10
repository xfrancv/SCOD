import numpy as np
import numpy.typing as npt
import warnings
from helpers.metrics import Metrics
from helpers.constants import N_TPR_SAMPLES, BETA, PEDANTIC
from base.BaseDoubleScore import BaseDoubleScore
from tqdm import tqdm
import matplotlib.pyplot as plt


class MultiplicationDoubleScore(BaseDoubleScore):
    def __init__(self,
                 y_true: npt.NDArray[int],
                 y_pred: npt.NDArray[int],
                 score_1: npt.NDArray[float],
                 score_2: npt.NDArray[float]):
        super().__init__(y_true=y_true, y_pred=y_pred, score_1=score_1, score_2=score_2)

        self.threshold = None
        self.metrics = Metrics(y_pred=y_pred, y_true=y_true, score=score_1*score_2)
        self.fast_metrics = self.metrics
        self.is_fit = False

    def reset_fit(self):
        self.is_fit = False
        self.threshold = None

    def fit(self, **kwargs):
        if not self.is_fit:
            self.fit_impl(self.y_true,
                            self.y_pred,
                            self.score_1,
                            self.score_2,
                            **kwargs)
        else:
            if PEDANTIC:
                raise RuntimeError(
                    f"Model is already fit. Call the '{self.reset_fit.__name__}' method before trying to refit.")
            else:
                warnings.warn(
                    f"Model is already fit. Call the '{self.reset_fit.__name__}' method before trying to refit.")

    def fit_impl(self,
                 y_true: npt.NDArray[int],
                 y_pred: npt.NDArray[int],
                 score_1: npt.NDArray[float],
                 score_2: npt.NDArray[float],
                 optimization_condition: str = 'tpr',
                 optimization_objective: str = 'joint risk',
                 minimize: bool = True,
                 **kwargs):

        if minimize:
            best_value = np.inf
        else:
            best_value = -np.inf

        final_threshold = None
        metrics = self.metrics

        if optimization_condition == 'tpr':
            if 'min_tpr' in kwargs:
                min_tpr = kwargs.get('min_tpr')
                select = (metrics.tpr >= min_tpr)
            else:
                raise RuntimeError(
                    f"'min_tpr' parameter is required when optimization condition is '{optimization_condition}'.")

        elif optimization_condition == 'fpr':
            if 'max_fpr' in kwargs:
                max_fpr = kwargs.get('max_fpr')
                select = (metrics.fpr <= max_fpr)
            else:
                raise RuntimeError(
                    f"'max_fpr' parameter is required when optimization condition is '{optimization_condition}'.")
        else:
            raise NotImplementedError

        if optimization_objective == 'joint risk':
            if 'beta' in kwargs:
                beta = kwargs.get('beta')
            else:
                beta = BETA
            potential_values = beta * metrics.sel_risk[select] + \
                (1-beta)*metrics.fpr[select]

        elif optimization_objective == 'selective risk':
            potential_values = metrics.sel_risk[select]

        elif optimization_objective == 'fpr':
            potential_values = metrics.fpr[select]

        elif optimization_objective == 'tpr':
            potential_values = metrics.tpr[select]

        else:
            raise NotImplementedError(
                "Optimization objective function now known.")

        potential_thresholds = np.concatenate(
            [np.array([np.max(metrics.score)+1]), metrics.score])[select]

        if potential_thresholds.size == 0:
            raise RuntimeError

        if minimize:
            select_best = np.argmin(potential_values)
            best_value = potential_values[select_best]
            final_threshold = potential_thresholds[select_best]
        else:
            select_best = np.argmax(potential_values)
            best_value = potential_values[select_best]
            final_threshold = potential_thresholds[select_best]

        if final_threshold is not None:
            self.threshold = final_threshold
            self.is_fit = True

    def evaluate(self,
                 y_true: npt.NDArray[int],
                 y_pred: npt.NDArray[int],
                 score_1: npt.NDArray[float],
                 score_2: npt.NDArray[float],
                 beta: float = BETA) -> dict:
        if self.is_fit:
            metrics = self.fast_metrics

            ix = np.argmin((metrics.score >= self.threshold))
            return {'fpr': metrics.fpr[ix],
                    'tpr': metrics.tpr[ix],
                    'sel_risk': metrics.sel_risk[ix],
                    'joint_risk': beta*metrics.sel_risk[ix] + (1-beta)*metrics.fpr[ix],
                    'beta': beta,
                    'threshold': self.threshold}
        else:
            if PEDANTIC:
                raise RuntimeError(
                    "Attempting to evaluate model that has not been fit!")
            else:
                warnings.warn(
                    "Attempting to evaluate model that has not been fit!", RuntimeWarning)

    def evaluate_at_tpr(self,
                        y_true: npt.NDArray[int],
                        y_pred: npt.NDArray[int],
                        score_1: npt.NDArray[float],
                        score_2: npt.NDArray[float],
                        at_tpr: float,
                        beta: float = BETA) -> dict:
        if self.is_fit:
            metrics = self.fast_metrics

            ix = np.argmin((metrics.tpr <= at_tpr))
            return {'fpr': metrics.fpr[ix],
                    'tpr': metrics.tpr[ix],
                    'sel_risk': metrics.sel_risk[ix],
                    'joint_risk': beta*metrics.sel_risk[ix] + (1-beta)*metrics.fpr[ix],
                    'beta': beta,
                    'alpha': self.alpha}
        else:
            if PEDANTIC:
                raise RuntimeError(
                    "Attempting to evaluate model that has not been fit!")
            else:
                warnings.warn(
                    "Attempting to evaluate model that has not been fit!", RuntimeWarning)

    def plot_decision_boundary(self,
                               axis: plt.Axes,
                               thresholds: int | npt.ArrayLike,
                               xmin: float = 0,
                               xmax: float = 1,
                               ymin: float = 0,
                               ymax: float = 1,
                               Nx: int = 100,
                               Ny: int = 100) -> None:
        if self.is_fit:
            score_1 = np.linspace(xmin, xmax, Nx, endpoint=True)
            score_2 = np.linspace(ymin, ymax, Ny, endpoint=True)
            s1, s2 = np.meshgrid(score_1, score_2)
            score = s1.ravel()*s2.ravel()
            score = score.reshape(s1.shape)
            cs = axis.contour(s1, s2, score, levels=thresholds, alpha=0.5,
                                colors='k', linestyles='-', label='Linear', linewidths=3)

        else:
            if PEDANTIC:
                raise RuntimeError(
                    "Attempting to evaluate model that has not been fit!")
            else:
                warnings.warn(
                    "Attempting to evaluate model that has not been fit!", RuntimeWarning)

    def compute_joint_risk_vs_tpr_curve(self,
                                        beta: float = BETA,
                                        n_tpr_samples: int = N_TPR_SAMPLES) -> [npt.NDArray[float], npt.NDArray[float], float]:
        """
        Computes the JointRisk-TPR curve. 
        For each possible value of TPR, the optimal (minimizing beta*selective risk + (1-beta)*fpr) selective classifier is found.

        Args:
            beta (float, optional): Parameter defining the metric, i.e., how selective risk and FPR are mixed.
            n_tpr_samples (int, optional): Sampling parameter for the X-axis. Defaults to N_TPR_SAMPLES.

        Returns:
            [npt.NDArray[np.float], npt.NDArray[np.float], float]: 1) Risk and 2) TPR arrays and 3) area under the curve.
        """

        tpr = np.linspace(0, 1, n_tpr_samples, endpoint=True)
        joint_risk = np.ones_like(tpr)*np.Inf
        for i, min_tpr in tqdm(enumerate(tpr), desc="Optimization, Multiplication", position=2, leave=False, total=len(tpr)):
            metric = self.metrics
            joint_risk[i] = metric.joint_risk[np.argmin(np.abs(metric.tpr - min_tpr))]

        area_under_curve = np.trapz(joint_risk, tpr)

        return joint_risk, tpr, area_under_curve

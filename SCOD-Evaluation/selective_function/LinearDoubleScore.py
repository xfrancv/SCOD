import numpy as np
import numpy.typing as npt
import warnings
from helpers.metrics import Metrics
from helpers.constants import N_TPR_SAMPLES, N_ALPHA_SAMPLES, BETA, PEDANTIC, LIMIT_ALPHA
from base.BaseDoubleScore import BaseDoubleScore
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib


class LinearDoubleScore(BaseDoubleScore):
    def __init__(self,
                 y_true: npt.NDArray[int],
                 y_pred: npt.NDArray[int],
                 score_1: npt.NDArray[float],
                 score_2: npt.NDArray[float],
                 fast=False):
        super().__init__(y_true=y_true, y_pred=y_pred, score_1=score_1, score_2=score_2)

        self.alpha = None
        self.threshold = None
        self.fast_metrics = None
        self.fast = fast

    def reset_fit(self):
        self.is_fit = False

    def fit(self, **kwargs):
        if not self.is_fit:
            if not self.fast:
                self.fit_impl(self.y_true,
                              self.y_pred,
                              self.score_1,
                              self.score_2,
                              **kwargs)
            else:
                self.fit_fast_impl(self.y_true,
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
                 n_alpha_samples: int = N_ALPHA_SAMPLES,
                 optimization_condition: str = 'tpr',
                 optimization_objective: str = 'joint risk',
                 minimize: bool = True,
                 **kwargs):

        if minimize:
            best_value = np.inf
        else:
            best_value = -np.inf

        final_threshold = None
        final_alpha = None

        if LIMIT_ALPHA:
            possible_alphas = sorted([i for i in np.linspace(
                0, np.pi/2, n_alpha_samples, endpoint=True)] + [0, np.pi/2.])
        else:
            possible_alphas = sorted([i for i in np.linspace(
                0, 2*np.pi, n_alpha_samples, endpoint=True)] + [0, np.pi/2., np.pi, np.pi*3./2.])

        for alpha in tqdm(possible_alphas, desc="Optimization, Linear", position=2, leave=False):
            score = np.cos(alpha)*score_1+np.sin(alpha)*score_2
            metrics = Metrics(y_true, y_pred, score)

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
                continue

            if minimize:
                if np.min(potential_values) <= best_value:
                    select_best = np.argmin(potential_values)
                    best_value = potential_values[select_best]
                    final_threshold = potential_thresholds[select_best]
                    final_alpha = alpha
            else:
                if np.max(potential_values) >= best_value:
                    select_best = np.argmax(potential_values)
                    best_value = potential_values[select_best]
                    final_threshold = potential_thresholds[select_best]
                    final_alpha = alpha

        if final_alpha is not None:
            self.alpha = final_alpha
            self.threshold = final_threshold
            self.is_fit = True
        else:
            warnings.warn("Failed to fit.", RuntimeWarning)

    def fit_fast_impl(self,
                      y_true: npt.NDArray[int],
                      y_pred: npt.NDArray[int],
                      score_1: npt.NDArray[float],
                      score_2: npt.NDArray[float],
                      optimization_condition: str = 'tpr',
                      optimization_objective: str = 'joint risk',
                      minimize: bool = True,
                      **kwargs):

        if not minimize:
            raise NotImplementedError

        if optimization_condition == 'tpr':
            if 'min_tpr' in kwargs:
                min_tpr = kwargs.get('min_tpr')

                # compensating for the mixture, i.e., dividing by 'param_a', should be done during score export from OpenOOD
                score = 1-score_1 - BETA*min_tpr/(1-BETA) * score_2

                self.fast_metrics = Metrics(
                    y_pred=y_pred, y_true=y_true, score=-score)
                self.threshold = self.fast_metrics.score[np.argmin(
                    np.abs(self.fast_metrics.tpr - min_tpr))]
                self.is_fit = True
            else:
                raise RuntimeError(
                    f"'min_tpr' parameter is required when optimization condition is '{optimization_condition}'.")
        else:
            raise NotImplementedError(
                "Optimization objective function now known.")

    def evaluate(self,
                 y_true: npt.NDArray[int],
                 y_pred: npt.NDArray[int],
                 score_1: npt.NDArray[float],
                 score_2: npt.NDArray[float],
                 beta: float = BETA) -> dict:
        if self.is_fit:
            if self.fast:
                metrics = self.fast_metrics
            else:
                score = np.cos(self.alpha)*score_1+np.sin(self.alpha)*score_2
                metrics = Metrics(y_true, y_pred, score)

            # ix = sum(metrics.score >= self.threshold)
            ix = np.argmin((metrics.score >= self.threshold))
            return {'fpr': metrics.fpr[ix],
                    'tpr': metrics.tpr[ix],
                    'sel_risk': metrics.sel_risk[ix],
                    'joint_risk': beta*metrics.sel_risk[ix] + (1-beta)*metrics.fpr[ix],
                    'beta': beta,
                    'alpha': self.alpha,
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
            if self.fast:
                metrics = self.fast_metrics
            else:
                score = np.cos(self.alpha)*score_1+np.sin(self.alpha)*score_2
                metrics = Metrics(y_true, y_pred, score)

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
                               Nx: int = 1000,
                               Ny: int = 1000,
                               color='k') -> None:
        if self.is_fit:
            if not self.fast:
                score_1 = np.linspace(xmin, xmax, Nx, endpoint=True)
                score_2 = np.linspace(ymin, ymax, Ny, endpoint=True)
                s1, s2 = np.meshgrid(score_1, score_2)
                score = np.cos(self.alpha)*s1.ravel() + \
                    np.sin(self.alpha)*s2.ravel()
                score = score.reshape(s1.shape)
                cs = axis.contour(s1, s2, score, levels=thresholds, alpha=0.8,
                                  colors=color, linestyles='-', label='Linear', linewidths=3)
                
                proxy = matplotlib.lines.Line2D([], [], alpha=0.8, color=color, linestyle='-', label='Linear', linewidth=3)
                return proxy
            
            else:
                raise NotImplementedError

        else:
            if PEDANTIC:
                raise RuntimeError(
                    "Attempting to evaluate model that has not been fit!")
            else:
                warnings.warn(
                    "Attempting to evaluate model that has not been fit!", RuntimeWarning)

    def compute_joint_risk_vs_tpr_curve(self,
                                        beta: float = BETA,
                                        n_tpr_samples: int = N_TPR_SAMPLES,
                                        n_alpha_samples: int = N_ALPHA_SAMPLES) -> [npt.NDArray[float], npt.NDArray[float], float]:
        """
        Computes the JointRisk-TPR curve. 
        For each possible value of TPR, the optimal (minimizing beta*selective risk + (1-beta)*fpr) selective classifier is found.

        Args:
            beta (float, optional): Parameter defining the metric, i.e., how selective risk and FPR are mixed.
            n_tpr_samples (int, optional): Sampling parameter for the X-axis. Defaults to N_TPR_SAMPLES.
            n_alpha_samples (int, optional): Sampling parameter for optimal selective function search. Defaults to N_ALPHA_SAMPLES.

        Returns:
            [npt.NDArray[np.float], npt.NDArray[np.float], float]: 1) Risk and 2) TPR arrays and 3) area under the curve.
        """

        if self.fast:
            tpr = np.linspace(0, 1, n_tpr_samples, endpoint=True)
            joint_risk = np.ones_like(tpr)*np.Inf
            for i, min_tpr in tqdm(enumerate(tpr), desc="Optimization, Fast Linear", position=2, leave=False, total=len(tpr)):

                # compensating for the mixture, i.e., dividing by 'param_a', should be done during score export from OpenOOD
                score = 1-self.score_1 - BETA*min_tpr/(1-BETA) * self.score_2
                metric = Metrics(self.y_true, self.y_pred, -score)
                joint_risk[i] = metric.joint_risk[np.argmin(
                    np.abs(metric.tpr - min_tpr))]

            area_under_curve = np.trapz(joint_risk, tpr)

        else:
            tpr = np.linspace(0, 1, n_tpr_samples, endpoint=True)
            joint_risk = np.ones_like(tpr)*np.Inf

            if LIMIT_ALPHA:
                alpha_samples = [i for i in np.linspace(
                    0, np.pi/2, n_alpha_samples, endpoint=True)] + [np.pi/2.]
            else:
                alpha_samples = [i for i in np.linspace(
                    0, 2*np.pi, n_alpha_samples, endpoint=True)] + [np.pi/2., np.pi, np.pi*3./2.]

            for alpha in tqdm(alpha_samples, desc="Optimization, Linear", position=2, leave=False):
                score = np.cos(alpha)*self.score_1 + np.sin(alpha)*self.score_2

                metric = Metrics(self.y_true, self.y_pred, score)
                for i, c in enumerate(tpr):
                    ind = np.argwhere((metric.tpr >= c))
                    joint_risk[i] = min(joint_risk[i], np.min(
                        beta*metric.sel_risk[ind] + (1-beta)*metric.fpr[ind]))

            area_under_curve = np.trapz(joint_risk, tpr)

        return joint_risk, tpr, area_under_curve

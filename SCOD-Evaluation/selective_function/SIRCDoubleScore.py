import numpy as np
import numpy.typing as npt
import warnings
from base.BaseDoubleScore import BaseDoubleScore
from helpers.metrics import Metrics
from helpers.constants import N_TPR_SAMPLES, N_A_SAMPLES, N_B_SAMPLES, BETA, PEDANTIC
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib


def sirc(score_1: npt.NDArray[float],
         score_2: npt.NDArray[float],
         a: float,
         b: float,
         score_1_max: float = 1):
    score_1_diff = score_1_max - score_1 + np.finfo(np.float32).eps
    assert np.all(np.isfinite(score_1_diff))
    assert np.all(~np.isnan(score_1_diff))
    assert np.all(score_1_diff >= 0)
    soft = np.log(score_1_diff)
    additional = np.logaddexp(np.zeros(len(score_2)), -b * (score_2 - a))

    return -soft-additional


class SIRCDoubleScore(BaseDoubleScore):
    def __init__(self,
                 y_true: npt.NDArray[int],
                 y_pred: npt.NDArray[int],
                 score_1: npt.NDArray[float],
                 score_2: npt.NDArray[float],
                 default=False):
        super().__init__(y_true=y_true, y_pred=y_pred, score_1=score_1, score_2=score_2)

        self.a = None
        self.b = None
        self.threshold = None
        self.default = default

    def reset_fit(self):
        self.is_fit = False

    def fit(self, **kwargs):
        if not self.is_fit:
            if not self.default:
                self.fit_impl(self.y_true,
                              self.y_pred,
                              self.score_1,
                              self.score_2,
                              **kwargs)
            else:
                self.fit_default_impl(self.y_true,
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

    def fit_default_impl(self,
                         y_true: npt.NDArray[int],
                         y_pred: npt.NDArray[int],
                         score_1: npt.NDArray[float],
                         score_2: npt.NDArray[float],
                         n_a_samples: int = N_A_SAMPLES,
                         n_b_samples: int = N_B_SAMPLES,
                         optimization_condition: str = 'tpr',
                         optimization_objective: str = 'joint risk',
                         minimize: bool = True,
                         **kwargs):

        mu = np.mean(score_2[y_true != -1])
        std = np.std(score_2[y_true != -1])
        
        # Patch for Residual score
        if std == 0:
            std = 0.001
        
        default_a = mu - 3*std
        default_b = 1/std
        self.a = default_a
        self.b = default_b

        if PEDANTIC:
            assert np.max(score_1) <= 1
            score = sirc(score_1=score_1,
                         score_2=score_2,
                         score_1_max=1,
                         a=self.a,
                         b=self.b)
        else:
            if np.max(score_1) > 1:
                warnings.warn(
                    f"Maximum of score_1 {np.max(score_1)} is larger than 1! Setting SIRC max to the found maximum, but true maximum should be used.", RuntimeWarning)
            score = sirc(score_1=score_1,
                         score_2=score_2,
                         score_1_max=np.max(score_1),
                         a=self.a,
                         b=self.b)

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
            [np.array([np.min(metrics.score)-1]), metrics.score])[select]

        if minimize:
            select_best = np.argmin(potential_values)
            best_value = potential_values[select_best]
            final_threshold = potential_thresholds[select_best]
        else:
            select_best = np.argmax(potential_values)
            best_value = potential_values[select_best]
            final_threshold = potential_thresholds[select_best]

        self.threshold = final_threshold
        self.is_fit = True

    def fit_impl(self,
                 y_true: npt.NDArray[int],
                 y_pred: npt.NDArray[int],
                 score_1: npt.NDArray[float],
                 score_2: npt.NDArray[float],
                 n_a_samples: int = N_A_SAMPLES,
                 n_b_samples: int = N_B_SAMPLES,
                 optimization_condition: str = 'tpr',
                 optimization_objective: str = 'joint risk',
                 minimize: bool = True,
                 **kwargs):

        if minimize:
            best_value = np.inf
        else:
            best_value = -np.inf

        mu = np.mean(score_2[y_true != -1])
        std = np.std(score_2[y_true != -1])
        
        # Patch for Residual score
        if std == 0:
            std = 0.001
        
        default_a = mu - 3*std
        default_b = 1/std
        min_a = default_a - 3*std
        max_a = default_a + 3*std
        min_b = 0.1*default_b
        max_b = 10*default_b

        aL = list(np.linspace(min_a, max_a, num=n_a_samples,
                  endpoint=True)) + [default_a]
        bL = list(np.linspace(min_b, max_b, num=n_b_samples,
                  endpoint=True)) + [default_b]
        av, bv = np.meshgrid(aL, bL)

        final_threshold = None
        final_a = None
        final_b = None
        possible_ab = np.stack((np.ravel(av), np.ravel(bv)), axis=1)

        for ab in tqdm(possible_ab, desc="Optimization, SIRC", position=2, leave=False):
            a, b = ab[0], ab[1]
            if PEDANTIC:
                assert np.max(score_1) <= 1
                score = sirc(score_1=score_1,
                             score_2=score_2,
                             score_1_max=1,
                             a=a,
                             b=b)
            else:
                if np.max(score_1) > 1:
                    warnings.warn(
                        f"Maximum of score_1 {np.max(score_1)} is larger than 1! Setting SIRC max to the found maximum, but true maximum should be used.", RuntimeWarning)
                score = sirc(score_1=score_1,
                             score_2=score_2,
                             score_1_max=np.max(score_1),
                             a=a,
                             b=b)

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
                [np.array([np.min(metrics.score)-1]), metrics.score])[select]

            if potential_thresholds.size == 0:
                continue

            if minimize:
                if np.min(potential_values) <= best_value:
                    select_best = np.argmin(potential_values)
                    best_value = potential_values[select_best]
                    final_threshold = potential_thresholds[select_best]
                    final_a = a
                    final_b = b
            else:
                if np.max(potential_values) >= best_value:
                    select_best = np.argmax(potential_values)
                    best_value = potential_values[select_best]
                    final_threshold = potential_thresholds[select_best]
                    final_a = a
                    final_b = b

        if final_a is not None:
            self.a = final_a
            self.b = final_b
            self.threshold = final_threshold
            self.is_fit = True
        else:
            warnings.warn("Failed to fit.", RuntimeWarning)

    def evaluate(self,
                 y_true: npt.NDArray[int],
                 y_pred: npt.NDArray[int],
                 score_1: npt.NDArray[float],
                 score_2: npt.NDArray[float],
                 beta: float = BETA) -> dict:
        if self.is_fit:

            if PEDANTIC:
                assert np.max(score_1) <= 1
                score = sirc(score_1=score_1,
                             score_2=score_2,
                             score_1_max=1,
                             a=self.a,
                             b=self.b)
            else:
                if np.max(score_1) > 1:
                    warnings.warn(
                        f"Maximum of score_1 {np.max(score_1)} is larger than 1! Setting SIRC max to the found maximum, but true maximum should be used.", RuntimeWarning)
                score = sirc(score_1=score_1,
                             score_2=score_2,
                             score_1_max=np.max(score_1),
                             a=self.a,
                             b=self.b)

            metrics = Metrics(y_true, y_pred, score)
            ix = np.argmin((metrics.score >= self.threshold))
            return {'fpr': metrics.fpr[ix],
                    'tpr': metrics.tpr[ix],
                    'sel_risk': metrics.sel_risk[ix],
                    'joint_risk': beta*metrics.sel_risk[ix] + (1-beta)*metrics.fpr[ix],
                    'beta': beta,
                    'a': self.a,
                    'b': self.b,
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

            if PEDANTIC:
                assert np.max(score_1) <= 1
                score = sirc(score_1=score_1,
                             score_2=score_2,
                             score_1_max=1,
                             a=self.a,
                             b=self.b)
            else:
                if np.max(score_1) > 1:
                    warnings.warn(
                        f"Maximum of score_1 {np.max(score_1)} is larger than 1! Setting SIRC max to the found maximum, but true maximum should be used.", RuntimeWarning)
                score = sirc(score_1=score_1,
                             score_2=score_2,
                             score_1_max=np.max(score_1),
                             a=self.a,
                             b=self.b)

            metrics = Metrics(y_true, y_pred, score)
            ix = np.argmin((metrics.tpr <= at_tpr))
            return {'fpr': metrics.fpr[ix],
                    'tpr': metrics.tpr[ix],
                    'sel_risk': metrics.sel_risk[ix],
                    'joint_risk': beta*metrics.sel_risk[ix] + (1-beta)*metrics.fpr[ix],
                    'beta': beta,
                    'a': self.a,
                    'b': self.b}
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
            score_1 = np.linspace(xmin, xmax, Nx, endpoint=True)
            score_2 = np.linspace(ymin, ymax, Ny, endpoint=True)
            s1, s2 = np.meshgrid(score_1, score_2)

            if PEDANTIC:
                assert np.max(score_1) <= 1
                score = sirc(score_1=s1.ravel(),
                             score_2=s2.ravel(),
                             score_1_max=1,
                             a=self.a,
                             b=self.b)
            else:
                if np.max(s1.ravel()) > 1:
                    warnings.warn(
                        f"Maximum of score_1 {np.max(s1.ravel())} is larger than 1! Setting SIRC max to the found maximum, but true maximum should be used.", RuntimeWarning)
                score = sirc(score_1=s1.ravel(),
                             score_2=s2.ravel(),
                             score_1_max=np.max(s1),
                             a=self.a,
                             b=self.b)

            score = score.reshape(s1.shape)
            cs = axis.contour(s1, s2, score, levels=thresholds, alpha=0.8,
                              colors=color, linestyles='-', label='SIRC', linewidths=3)

            proxy = matplotlib.lines.Line2D([], [], alpha=0.8, color=color, linestyle='-', label='Linear', linewidth=3)
            return proxy

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
                                        n_a_samples: int = N_A_SAMPLES,
                                        n_b_samples: int = N_B_SAMPLES) -> [npt.NDArray[float], npt.NDArray[float], float]:
        """
        Computes the JointRisk-TPR curve. 
        For each possible value of TPR, the optimal (minimizing beta*selective risk + (1-beta)*fpr) selective classifier is found.

        Args:
            beta (float, optional): Parameter defining the metric, i.e., how selective risk and FPR are mixed.
            n_tpr_samples (int, optional): Sampling parameter for the X-axis. Defaults to N_TPR_SAMPLES.

        Returns:
            [npt.NDArray[np.float], npt.NDArray[np.float], float]: 1) Risk and 2) TPR arrays and 3) area under the curve.
        """
        if self.default:
            if self.is_fit:
                if PEDANTIC:
                    assert np.max(self.score_1) <= 1
                    score = sirc(score_1=self.score_1,
                                 score_2=self.score_2,
                                 score_1_max=1,
                                 a=self.a,
                                 b=self.b)
                else:
                    if np.max(self.score_1) > 1:
                        warnings.warn(
                            f"Maximum of score_1 {np.max(self.score_1)} is larger than 1! Setting SIRC max to the found maximum, but true maximum should be used.", RuntimeWarning)
                    score = sirc(score_1=self.score_1,
                                 score_2=self.score_2,
                                 score_1_max=np.max(self.score_1),
                                 a=self.a,
                                 b=self.b)

                metrics = Metrics(y_pred=self.y_pred,
                                  y_true=self.y_true,
                                  score=score,
                                  beta=beta)
                return metrics.joint_risk, metrics.tpr, metrics.AUJR
            else:
                warnings.warn(
                    "Attempting to evaluate model that has not been fit! Fitting parameters instead.", RuntimeWarning)

        tpr = np.linspace(0, 1, n_tpr_samples, endpoint=True)
        joint_risk = np.ones_like(tpr)*np.Inf

        mu = np.mean(self.score_2[self.y_true != -1])
        std = np.std(self.score_2[self.y_true != -1])        
        
        # Patch for Residual score
        if std == 0:
            std = 0.001
            
        default_a = mu - 3*std
        default_b = 1/std
        min_a = default_a - 3*std
        max_a = default_a + 3*std
        min_b = 0.1*default_b
        max_b = 10*default_b

        aL = np.linspace(min_a, max_a, num=n_a_samples, endpoint=True)
        bL = np.linspace(min_b, max_b, num=n_b_samples, endpoint=True)
        av, bv = np.meshgrid(aL, bL)
        possible_ab = np.stack((np.ravel(av), np.ravel(bv)), axis=1)

        for ab in tqdm(possible_ab, desc="Optimization, SIRC", position=2, leave=False):
            a, b = ab[0], ab[1]
            if PEDANTIC:
                assert np.max(self.score_1) <= 1
                score = sirc(score_1=self.score_1,
                             score_2=self.score_2,
                             score_1_max=1,
                             a=a,
                             b=b)
            else:
                if np.max(self.score_1) > 1:
                    warnings.warn(
                        f"Maximum of score_1 {np.max(self.score_1)} is larger than 1! Setting SIRC max to the found maximum, but true maximum should be used.", RuntimeWarning)
                score = sirc(score_1=self.score_1,
                             score_2=self.score_2,
                             score_1_max=np.max(self.score_1),
                             a=a,
                             b=b)

            metrics = Metrics(self.y_true, self.y_pred, score)
            for i, c in enumerate(tpr):
                ind = np.argwhere((metrics.tpr >= c))
                joint_risk[i] = min(joint_risk[i], np.min(
                    beta*metrics.sel_risk[ind] + (1-beta)*metrics.fpr[ind]))

        area_under_curve = np.trapz(joint_risk, tpr)

        return joint_risk, tpr, area_under_curve

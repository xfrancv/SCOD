import numpy as np
import numpy.typing as npt
from helpers.constants import BETA, LOSS


class Metrics:
    """Metrics to evaluate OOD detector given a single uncertainty scores.
    """

    def __init__(self,
                 y_true: npt.NDArray[int],
                 y_pred: npt.NDArray[int],
                 score: npt.NDArray[float],
                 beta: float = BETA,
                 loss: str = LOSS):
        """
        Initializes the object and computes all achievable values of:
            1) selective risk, 
            2) TPR (a.k.a. coverage or recall),
            3) FPR, 
            4) Precision, 
            5) CCR (correct-classification-rate, a.k.a. 1-selective risk for the 0/1 loss), 
            6) areas under curves defined by the above, e.g., AUROC, AUPR, OSCR ...

        Args:
            y_true (npt.NDArray[np.int]): Array of groundtruth labels. For ID samples, the class labels are [0, 1, 2, ...]. The value of -1 corresponds to OOD samples.
            y_pred (npt.NDArray[np.int]): Array of predicted class labels.
            score (npt.NDArray[np.float]): Array of scores obtained by some OOD detection method.
        """
        y_pred[y_true == -1] = -1
        
        # We interpret the score as uncertainty
        
        if loss == '0/1':
            loss = np.array(y_true != y_pred, dtype=np.float16)
        elif loss == 'mae':
            loss = np.array(np.abs(y_true - y_pred), dtype=np.float16)
        else:
            raise NotImplementedError

        # Sort scores and losses according to score
        indices = [i for i in reversed(np.argsort(score, kind='mergesort'))]
        sorted_loss = loss[indices]
        sorted_y_true = y_true[indices]
        sorted_score = score[indices]
        
        assert np.all(np.diff(sorted_score) <= 0), np.diff(sorted_score)[np.diff(sorted_score) > 0]  

        # Number of samples
        n = len(y_true)
        n_in = np.sum(y_true != -1)
        n_out = n-n_in

        # Initialize empty arrays to hold the metric values at different thresholds
        self.sel_risk = np.zeros(n+1)
        self.coverage = np.zeros(n+1)
        self.fpr = np.zeros(n+1)
        self.prec = np.zeros(n+1)
        self.score = score[indices]
        self.y_true = sorted_y_true
        self.loss = sorted_loss
        self.beta = beta
        cur_sel_risk = 0
        cur_coverage = 0
        cur_fpr = 0
        cur_tp = 0

        self.prec[0] = np.NaN
        self.ccr = np.zeros(n+1)
        self.ccr[0] = np.NaN
        cur_ccr = 0

        # Loop over all thresholds (defined by sample scores) and fill in the metric arrays
        for i in range(n):
            if sorted_y_true[i] == -1:
                cur_fpr += 1
            else:
                cur_tp += 1
                cur_coverage += 1
                cur_sel_risk += sorted_loss[i]
                cur_ccr += 1-sorted_loss[i]

            self.coverage[i+1] = cur_coverage / n_in
            self.sel_risk[i+1] = cur_sel_risk / \
                cur_coverage if cur_coverage > 0 else 0.0
            self.ccr[i+1] = cur_ccr / cur_coverage if cur_coverage > 0 else 1.0
            self.fpr[i+1] = cur_fpr / n_out if n_out > 0 else 0.0
            self.prec[i+1] = cur_tp / (i+1)

        self.tpr = self.coverage
        self.recall = self.coverage
        self.joint_risk = beta*self.sel_risk + (1-beta)*self.fpr

        self.AUROC = np.trapz(self.coverage[1:], self.fpr[1:])
        self.AUJR = np.trapz(self.joint_risk[1:], self.coverage[1:])
        self.AUPR = np.trapz(self.prec[1:], self.coverage[1:])
        self.AURF = np.trapz(self.sel_risk[1:], self.fpr[1:])
        self.AURC = np.trapz(self.sel_risk[1:], self.coverage[1:])
        self.OSCR = np.trapz(self.ccr[1:], self.fpr[1:])

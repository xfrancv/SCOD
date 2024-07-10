import os
from selective_function.LinearDoubleScore import LinearDoubleScore
from selective_function.SIRCDoubleScore import SIRCDoubleScore 
from selective_function.MultiplicationDoubleScore import MultiplicationDoubleScore
from helpers.visualization import plot_scores
from helpers.utils import load_result, get_results, get_metrics
from helpers.metrics import Metrics
from helpers.constants import N_A_SAMPLES, N_B_SAMPLES, N_ALPHA_SAMPLES
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

plt.style.use('sequential.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# Specify data location and scores functions to use
scores_root = 'synthetic_example/s0/scores/'
nearfar = 'near'
ood_dataset = 'gauss'
method = f'lhood_ratio'
primary_method = 'cond_risk'

# @TPR
AT_TPR = 0.9

# Load results from OpenOOD
y_pred_1, score_1, y_true_1 = get_results(os.path.join(scores_root, f'{primary_method}.pkl'), nearfar=nearfar, ood_dataset=ood_dataset)
y_pred_2, score_2, y_true_2 = get_results(os.path.join(scores_root, f'{method}.pkl'), nearfar=nearfar, ood_dataset=ood_dataset)

# Make sure that the results were evaluated on the same data
assert np.all(y_true_1==y_true_2)

# Instantiate double-score strategies
# Linear
linear = LinearDoubleScore(y_pred=y_pred_1, y_true=y_true_1, score_1=score_1, score_2=score_2)
linear_plugin = LinearDoubleScore(y_pred=y_pred_1, y_true=y_true_1, score_1=score_1, score_2=score_2, fast=True)
# SIRC
sirc = SIRCDoubleScore(y_pred=y_pred_1, y_true=y_true_1, score_1=score_1, score_2=score_2)
sirc_plugin = SIRCDoubleScore(y_pred=y_pred_1, y_true=y_true_1, score_1=score_1, score_2=score_2, default=True)
# Fit the double-score strategies to the data. For the plugin scores, only the threshold is tuned.
linear.fit(minimize=True, optimization_objective = 'joint risk', optimization_condition='tpr', min_tpr = AT_TPR, n_alpha_samples=N_ALPHA_SAMPLES)
linear_plugin.fit(minimize=True, optimization_objective = 'joint risk', optimization_condition='tpr', min_tpr = AT_TPR)
sirc.fit(minimize=True, optimization_objective = 'joint risk', optimization_condition='tpr', min_tpr = AT_TPR, n_a_samples=N_A_SAMPLES, n_b_samples=N_B_SAMPLES)
sirc_plugin.fit(minimize=True, optimization_objective = 'joint risk', optimization_condition='tpr', min_tpr = AT_TPR)

# Evaluate the double-score strategies
print("Linear")
print(linear.self_evaluate())
print("Linear Plugin")
print(linear_plugin.self_evaluate())
print("SIRC")
print(sirc.self_evaluate())
print("SIRC Plugin")
print(sirc_plugin.self_evaluate())


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Plot the decision boundary of the double-score strategies in 2D (s1, s2) space  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
plt.figure()
proxy, labels = plot_scores(axis=plt.gca(), 
            score_1=score_1,              
            score_2=score_2, 
            scatter_size = 6,
            scatter_alpha = 0.6, 
            ood=y_true_1 == -1, 
            correct=y_pred_1 == y_true_1, 
            contourf_alpha = 0.0,
            bandwidth=[0.15],
            Nx=200,
            Ny=200,
            subsample=3)

# Plot decision boundaries, keep proxy artists            
db_sirc = sirc.plot_decision_boundary(color=colors[1], axis=plt.gca(), thresholds=[sirc.threshold], xmin=np.min(score_1), xmax=np.max(score_1), ymin=max(np.min(score_2), -3), ymax=np.max(score_2))
db_linear = linear.plot_decision_boundary(color='k', axis=plt.gca(), thresholds=[linear.threshold], xmin=np.min(score_1), xmax=np.max(score_1), ymin=max(np.min(score_2), -3), ymax=np.max(score_2))

# Plot also decision boundaries of the single-score strategies using s1 and s2
metrics_1 = linear.get_score_1_metrics()
metrics_2 = linear.get_score_2_metrics()

ix = np.argmin((metrics_1.tpr <= AT_TPR))
risk_1 = metrics_1.AUJR
rk_1 = metrics_1.joint_risk[ix]
plt.axvline(metrics_1.score[ix], alpha=0.8, color=colors[0], linestyle='-', linewidth=3)

ix = np.argmin((metrics_2.tpr <= AT_TPR))
rk_2 = metrics_2.joint_risk[ix]
risk_2 = metrics_2.AUJR
plt.axhline(metrics_2.score[ix], alpha=0.8, color=colors[4], linestyle='-', linewidth=3)

db_metrics_1 = matplotlib.lines.Line2D([], [], alpha=0.8, color=colors[0], linestyle='-', linewidth=3)
db_metrics_2 = matplotlib.lines.Line2D([], [], alpha=0.8, color=colors[4], linestyle='-', linewidth=3)

# Compute area under SCOD RISK - TPR curve for the strategies
sirc_risk, sirc_tpr, sirc_auc = sirc.compute_joint_risk_vs_tpr_curve()
linear_risk, linear_tpr, linear_auc = linear.compute_joint_risk_vs_tpr_curve()
rk_sirc = np.min(sirc_risk[sirc_tpr >= AT_TPR])
rk_lin = np.min(linear_risk[linear_tpr >= AT_TPR])

# Add legend, labels and axis limits
plt.xlim([None, None])
plt.ylim([-3, None])
plt.xlabel("$s_1(x)$")
plt.ylabel("$s_2(x)$")

plt.gca().legend(proxy + [db_linear, db_sirc, db_metrics_1, db_metrics_2], 
           labels + [f'Linear: \t{np.round(100*linear_auc, 2)}/{np.round(100*rk_lin, 2)}',
                     f'SIRC: \t{np.round(100*sirc_auc, 2)}/{np.round(100*rk_sirc, 2)}', 
                     '$\\theta_{s_1(x)}$: \t' + f'{np.round(100*risk_1, 2)}/{np.round(100*rk_1, 2)}', 
                     '$\\theta_{s_2(x)}$: \t' + f'{np.round(100*risk_2, 2)}/{np.round(100*rk_2, 2)}'],
           loc='best')

plt.tight_layout()
plt.savefig("2d-example.pdf", bbox_inches='tight')
plt.show()

# # # # # # # # # # # # # # # # # #
# Plot the SCOD RISK - TPR curve  #
# # # # # # # # # # # # # # # # # #

sirc_risk, sirc_tpr, sirc_auc = sirc.compute_joint_risk_vs_tpr_curve()
sirc_plugin_risk, sirc_plugin_tpr, sirc_plugin_auc = sirc_plugin.compute_joint_risk_vs_tpr_curve()
linear_risk, linear_tpr, linear_auc = linear.compute_joint_risk_vs_tpr_curve()
linear_plugin_risk, linear_plugin_tpr, linear_plugin_auc = linear_plugin.compute_joint_risk_vs_tpr_curve()

plt.figure()
plt.plot(sirc_tpr, sirc_risk, label=f'SIRC {sirc_auc}')
plt.plot(sirc_plugin_tpr, sirc_plugin_risk, label=f'SIRC Plugin {sirc_plugin_auc}')
plt.plot(linear_tpr, linear_risk, label=f'Linear {linear_auc}')
plt.plot(linear_plugin_tpr, linear_plugin_risk, label=f'Linear Plugin {linear_plugin_auc}')

metrics_1 = linear.get_score_1_metrics()
metrics_2 = linear.get_score_2_metrics()

plt.plot(metrics_1.tpr, metrics_1.joint_risk, label=f'{primary_method.upper()} {metrics_1.AUJR}')
plt.plot(metrics_2.tpr, metrics_2.joint_risk, label=f'{method.upper()} {metrics_2.AUJR}')
plt.title("SCOD Risk - TPR curve")
plt.legend(loc='best')
plt.savefig("scodrisk-tpr.pdf", bbox_inches='tight')
plt.show()
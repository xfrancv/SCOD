# Math
import numpy as np
from scipy.stats import multivariate_normal
# Helpers
from tqdm import tqdm
import os
# Type checks
import numpy.typing as npt
# Exporting results
import pickle
# Visualization
import matplotlib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression


plt.style.use('qualitative.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

class_colors = [colors[1], colors[0], colors[4]]

PLOT = True

def get_gaussians():
    """
    Builds gaussian distributions to generate synthetic data from.
    """
    mu_1 = np.array([-8, 0])
    mu_2 = np.array([8, 0])
    mu_3 = np.array([0, -5])

    c_1 = np.array([[3, 2], [2, 3]])*30
    c_2 = np.array([[3, -2], [-2, 3]])*30
    c_3 = np.array([[0.5, 0.1], [0.1, 3]])*60

    return get_gaussian_distributions([mu_1, mu_2, mu_3], [c_1, c_2, c_3])


def get_gaussian_distributions(mus, covars):
    """
    Wrapper around scipy multivariate_normal, returning list of the distributions for a given list of means and covariance matrices.
    """
    return [multivariate_normal(mean=mu, cov=cov) for mu, cov in zip(mus, covars)]


def get_gt_probabilities(x):
    """
    Returns likelihoods of data using the exact gaussian distributions.
    """
    distributions = get_gaussians()
    lpexp = [distrib.pdf(x) for distrib in distributions]
    sm = sum(lpexp)
    return [float(_/sm) for _ in lpexp]


def prepare_dataset(N: int = 100):
    """
    Prepares the training dataset. Returns Class x N samples, N samples of each class.

    Args:
        N (int, optional): Number of samples per class.

    Returns:
        X: ["batch", "Class*N"]; data
        y: ["Class*N"]; labels
    """
    distributions = get_gaussians()
    samples = [distribution.rvs(N) for distribution in distributions]
    labels = [np.array([i]*int(N)) for i in range(len(distributions))]

    X = np.concatenate(samples)
    y = np.concatenate(labels)
    return X, y, distributions


def visualize_gaussian(distribution,
                       N: int = 200,
                       xmin: float = -20,
                       xmax: float = 20,
                       ymin: float = -20,
                       ymax: float = 25,
                       color='k',
                       levels=None):
    X = np.linspace(xmin, xmax, N)
    Y = np.linspace(ymin, ymax, N)
    X, Y = np.meshgrid(X, Y)
    pos = np.dstack((X, Y))
    Z = distribution.pdf(pos)
    ix = np.unravel_index(np.argmax(Z, axis=None), Z.shape)
    plt.contour(X, Y, Z, levels=levels, colors=color,
                cmap=None, alpha=0.5, linewidths=2)


def visualize_decision_boundary(clf,
                                N: int = 1000,
                                xmin: float = -20,
                                xmax: float = 20,
                                ymin: float = -20,
                                ymax: float = 25,
                                levels=None):
    X = np.linspace(xmin, xmax, N)
    Y = np.linspace(ymin, ymax, N)
    X, Y = np.meshgrid(X, Y)
    Z = clf.predict(np.c_[X.ravel(), Y.ravel()])
    Z = Z.reshape(X.shape)
    plt.contourf(X, Y, Z, cmap=matplotlib.colors.ListedColormap(
        [class_colors[0], class_colors[1]]), alpha=0.2, linewidths=0)


def visualize_decision_boundaries(clf,
                                  N: int = 1000,
                                  xmin: float = -20,
                                  xmax: float = 20,
                                  ymin: float = -20,
                                  ymax: float = 25,
                                  levels=None):
    X = np.linspace(xmin, xmax, N)
    Y = np.linspace(ymin, ymax, N)
    X, Y = np.meshgrid(X, Y)
    Z = clf.predict_proba(np.c_[X.ravel(), Y.ravel()])[:, 0]
    Z = Z.reshape(X.shape)
    plt.contour(X, Y, Z, levels=levels, colors='k', alpha=0.2, linewidths=2)


def visualize_optimal_decision_boundary(distributions,
                                        N: int = 1000,
                                        xmin: float = -20,
                                        xmax: float = 20,
                                        ymin: float = -20,
                                        ymax: float = 25):
    X = np.linspace(xmin, xmax, N)
    Y = np.linspace(ymin, ymax, N)
    X, Y = np.meshgrid(X, Y)
    pos = np.dstack((X, Y))
    Zs = [distribution.pdf(pos) for distribution in distributions[:-1]]
    Z = np.array([np.argmax(values)
                 for values in zip(*[_.flatten() for _ in Zs])])
    Z = Z.reshape(X.shape)
    plt.contourf(X, Y, Z, cmap=matplotlib.colors.ListedColormap(
        [class_colors[0], class_colors[1]]), alpha=0.2)


def visualize_optimal_ood_decision_boundary(distributions,
                                            N: int = 1000,
                                            xmin: float = -20,
                                            xmax: float = 20,
                                            ymin: float = -20,
                                            ymax: float = 25):
    X = np.linspace(xmin, xmax, N)
    Y = np.linspace(ymin, ymax, N)
    X, Y = np.meshgrid(X, Y)
    pos = np.dstack((X, Y))
    Zs = [distribution.pdf(pos) for distribution in distributions]
    pos = []
    pis = []
    for values in zip(*[_.flatten() for _ in Zs]):
        pos.append(values[-1])
        pis.append(sum(values[:-1]))

    Z = np.array([po/pi for po, pi in zip(pos, pis)])

    Z = Z.reshape(X.shape)
    plt.contour(X, Y, Z, levels=[0, 1, 100, 1000, 10000], colors='k', alpha=1)


splits = ['s0', 's1', 's2']
seeds = [1, 2, 3]
for split, seed in tqdm(zip(splits, seeds), total=len(splits)):
    np.random.seed(seed)
    
    # Training data
    X, y, distributions = prepare_dataset(N=1000)
    clf = LogisticRegression()
    clf.fit(X[y != 2], y[y != 2])

    # Testing data
    X, y, distributions = prepare_dataset(N=10000)
    Xmin = np.min(X[:, 0])
    Xmax = np.max(X[:, 0])
    Ymin = np.min(X[:, 1])
    Ymax = np.max(X[:, 1])

    if PLOT:
        os.makedirs(f'synthetic_example', exist_ok=True)
        for distribution, color in zip(distributions, class_colors):
            visualize_gaussian(distribution=distribution, color=color,
                               levels=3, xmin=Xmin, xmax=Xmax, ymin=Ymin, ymax=Ymax)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        plt.savefig(f"synthetic_example/{split}_gaussians.png", bbox_inches='tight')
        plt.close()

    if PLOT:
        os.makedirs(f'synthetic_example', exist_ok=True)
        gt_probs = [get_gt_probabilities(x) for x in X]
        y_pred = np.array([np.argmax(probs) for probs in gt_probs])
        cs = [class_colors[pred] for pred in y_pred]

        visualize_decision_boundary(
            clf=clf, xmin=Xmin, xmax=Xmax, ymin=Ymin, ymax=Ymax)

        plt.scatter(X[:, 0], X[:, 1], c=cs, alpha=0.5, linewidths=0)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        plt.savefig(f"synthetic_example/{split}_learned_gaussian_predictor.png",
                    bbox_inches='tight')
        plt.close()

    if PLOT:
        os.makedirs(f'synthetic_example', exist_ok=True)
        y_pred = np.array([np.argmax(probs) for probs in gt_probs])
        cs = [class_colors[pred] for pred in y_pred]

        plt.scatter(X[:, 0], X[:, 1], c=cs, alpha=0.5, linewidths=0)

        visualize_optimal_decision_boundary(
            distributions=distributions, xmin=Xmin, xmax=Xmax, ymin=Ymin, ymax=Ymax)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        plt.savefig(f"synthetic_example/{split}_bayes_gaussian_predictor.png", bbox_inches='tight')
        plt.close()

    if PLOT:
        os.makedirs(f'synthetic_example', exist_ok=True)
        y_pred = np.array([np.argmax(probs) for probs in gt_probs])
        cs = [class_colors[pred] for pred in y_pred]

        plt.scatter(X[:, 0], X[:, 1], c=cs, alpha=0.5, linewidths=0)

        visualize_optimal_ood_decision_boundary(
            distributions=distributions, xmin=Xmin, xmax=Xmax, ymin=Ymin, ymax=Ymax)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        plt.savefig(f"synthetic_example/{split}_bayes_ood_detector.png", bbox_inches='tight')
        plt.close()

    y_pred = np.argmax(clf.predict_proba(X), axis=1)
    score_msp = np.max(clf.predict_proba(X), axis=1)

    distribution_probas = [distribution.pdf(
        X) for distribution in distributions]
    score_conditional_risk = []
    for probas in zip(*distribution_probas[:-1]):
        probas = np.array(probas) / np.sum(probas)
        score_conditional_risk.append(np.max(probas))

    score_conditional_risk = np.array(score_conditional_risk)

    distribution_probas = [distribution.pdf(
        X) for distribution in distributions]
    score_likelihood_ratio = []
    for probas in zip(*distribution_probas):
        probas = np.array(probas)
        score_likelihood_ratio.append(probas[-1] / np.sum(probas[:-1]))
    score_likelihood_ratio = np.array(score_likelihood_ratio)
    score_likelihood_ratio = -score_likelihood_ratio

    distribution_probas = [distribution.pdf(
        X) for distribution in distributions]
    score_likelihood_ID = []
    for probas in zip(*distribution_probas):
        probas = np.array(probas)
        score_likelihood_ID.append(np.sum(probas[:-1]))
    score_likelihood_ID = np.array(score_likelihood_ID)

    y_true = np.argmax(np.stack([distribution.pdf(X)
                       for distribution in distributions]), axis=0)
    y_true[y_true == 2] = -1

    result_1 = {'id': {'test': (y_pred[y_true != -1],
                                score_msp[y_true != -1],
                                y_true[y_true != -1])},
                'ood': {'near': {'gauss': (y_pred[y_true == -1],
                                           score_msp[y_true == -1],
                                           y_true[y_true == -1])
                                 },
                        'far': {}}}

    result_2 = {'id': {'test': (y_pred[y_true != -1],
                                score_conditional_risk[y_true != -1],
                                y_true[y_true != -1])},
                'ood': {'near': {'gauss': (y_pred[y_true == -1],
                                           score_conditional_risk[y_true == -1],
                                           y_true[y_true == -1])
                                 },
                        'far': {}}}

    result_3 = {'id': {'test': (y_pred[y_true != -1],
                                score_likelihood_ratio[y_true != -1],
                                y_true[y_true != -1])},
                'ood': {'near': {'gauss': (y_pred[y_true == -1],
                                           score_likelihood_ratio[y_true == -1],
                                           y_true[y_true == -1])
                                 },
                        'far': {}}}

    result_4 = {'id': {'test': (y_pred[y_true != -1],
                                score_likelihood_ID[y_true != -1],
                                y_true[y_true != -1])},
                'ood': {'near': {'gauss': (y_pred[y_true == -1],
                                           score_likelihood_ID[y_true == -1],
                                           y_true[y_true == -1])
                                 },
                        'far': {}}}

    os.makedirs(f'synthetic_example/{split}', exist_ok=True)
    os.makedirs(f'synthetic_example/{split}/scores/', exist_ok=True)
    
    with open(f'synthetic_example/{split}/scores/msp.pkl', 'wb') as handle:
        pickle.dump(result_1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'synthetic_example/{split}/scores/cond_risk.pkl', 'wb') as handle:
        pickle.dump(result_2, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'synthetic_example/{split}/scores/lhood_ratio.pkl', 'wb') as handle:
        pickle.dump(result_3, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'synthetic_example/{split}/scores/lhood_id.pkl', 'wb') as handle:
        pickle.dump(result_4, handle, protocol=pickle.HIGHEST_PROTOCOL)

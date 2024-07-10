import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from helpers.metrics import Metrics
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm


def plot_scores(axis: plt.Axes,
                score_1: npt.NDArray[float],
                score_2: npt.NDArray[float],
                ood: npt.NDArray[bool],
                correct: npt.NDArray[bool] = None,
                Nx: int = 20,
                Ny: int = 20,
                bandwidth: float = 0.05,
                contourf_alpha: float = 0.0,
                countour_levels: int = 5,
                contour_alpha: float = 0.9,
                scatter_alpha: float = 0.2,
                scatter_size: float = 2,
                scatter_color_intensity: float = 150,
                linewidths: float = 2,
                normalize: bool = False,
                subsample=1):
    """
    Estimates density of ID/OOD data and visualizes it.

    Args:
        score_1 (npt.NDArray[float]): Array of score from some method.
        score_2 (npt.NDArray[float]): Array of score from some other method.
        ood (npt.NDArray[bool]): Boolean array indicating whether samples are OOD (True) or ID (False).
        axis (plt.Axes): Axis to plot into.
        correct (npt.NDArray[bool], optional): If not None, is expected to be boolean array indicating whether prediction on ID data is correct. Defaults to None.
        Nx (int, optional): Density X axis grid. Defaults to 30.
        Ny (int, optional): Density Y axis grid. Defaults to 30.
        bandwidth (float, optional): Density estimation bandwidth. Defaults to 0.03.
        contourf_alpha (float, optional): Transparency of filled contour plot of density. Defaults to 0.5.
        countour_levels (int, optional): Number of levels of contour to show. Defaults to 10.
        scatter_alpha (float, optional): Transparency of data samples. Defaults to 0.4.
        scatter_size (float, optional): Size of data samples. Defaults to 0.7.
        scatter_color_intensity (float, optional): Color (from colormap) of data samples. Defaults to 0.7.
        linewidths (float, optional): Width of contour and scatter lines. Defaults to 0.7.
        normalize (bool, optional): Normalize scores before visualization. Defaults to False.

    """

    def clamp(arr):
        pseudomax = 10*np.mean(np.abs(arr))
        return np.clip(arr, a_min=-pseudomax, a_max=pseudomax)

    def normalization(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx > mn:
            return (arr - mn) / (mx - mn)
        else:
            return (arr-mn)

    if normalize:
        score_2, score_1 = normalization(score_2), normalization(score_1)
        
    score_1, score_2 = clamp(score_1), clamp(score_2)

    Xtrain = np.column_stack([score_2, score_1])
    xi = np.linspace(np.min(score_1), np.max(score_1), Nx)
    yi = np.linspace(np.min(score_2), np.max(score_2), Ny)

    Xi, Yi = np.meshgrid(xi, yi)
    XY = np.vstack([Yi.ravel(), Xi.ravel()]).T

    if contourf_alpha == 0.0:
        # do not plot the density
        pass
    else:
        if type(bandwidth) is list:
            grid = GridSearchCV(KernelDensity(metric="euclidean",
                                            kernel="exponential"),
                                {'bandwidth': bandwidth}, cv=5)
            grid.fit(Xtrain[ood])
            kde = grid.best_estimator_
        else:
            kde = KernelDensity(
                bandwidth=bandwidth,
                metric="euclidean",
                kernel="exponential"
            )

        # Fit to OOD samples
        kde.fit(Xtrain[ood])
        Zi = np.exp(kde.score_samples(XY)).reshape(Xi.shape)

        # Plot contours of the density
        levels = np.linspace(0, Zi.max(), countour_levels)
        axis.contourf(Xi, Yi, Zi,
                    levels=levels,
                    cmap=plt.cm.Reds,
                    alpha=contourf_alpha,
                    extend='both')
        CS1 = axis.contour(Xi, Yi, Zi,
                        levels=levels,
                        linewidths=linewidths,
                        cmap=plt.cm.Reds,
                        alpha=contour_alpha)

        if type(bandwidth) is list:
            grid = GridSearchCV(KernelDensity(metric="euclidean",
                                            kernel="exponential"),
                                {'bandwidth': bandwidth}, cv=5)
            grid.fit(Xtrain[~ood & correct])
            kde = grid.best_estimator_
        else:
            kde = KernelDensity(
                bandwidth=bandwidth,
                metric="euclidean",
                kernel="exponential"
            )

        # Fit to correctly classified ID samples
        kde.fit(Xtrain[~ood & correct])
        Zi = np.exp(kde.score_samples(XY)).reshape(Xi.shape)

        # Plot contours of the density
        levels = np.linspace(0, Zi.max(), countour_levels)
        axis.contourf(Xi, Yi, Zi,
                    levels=levels,
                    cmap=plt.cm.Blues,
                    alpha=contourf_alpha,
                    extend='both')
        CS1 = axis.contour(Xi, Yi, Zi,
                        levels=levels,
                        linewidths=linewidths,
                        cmap=plt.cm.Blues,
                        alpha=contour_alpha)

        if type(bandwidth) is list:
            grid = GridSearchCV(KernelDensity(metric="euclidean",
                                            kernel="exponential"),
                                {'bandwidth': bandwidth}, cv=5)
            grid.fit(Xtrain[~ood & ~correct])
            kde = grid.best_estimator_
        else:
            kde = KernelDensity(
                bandwidth=bandwidth,
                metric="euclidean",
                kernel="exponential"
            )

        # Fit to incorrectly classified ID samples
        kde.fit(Xtrain[~ood & ~correct])
        Zi = np.exp(kde.score_samples(XY)).reshape(Xi.shape)

        # Plot contours of the density
        levels = np.linspace(0, Zi.max(), countour_levels)
        axis.contourf(Xi, Yi, Zi,
                    levels=levels,
                    cmap=plt.cm.Greens,
                    alpha=contourf_alpha,
                    extend='both')
        CS1 = axis.contour(Xi, Yi, Zi,
                        levels=levels,
                        linewidths=linewidths,
                        cmap=plt.cm.Greens,
                        alpha=contour_alpha)

    # Scatter data points
    # We do the scatter after plotting contours. Otherwise, the contours cover up the points
    axis.scatter(score_1[ood][::subsample], score_2[ood][::subsample],
                 alpha=scatter_alpha,   
                 s=scatter_size,
                 marker='.',
                 color=matplotlib.colormaps.get_cmap(
                     'Reds')(scatter_color_intensity),
                 linewidths=0)
    
    p1 = axis.scatter([], [],
                 alpha=0.8,
                 s=scatter_size*20,
                 marker='.',
                 color=matplotlib.colormaps.get_cmap(
                     'Reds')(scatter_color_intensity),
                 linewidths=0)

    axis.scatter(score_1[~ood & correct][::subsample], score_2[~ood & correct][::subsample],
                 alpha=scatter_alpha,
                 marker='.',
                 s=scatter_size,
                 color=matplotlib.colormaps.get_cmap(
                     'Blues')(scatter_color_intensity),
                 linewidths=0)
    
    p2 = axis.scatter([], [],
                 alpha=0.8,
                 s=scatter_size*20,
                 marker='.',
                 color=matplotlib.colormaps.get_cmap(
                     'Blues')(scatter_color_intensity),
                 linewidths=0)
    
    axis.scatter(score_1[~ood & ~correct][::subsample], score_2[~ood & ~correct][::subsample],
                 alpha=scatter_alpha,
                 marker='.',
                 s=scatter_size,
                 color=matplotlib.colormaps.get_cmap(
                     'Greens')(scatter_color_intensity),
                 linewidths=0)

    p3 = axis.scatter([], [],
                 alpha=0.8,
                 s=scatter_size*20,
                 marker='.',
                 color=matplotlib.colormaps.get_cmap(
                     'Greens')(scatter_color_intensity),
                 linewidths=0)


    axis.set_xlim([np.min(score_1), np.max(score_1)])
    axis.set_ylim([np.min(score_2), np.max(score_2)])

    """
    proxy = [plt.Circle((0, 0), 1, fc=cmap(160), alpha=scatter_alpha) for cmap in [matplotlib.colormaps.get_cmap('Reds'),
                                                                    matplotlib.colormaps.get_cmap(
                                                                        'Blues'),
                                                                    matplotlib.colormaps.get_cmap('Greens')]]
    """
    proxy = [p1, p2, p3]
        
    return proxy, ["OOD", "\ding{51} ID", "\ding{55} ID"]

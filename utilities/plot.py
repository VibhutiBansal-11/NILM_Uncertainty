import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import jax.numpy as jnp
from sklearn.metrics import brier_score_loss
from probml_utils import is_latexify_enabled

def plot_actualdata(X, y, x_test, y_test):
    plt.scatter(X, y, color="black", label="Train Data")
    plt.scatter(x_test, y_test, color="blue", label="Test Data")
    plt.xlabel("$x$" )
    plt.ylabel("$y$")
    plt.legend()
    sns.despine()


def calibration_regression(mean, sigma, Y,label, color, ax=None):
    """
    mean : (n_samples,1) or (n_sample,) prediction mean 
    sigma : (n_samples,1) or (n_sample,) prediction sigma 
    Y : (n_samples,1) or (n_sample,) Y co-ordinate of ground truth 
    label :  string, 
    
    
    """

    marker_size = 6 if is_latexify_enabled else None
    if ax is None:
        fig, ax = plt.subplots()
    df = pd.DataFrame()
    df["mean"] = mean
    df["sigma"] = sigma
    df["Y"] = Y
    df["z"] = (df["Y"] - df["mean"]) / df["sigma"]
    df["perc"] = st.norm.cdf(df["z"])
    k = jnp.arange(0, 1.1, 0.1)
    counts = []
    df2 = pd.DataFrame()
    df2["Interval"] = k
    df2["Ideal"] = k
    for i in range(0, 11):
        l = df[df["perc"] < 0.5 + i * 0.05]
        l = l[l["perc"] >= 0.5 - i * 0.05]
        counts.append(len(l) / len(df))
    df2["Counts"] = counts

    ax.plot(k, counts, color=color, label=label)

    ax.scatter(k, counts, color=color,s=marker_size)
    ax.scatter(k, k,color="green",s=marker_size)
    ax.set_yticks(k)
    ax.set_xticks(k)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    # ax.legend()
    ax.set_xlabel("decile")
    ax.set_ylabel("ratio of points")
    ax.plot(k, k, color="green")
    sns.despine()
    return df, df2


def plot_prediction_reg(
    X_train,
    Y_train,
    x_test,
    y_test,
    X_linspace,
    predict_mean,
    predict_sigma,
    title,
    y_min=None,
    y_max=None,
    marker_size=None,
    ax=None,

):
    """
    plots the prediction in 1d case.
    X_train: (n_samples,1) or (n_sample,) X coordinates of the training points
    Y_train: (n_sample,1) or (n_samples,) True Y coordinates of the training points
    X_test: (n_samples,1) or (n_sample,) X coordinates of the given test points
    Y_test: (n_samples,1) or (n_sample,) True Y coordinates of given test points
    X_linspace: (n_points,) X coordinates used for predictions
    predict_mean: (n_points,) mean of predicted values over X_linspace
    predict_sigma: (n_points,) variance of predicted values over X_linspace
    title: title of the plot
    """
    if marker_size ==None:
        
        marker_size  = 4 if is_latexify_enabled() else None
    if ax == None:
        fig, ax = plt.subplots(1)
    ax.plot(X_linspace, predict_mean, color="red")
    for i_std in range(1, 4):
        ax.fill_between(
            X_linspace.reshape((-1,)),
            jnp.array((predict_mean - i_std * predict_sigma)),
            jnp.array((predict_mean + i_std * predict_sigma)),
            color="lightsalmon",
            alpha=3/ (4 * i_std),
            label=f"$\mu\pm{i_std}\sigma$",
        )

    ax.scatter(x_test, y_test, color="blue", alpha=0.5,s=marker_size,label='Test')
    ax.scatter(X_train, Y_train, color="black", alpha=0.5,s=marker_size,label='Train')
    # ax.vlines(
    #     min(X_train),
    #     min(min(y_test), min(predict_mean - 3 * predict_sigma)),
    #     max(max(y_test), max(predict_mean + 3 * predict_sigma)),
    #     colors="black",
    #     linestyles="--",
    # )
    # ax.vlines(
    #     max(X_train),
    #     min(min(y_test), min(predict_mean - 3 * predict_sigma)),
    #     max(max(y_test), max(predict_mean + 3 * predict_sigma)),
    #     colors="black",
    #     linestyles="--",
    # )
    ax.vlines(min(X_train),y_min,y_max,  colors="black",
        linestyles="--",)
    ax.vlines(max(X_train),y_min,y_max,  colors="black",
        linestyles="--",)
    ax.set_ylim([y_min,y_max])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    # ax.set_ylim(
    #     [
    #         min(min(y_test), min(predict_mean - 3 * predict_sigma)),
    #         max(max(y_test), max(predict_mean + 3 * predict_sigma)),
    #     ]
    # )
    ax.set_xlim([min(x_test), max(x_test)])
    # ax.set_ylim(-3,3)
    ax.set_title(title)
    # ax.legend()
    sns.despine()
    return ax


def plot_binary_class(
    X_scatters,
    y_scatters,
    XX1_grid,
    XX2_grid,
    grid_preds_mean,
    grid_preds_sigma,
    titles: tuple,
):
    """
  funtion to binary classificaton outputs

  X: points shape=(n_samples,2)
  y_hat: predictions for X shape=(n_samples,)
  XX1,XX2: grid outputs shape=(n_points,n_points)
  Z: mean of the predictions shape = (n_points,n_points)
  sigma_Z: variance of the predictions shape= (n_points,n_points) 
  titles: tuple with title of the two images. 
  """

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))

    ax[0].set_title(titles[0], fontsize=16)
    ax[0].contourf(XX1_grid, XX2_grid, grid_preds_mean, cmap="coolwarm", alpha=0.8)
    hs = ax[0].scatter(X_scatters.T[0], X_scatters.T[1], c=y_scatters, cmap="bwr")
    # *hs is similar to hs[0],hs[1]
    ax[0].legend(*hs.legend_elements(), fontsize=20)

    ax[1].set_title(titles[1], fontsize=16)
    CS = ax[1].contourf(XX1_grid, XX2_grid, grid_preds_sigma, cmap="viridis", alpha=0.8)
    hs = ax[1].scatter(X_scatters.T[0], X_scatters.T[1], c=y_scatters, cmap="bwr")
    # ax[1].legend(*hs.legend_elements(), fontsize=20)
    fig.colorbar(CS, ax=ax[1])
    sns.despine()


def plot_scatter_predictions(x, y_true, y_test, ax=None):
    if ax == None:
        fig, ax = plt.subplots(1, figsize=(10, 6))
    hs = ax.scatter(x[:, 0], x[:, 1], c=y_test, cmap="seismic")
    ax.set_title(f"Train Brier Loss {brier_score_loss(y_true,y_test)}")
    ax.legend(*hs.legend_elements())
    sns.despine()

def plot_prediction_regression_without_test(x_train,y_train,x_linspace_test,mean,sigma,y_min=None,y_max=None,title='',marker_size=None):
    if marker_size ==None:
        
        marker_size  = 4 if is_latexify_enabled() else None
    fig,ax=plt.subplots(1)
    ax.vlines(min(x_train),y_min,y_max,  colors="black",
        linestyles="--",)
    ax.vlines(max(x_train),y_min,y_max,  colors="black",
        linestyles="--",)
    ax.set_ylim([y_min,y_max])
    ax.scatter(x_train, y_train, color="black",s=marker_size)
    ax.plot(x_linspace_test, mean, "red", linewidth=2)
    for i in range(1,4):
        plt.fill_between(x_linspace_test.reshape(-1), mean - i*sigma, mean + i*sigma, 
        color="crimson", alpha = 1/(i*3),  label =  f"$\mu\pm{i}\sigma$")
    # ax.vlines(
    #     min(x_train),
    #     min(min(y_train), min(mean - 3 * sigma)),
    #     max(max(y_train), max(mean + 3 * sigma)),
    #     colors="black",
    #     linestyles="--",
    # )
    # ax.vlines(
    #     max(x_train),
    #     min(min(y_train), min(mean - 3 * sigma)),
    #     max(max(y_train), max(mean + 3 * sigma)),
    #     colors="black",
    #     linestyles="--",
    # )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    # ax.set_ylim(
    #     [
    #         min(min(y), min(predict_mean - 3 * predict_sigma)),
    #         max(max(y_test), max(predict_mean + 3 * predict_sigma)),
    #     ]
    # )
    # ax.set_xlim([min(x_test), max(x_test)])
    # ax.set_ylim(-3,3)
    ax.set_title(title)
    # ax.legend()
    sns.despine()
    plt.xlabel("X")
    plt.ylabel("y")
    sns.despine()
    return ax



# def plot_train_test(X_train,X_test,y_pred_train,y_pred_test,y_train,y_test):
#     """
#     predicts using the given model and parameters on training and testing data and plots side by side.
#     X_train: Training points
#     X_test: Testing points
#     y_pred_train: predicted Y labels for training data
#     y_pred_test: predicted Y labels for test data
#     y_train: true y labels of training points
#     y_test: true y labels of testing points
#     """
#     fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
#     ax1.scatter(X_train[:,0],X_train[:,1],c=y_pred_train,cmap='seismic')
#     ax1.set_title(f'Train Brier Loss {brier_score_loss(y_train,y_pred_train)}')
#     ax2.scatter(X_test[:,0],X_test[:,1],c=y_pred_test,cmap='seismic')
#     ax2.set_title(f'Test Brier Loss {brier_score_loss(y_test,y_pred_test)}')
#     # ax2.set_title(brier_score_loss(y_test,y_pred_test))
#     sns.despine()
def plot(idx,y_test,mean,sigma):
    # idx = idx
    
    fig,ax = plt.subplots()
    ax.plot(jnp.arange(idx), y_test[:idx], label = "Actual")
    ax.plot(jnp.arange(idx), mean[:idx], label = "Predicted",color='black')
    for i in range(1,4):
        plt.fill_between(jnp.arange(idx), mean[:idx] - i*sigma[:idx], mean[:idx] + i*sigma[:idx],
                    color="red", alpha=(1/(i*3)), label=f"$\mu\pm{i}*\sigma$")
    # ax.legend()
    # ax.despine()
    sns.despine()
    return ax
    # plt.savefig("seq2point.pdf", bbox_inches="tight")
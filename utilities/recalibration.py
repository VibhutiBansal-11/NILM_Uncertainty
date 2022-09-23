import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd

def mass_to_std_factor(mass=0.95):
    rv = norm(0.0, 1.0)
    std_factor = rv.ppf((1.0 + mass) / 2)
    return std_factor

def plot_find_p(y,mean_prediction, std_prediction, mass=0.95):
    std_factor = mass_to_std_factor(mass)
    idx = np.where(
        (y < mean_prediction + std_factor * std_prediction)
        & (y > mean_prediction - std_factor * std_prediction)
    )[0]

    p_hat = len(idx) / len(y)
    return (mass, p_hat)

def find_p_hat(y, mean_prediction,std_prediction):
    out = {}
    for mass in np.linspace(1e-10, 1-1e-20, 1000):
    #for mass in jnp.arange(0, 1.1, 0.1):
        out[mass] = plot_find_p(y, mean_prediction,std_prediction, mass)[1]
    df = pd.Series(out).to_frame()
    df.index.name = 'p'
    df.columns = ['p_hat']
    
    return df
def find_p_hat_(y, mean_prediction,std_prediction):
    out = {}
    for mass in np.linspace(1e-10, 1-1e-20, 1000):
    #for mass in jnp.arange(0, 1.1, 0.1):
        out[mass] = plot_find_p(y, mean_prediction,std_prediction, mass)[1]
    df = pd.Series(out).to_frame()
    df.index.name = 'p'
    df.columns = ['p_hat']
    df=df.reset_index()
    return df
def find_new_p_hat(y,new_p,mean,sigma):
    new_p_hat=[]
    for i in range(len(new_p)):
        new_p_hat.append(plot_find_p(y,mean,sigma,new_p[i])[1])
    return np.array(new_p_hat).reshape(-1,1)
    
def isotonic_regression_fit(iso_reg,p,p_hat):
    iso_reg.fit(p_hat, p)
    return iso_reg



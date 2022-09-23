import tensorflow_probability.substrates.jax as tfp
import jax.numpy as jnp
import jax
dist = tfp.distributions
# def loss(mean,sigma,y):
#     """
#     mean : (n_samples,1) or (n_sample,) prediction mean 
#     sigma : (n_samples,1) or (n_sample,) prediction sigma 
#     y : (n_samples,1) or (n_sample,) Y co-ordinate of ground truth 
#     """
#     def loss_fn(mean, sigma, y):
#         d = dist.Normal(loc=mean, scale=sigma)
#         return -d.log_prob(y)
#     return jnp.mean(jax.vmap(loss_fn, in_axes=(0, 0, 0))(mean, sigma, y))



def rmse(y,yhat):
  def rmse_loss(y,yhat):
      return (y-yhat)**2
  return jnp.sqrt(jnp.mean(jax.vmap(rmse_loss,in_axes=(0,0))(y,yhat)))

def ace(dataframe):
    """
    dataframe : pandas dataframe with Ideal and Counts as column for regression calibration
    It can be directly used as 2nd output from calibration_regression in plot.py 
    """
    def rmse_loss(y,yhat):
      return jnp.abs(y-yhat)
    return jnp.mean(jax.vmap(rmse_loss,in_axes=(0,0))(dataframe['Ideal'].values,dataframe['Counts'].values))
    # return(jnp.sum(jnp.abs(dataframe['Ideal'].values-dataframe['Counts'].values)))
def NLL(mean,sigma,y):
    def loss_fn(mean, sigma, y):
      d = dist.Normal(loc=mean, scale=sigma)
      return -d.log_prob(y)
    return jnp.mean(jax.vmap(loss_fn, in_axes=(0, 0, 0))(mean, sigma, y))
def mae(y,yhat):
  def mae_loss(y,yhat):
      return jnp.abs(y-yhat)
  return jnp.mean(jax.vmap(mae_loss,in_axes=(0,0))(y,yhat))
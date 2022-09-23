import jax.numpy as jnp


def gmm_mean_var(means_stack, sigmas_stack):
    means = jnp.stack(means_stack)
    final_mean = means.mean(axis=0)
    sigmas = jnp.stack(sigmas_stack)
    final_sigma = jnp.sqrt((sigmas**2 + means ** 2).mean(axis=0) - final_mean ** 2)
    return final_mean, final_sigma

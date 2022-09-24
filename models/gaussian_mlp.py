from flax import linen as nn
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

dist = tfp.distributions
class gmlp(nn.Module):
    features: list
    activations: list
    dropout_rate: list

    @nn.compact
    def __call__(self, X, deterministic):
        if len(self.activations) != len(self.features):
            raise Exception(
                f"Length of activations should be equal to {len(self.features)}"
            )
        if len(self.dropout_rate) != len(self.features):
            raise Exception(
                f"Length of dropout rates should be equal to {len(self.features)}"
            )
        for i, feature in enumerate(self.features):
            X = nn.Dense(
                feature,
                kernel_init=jax.nn.initializers.glorot_normal(),
                name=f"{i}_Dense",
            )(X)
            X = self.activations[i](X)
            X = nn.Dropout(
                rate=self.dropout_rate[i],
                deterministic=deterministic,
                name=f"{i}_Dropout_{self.dropout_rate[i]}",
            )(X)
        X = nn.Dense(2, name=f"Gaussian")(X)
        mean = X[:, 0]
        sigma = nn.softplus(X[:, 1])
        return mean, sigma

    def loss_fn(self, params, X, y, deterministic=False, rng=jax.random.PRNGKey(0)):
        mean, sigma = self.apply(
            params, X, deterministic=deterministic, rngs={"dropout": rng}
        )

        def loss(mean, sigma, y):
            d = dist.Normal(loc=mean, scale=sigma)
            return -d.log_prob(y)

        return jnp.mean(jax.vmap(loss, in_axes=(0, 0, 0))(mean, sigma, y))

from flax import linen as nn
import jax
import jax.numpy as jnp
import optax


class cmlp(nn.Module):
    features: list
    activations: list
    dropout_rate: list

    @nn.compact
    def __call__(self, X, deterministic):
        if len(self.activations) != len(self.features) - 1:
            raise Exception(
                f"Length of activations should be equal to {len(self.features) - 1}"
            )

        if len(self.dropout_rate) != len(self.features) - 1:
            raise Exception(
                f"Length of dropout_rate should be equal to {len(self.features) - 1}"
            )

        for i, feature in enumerate(self.features):
            X = nn.Dense(feature, name=f"{i}_Dense")(X)
            if i != len(self.features) - 1:
                X = self.activations[i](X)
                X = nn.Dropout(
                    rate=self.dropout_rate[i],
                    deterministic=deterministic,
                    name=f"{i}_Dropout_{self.dropout_rate[i]}",
                )(X)
        return nn.sigmoid(X)

    def loss_fn(self, params, X, y, deterministic=False, rng=jax.random.PRNGKey(0)):
        y_pred = self.apply(params, X, deterministic=False, rngs={"dropout": rng})
        cost0 = jnp.dot(y.T, jnp.log(y_pred + 1e-7))
        cost1 = jnp.dot((1 - y).T, jnp.log(1 - y_pred + 1e-7))
        loss = (cost0 + cost1) / len(X)
        return -loss.squeeze()

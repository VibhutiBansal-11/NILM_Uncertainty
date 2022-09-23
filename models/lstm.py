import flax.linen as nn
import jax
import jax.numpy as jnp

class AttentionLayer():
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W = nn.Dense(units)
        self.V = nn.Dense(1)

    def __call__(self, encoder_output, **kwargs):
        score = self.V(jnp.tanh(self.W(encoder_output)))
        attention_weights = nn.softmax(score, axis=1)

        context_vector = attention_weights * encoder_output
        context_vector = jnp.sum(context_vector, axis=1)
        return context_vector

class lstm_mlp(nn.Module):
    def setup(self):
        LSTMLayer = nn.scan(nn.LSTMCell,
                               variable_broadcast="params",
                               split_rngs={"params": False},
                               in_axes=1, out_axes=1,
                               reverse=False)
        self.lstm1 = LSTMLayer()
        self.lstm2 = LSTMLayer()
        self.lstm3 = LSTMLayer()
        self.lstm4 = LSTMLayer()

    @nn.compact
    def __call__(self, X, deterministic):
        X = nn.Conv(16, kernel_size=(4,))(X)

        X_flip = jnp.flip(X, axis=-1)
        c1 = nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (X.shape[0], ), 128)
        c1, X1 = self.lstm1(c1, X)
        c2 = nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (X_flip.shape[0], ), 128)
        c2, X2 = self.lstm2(c2, X_flip)
        
        X = jnp.concatenate([X1, X2], axis=-1)
        
        X_flip = jnp.flip(X, axis=-1)
        c3 = nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (X.shape[0], ), 256)
        c3, X1 = self.lstm3(c3, X)
        c4 = nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (X_flip.shape[0], ), 256)
        c4, X2 = self.lstm4(c4, X_flip)

        X = jnp.concatenate([X1, X2], axis=-1)

        X = AttentionLayer(units=128)(X)
        X = nn.Dense(128)(X)
        X = nn.tanh(X)
        X = nn.Dropout(0.2, deterministic=deterministic)(X)
        X = nn.Dense(1)(X)
        return X

    def loss_fn(self, params, X, y, deterministic=False, rng=jax.random.PRNGKey(0)):
        y_pred = self.apply(
            params, X, deterministic=deterministic, rngs={"dropout": rng}
        )
        loss = jnp.mean((y - y_pred) ** 2)
        return loss
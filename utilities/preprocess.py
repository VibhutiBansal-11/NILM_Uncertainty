import jax.numpy as jnp
from sklearn.preprocessing import StandardScaler
def call_preprocessing(x, y, scaler_x = None, scaler_y = None, method="train"):
    sequence_length = 99
    if method == "train":
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
    elif scaler_x == None or scaler_y == None:
        raise "For testing part, scaler_x, scaler_y requires."
        
    n = sequence_length
    units_to_pad = n // 2
    x = jnp.pad(x, (units_to_pad, units_to_pad), 'constant', constant_values = (0,0))
    x = jnp.array([x[i: i + n] for i in range(len(x) - n + 1)])
    if method == "train":
        x = scaler_x.fit_transform(x)
        y = scaler_y.fit_transform(y.reshape(-1,1))
        return jnp.array(x), jnp.array(y), scaler_x, scaler_y
    else:
        x = scaler_x.transform(x)
        
    return jnp.array(x)
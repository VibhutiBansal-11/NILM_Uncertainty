from sklearn.preprocessing import StandardScaler
import numpy as np
def generate(dataset, seq=99):
    m = len(dataset)
    seq_Xlist = []
    seq_ylist = []
    half_seq_size = 99//2
#     pad = jnp.zeros(half_seq_size)
    ref = dataset["Refrigerator"].values
    washer = dataset["Washer Dryer"].values
    
    for i in range(len(dataset) - half_seq_size):
        if i < half_seq_size:
            pad = np.zeros((half_seq_size - i,1))
            arr = np.array(dataset["main"][:i + half_seq_size + 1])
            scaler = StandardScaler()
            arr = scaler.fit_transform(arr.reshape(-1,1))
            seq_Xlist.append(np.concatenate([pad,arr], axis=0))
        else:
            arr = dataset["main"][i - half_seq_size : i + half_seq_size + 1].values
            scaler = StandardScaler()
            arr = scaler.fit_transform(arr.reshape(-1,1))
            seq_Xlist.append(arr)
        seq_ylist.append([ref[i], washer[i]])

    return np.array(seq_Xlist), np.array(seq_ylist)
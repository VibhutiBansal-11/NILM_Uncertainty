import pandas as pd
import jax.numpy as jnp
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def dataset_load(appliances, train, test=None,n=19,split_factor=0.3):
    x_train = []
    y_train = []
    units_to_pad = n // 2
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    #train
    for key, values in train.items():
        df = pd.read_csv(f"datasets/Building{key}_NILM_data_basic.csv", usecols=["Timestamp","main", appliances[0]])
        df["date"] = pd.to_datetime(df["Timestamp"]).dt.date
        startDate = datetime.strptime(values["start_time"], "%Y-%m-%d").date()
        endDate = datetime.strptime(values["end_time"], "%Y-%m-%d").date()
        
        if startDate > endDate:
            raise "Start Date must be smaller than Enddate."
        
        df = df[(df["date"] >= startDate) & (df["date"] <= endDate)]
        df.dropna(inplace=True)
        x = df["main"].values
        y = df[appliances[0]].values
        x = jnp.pad(x, (units_to_pad, units_to_pad), 'constant', constant_values = (0,0))
        x = jnp.array([x[i: i + n] for i in range(len(x) - n + 1)])
        x_train.extend(x)
        y_train.extend(y)
    
    x_train = jnp.array(x_train)    
    y_train = jnp.array(y_train).reshape(-1,1)
    x_train = scaler_x.fit_transform(x_train)
    y_train = scaler_y.fit_transform(y_train)
    #test
    x_test = []
    y_test = []
    x_test_timestamp = []
    for key, values in test.items():
        df = pd.read_csv(f"datasets/Building{key}_NILM_data_basic.csv", usecols=["Timestamp","main", appliances[0]])
        df["date"] = pd.to_datetime(df["Timestamp"]).dt.date
        startDate = datetime.strptime(values["start_time"], "%Y-%m-%d").date()
        endDate = datetime.strptime(values["end_time"], "%Y-%m-%d").date()
        
        if startDate > endDate:
            raise "Start Date must be smaller than Enddate."     
        df = df[(df["date"] >= startDate) & (df["date"] <= endDate)]
        df.dropna(inplace=True)
        x = df["main"].values
        y = df[appliances[0]].values
        timestamp = df["Timestamp"].values
        x = jnp.pad(x, (units_to_pad, units_to_pad), 'constant', constant_values = (0,0))
        x = jnp.array([x[i: i + n] for i in range(len(x) - n + 1)])
        x_test.extend(x)
        y_test.extend(y)
        x_test_timestamp.extend(timestamp)
  
    x_test = jnp.array(x_test)
    y_test = jnp.array(y_test).reshape(-1,1)
    x_test = scaler_x.transform(x_test)
    x_train = jnp.array(x_train).reshape(x_train.shape[0], n, 1)
    y_train = jnp.array(y_train)
    x_test = jnp.array(x_test).reshape(x_test.shape[0], n, 1)
    y_test = jnp.array(y_test)
    x_train,x_cal , y_train, y_cal = train_test_split(x_train, y_train, test_size=split_factor, random_state=42)
    return x_train, y_train,x_cal,y_cal, x_test, y_test, x_test_timestamp, scaler_x, scaler_y
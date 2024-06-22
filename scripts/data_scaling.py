import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

def scale_data(train, test):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    return train_scaled, test_scaled, scaler

if __name__ == "__main__":
    train = pd.read_pickle("train.pkl")
    test = pd.read_pickle("test.pkl")
    train_scaled, test_scaled, scaler = scale_data(train, test)
    joblib.dump(scaler, "scaler.pkl")
    pd.DataFrame(train_scaled).to_pickle("train_scaled.pkl")
    pd.DataFrame(test_scaled).to_pickle("test_scaled.pkl")

import pandas as pd

def split_data(data, test_size):
    x = len(data) - test_size
    train = data.iloc[:x]
    test = data.iloc[x:]
    return train, test

if __name__ == "__main__":
    data_confirmed_country = pd.read_pickle("data_confirmed_country.pkl")
    train, test = split_data(data_confirmed_country, 14)
    train.to_pickle("train.pkl")
    test.to_pickle("test.pkl")

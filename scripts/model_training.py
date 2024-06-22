import pandas as pd
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense

def create_and_train_model(train_scaled, seq_size, n_features, epochs):
    train_generator = TimeseriesGenerator(train_scaled, train_scaled, length=seq_size, batch_size=1)
    model = Sequential()
    model.add(LSTM(150, activation="relu", return_sequences=True, input_shape=(seq_size, n_features)))
    model.add(LSTM(150, activation="relu"))
    model.add(Dense(64))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_generator, epochs=epochs)
    return model

if __name__ == "__main__":
    train_scaled = pd.read_pickle("train_scaled.pkl").values
    seq_size = 7
    n_features = 1
    model = create_and_train_model(train_scaled, seq_size, n_features, epochs=10)
    model.save("trained_model.h5")

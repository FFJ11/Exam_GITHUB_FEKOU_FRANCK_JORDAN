import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator


def evaluate_model(test_scaled, train_scaled, model, seq_size, n_features, future):
    test_generator = TimeseriesGenerator(test_scaled, test_scaled, length=seq_size, batch_size=1)
    history = model.evaluate(test_generator)

    prediction = []
    current_batch = train_scaled[-seq_size:].reshape(1, seq_size, n_features)
    for i in range(len(test_scaled) + future):
        current_pred = model.predict(current_batch)[0]
        prediction.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    return prediction


def plot_history(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'y', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training Loss and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_scaled = pd.read_pickle("train_scaled.pkl").values
    test_scaled = pd.read_pickle("test_scaled.pkl").values
    model = load_model("trained_model.h5")
    scaler = joblib.load("scaler.pkl")

    seq_size = 7
    n_features = 1
    future = 7

    prediction = evaluate_model(test_scaled, train_scaled, model, seq_size, n_features, future)
    rescaled_prediction = scaler.inverse_transform(prediction)

    test = pd.read_pickle("test.pkl")
    time_series_array = test.index
    for k in range(0, future):
        time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))

    df_forecast = pd.DataFrame(columns=["actual_confirmed", "predicted"], index=time_series_array)
    df_forecast.loc[:, "predicted"] = rescaled_prediction[:, 0]
    df_forecast.loc[:, "actual_confirmed"] = test["confirmed"]

    df_forecast.to_pickle("df_forecast.pkl")
    print(df_forecast.tail(10))

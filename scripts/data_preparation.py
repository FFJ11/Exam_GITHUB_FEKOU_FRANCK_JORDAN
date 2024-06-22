import pandas as pd

def load_and_prepare_data(filepath, country):
    data = pd.read_csv(filepath)
    data_country = data[data["Country/Region"] == country]
    data_confirmed_country = pd.DataFrame(data_country[data_country.columns[4:]].sum(), columns=["confirmed"])
    data_confirmed_country.index = pd.to_datetime(data_confirmed_country.index, format='%m/%d/%y')
    return data_confirmed_country

if __name__ == "__main__":
    data_confirmed_country = load_and_prepare_data('time_series_covid19_confirmed_global.csv', 'Cameroon')
    data_confirmed_country.to_pickle("data_confirmed_country.pkl")
    print(data_confirmed_country.head())
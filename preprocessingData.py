import pandas as pd

months = {
    "Jan": "01",
    "Feb": "02",
    "Mar": "03",
    "Apr": "04",
    "May": "05",
    "Jun": "06",
    "Jul": "07",
    "Aug": "08",
    "Sep": "09",
    "Oct": "10",
    "Nov": "11",
    "Dec": "12"
}

cash = {
    "K": "0",
    "M": "0000",
    "B": "0000000"
}


def reformat_date(x, y=0):
    t = x.replace(",", "").split(" ")
    return t[2] + '-' + months[t[0]] + '-' + t[1]


def reformat_cash(x, y=0):
    t = x.replace(".", "")
    return t[:-1] + cash[t[-1]] + ".0"


def preprocessing(filename):
    df = pd.read_csv("Data/" + filename)
    print(df.head())
    df['Date'] = df['Date'].apply(reformat_date)
    df['Vol.'] = df['Vol.'].apply(reformat_cash)
    print(df.head())
    df = df.drop(columns=["Change%"])
    print(df.head())
    df = df.rename(columns={"Price(in dollars)": "Close", "Vol.": "Volume", "Date": "Timestamp"})
    print(df.head())

    df.to_csv("Data/NEW" + filename, index=False)


file1 = "Cardano_2018-1-1_2021-7-27.csv"
file2 = "Litecoin_2016-8-24_2021-7-27.csv"
file3 = "Binance_2017-11-9_2021-7-27.csv"

preprocessing(file1)
preprocessing(file2)
preprocessing(file3)
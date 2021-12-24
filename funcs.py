from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import joblib
from sklearn.metrics import mean_squared_error
from math import sqrt

# Not used
def get_outliers(df_col:pd.Series, contam:float)-> list[int]:
    ''' Takes a dataframe column and returns 
        a list of index values with outliers 
        ====================================
        X: get array and reshape it for fitpredict()
        contamination:  proportion of outliers 
                        in the data
        labels: 1 = not outlier '''

    X = np.array(df_col).reshape(-1,1)
    model = LocalOutlierFactor(n_neighbors = 5, contamination = contam)
    labels = model.fit_predict(X)
    return np.array([i for i, label in enumerate(labels) if label!=1], dtype=np.int64)

# Not used
def delete_outliers(df:pd.DataFrame) -> None:
    ''' Takes a dataframe identifies outliers
        and deletes them '''
    outliers_list = []
    outliers_list.append(get_outliers(df_col = df['nvts'], contam=0.000001))
    outliers_list.append(get_outliers(df_col = df['nvt'], contam=0.000001))
    outliers_list.append(get_outliers(df_col = df['velocity'], contam = 0.000001))
    df.drop(df.index[list(np.concatenate(outliers_list, dtype=np.int64))], inplace=True)


def get_nvt_df(nvt_data:pd.DataFrame, coin:str)->pd.DataFrame:
    ''' Takes the consolidated NVT dataframe and returns 
        a dataframe with specified coin data '''
    return nvt_data[nvt_data['symbol']==coin].copy()


def pre_process_nvt_data(df:pd.DataFrame)-> pd.DataFrame:
    ''' Returns a processed df with daily frequency
        ================================================== 
        Deletes any NaN values 
        res: 1 value per day -> takes an average
        idx: creates a datetime index with day as frequency &
        fills any missing days with previous day's data  '''
    # Drop any NaN values
    df = df.dropna()
    # Convert data into daily fequency by averaging 
    res = df.resample('D').mean()
    # Create an index from the first and last dates of the df
    idx = pd.date_range(start=res.index.min(), end=res.index.max(), freq='D')
    # Fill any gaps in the data using forward fill method
    nvt_df = res.reindex(index=idx, method='ffill')
    return nvt_df


def get_hist_dict(coins:list) -> dict:
    ''' Returns a dict of dict with coin name as main key 
        and pairs and historial data from yFinance as other keys
        ========================================================
        pairs: a list of pairs with the coin and USD
        pairs_dict: creates a nested dict with the coin as main key 
        and 'pair' as the coin-USD key
        Checks to see if the pair is available on yFinance, downloads
        historical prices into dict.'''

    pairs = [coin+'-USD' for coin in coins]
    pairs_dict = {coin:{'pair':pair} for coin, pair in zip(coins, pairs)}
    for coin, pair in zip(coins,pairs):
        tick = yf.Ticker(pair)
        if tick.info['regularMarketPrice']==None:
            del pairs_dict[coin]
        else:
            pairs_dict[coin]['hist_df']=tick.history(period="max")
    return pairs_dict


def pre_process_hist_data(hist_df:pd.DataFrame)-> pd.DataFrame:
    ''' Takes a historical df from yFinance sets a datetime index, 
        deletes and NaN values, fills and missing data 
        with the previous day's data and retuns the Close price
        column as a dataframe'''
    hist_df = hist_df.dropna()
    idx = pd.date_range(start=hist_df.index.min(), end=hist_df.index.max(), freq='D')
    re_hist = hist_df.reindex(index=idx, method='ffill')
    return re_hist['Close'].to_frame()


def dynamic_nvts(df:pd.DataFrame)-> pd.DataFrame: 
    ''' Creates Buy & Sell signals based on a dynamic NVTS data
        =======================================================
        overbought: trend line -> calculated by using 2 year (600 days)
        mean and standard deviation 
        oversold: trend line -> calculated by using 2 year (600 days)
        mean and standard deviation
        signal_df: Close price, NVTS, overbougt & oversold columns
        If NVTS is greater than overbought -> Sell
        If NVTS is lower than oversold -> Buy 
        All other cases -> Hold '''
    # Check to see if there is 600 days worth of data:
    if df.shape[0] < 700:
        roll_days = int(df.shape[0]/2) 
        df['overbought'] = df['nvts'].rolling(roll_days).mean() + 2 * df['nvts'].rolling(roll_days).std()
        df['oversold'] = df['nvts'].rolling(roll_days).mean() + 0.5 * df['nvts'].rolling(roll_days).std()
    else:
        df['overbought'] = df['nvts'].rolling(600).mean() + 2 * df['nvts'].rolling(600).std()
        df['oversold'] = df['nvts'].rolling(600).mean() + 0.5 * df['nvts'].rolling(600).std()
        # Add buy & sell signals 
    signal_df = df.loc[df['Close'].first_valid_index():][['nvts', 'Close','overbought', 'oversold']].copy()
    conditions = [(signal_df['nvts']> signal_df['overbought']), (signal_df['nvts'] < signal_df['oversold'])]
    choices = ['Sell', 'Buy']
    signal_df['signal'] = np.select(conditions, choices, 'Hold')
    return signal_df


def get_money_1000(corpus) -> tuple:
    ''' Checks to see if there is any money in corpus
        if there is tries to give in $1000 denominations
        otherwise gives whatever there is'''
    if corpus >= 1000:
        corpus -= 1000
        return 1000, corpus
    elif 0 < corpus < 1000:
        money = corpus
        corpus = 0
        return money, corpus
    else:
        return 0, corpus


def trade(df:pd.DataFrame) -> tuple:
    ''' Takes a datframe and makes trades based on signals '''

    # fiat
    corpus = 100000
    crypto = 0    

    assert (corpus >= 0) & (crypto >=0), 'corpus|crypto cannot be negative!' 

    # Trade records:
    buys = []
    sells = []

    for row_index in range(df.shape[0]):
        if df.iloc[row_index]['signal'] == 'Hold':
            continue
        elif df.iloc[row_index]['signal'] == 'Buy':
            money, corpus = get_money_1000(corpus)
            if money == 0: 
                continue
            else:
                buys.append(row_index) 
                crypto += money/df.iloc[row_index]['Close']
        elif df.iloc[row_index]['signal'] == 'Sell':
            # Try selling $1000 worth of crypto
            if 0 < 1000/df.iloc[row_index]['Close'] <= crypto:
                crypto -= 1000/df.iloc[row_index]['Close'] 
                corpus += 1000/df.iloc[row_index]['Close'] * df.iloc[row_index]['Close']
                sells.append(row_index)
            # Sell whatever we have
            elif 0 < crypto:
                corpus += crypto * df.iloc[row_index]['Close']
                crypto = 0
                sells.append(row_index)
            else: continue
        else:continue
    return corpus, crypto, buys, sells



def plot(df:pd.DataFrame, title:str, buy_list:list, sell_list:list, corpus:float, crypto:float) -> plt.Axes:
    ''' Plots the close price & NVTS line with overbought & oversold lines. 
        Shows the buy and sell points on the close price line. '''

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)
    fig.suptitle(title)

    # Close price plot

    axs[0].plot(df.loc[df['overbought'].first_valid_index():]['Close'], label='Close price', color='cornflowerblue')
    
    for i in buy_list:
        axs[0].scatter(df.iloc[i].name, df.iloc[i]['Close'], marker='^', color='green', s=20, edgecolors='darkgreen')    
    
    for j in sell_list:
        axs[0].scatter(df.iloc[j].name, df.iloc[j]['Close'], marker='v', color='red', s=20, edgecolors='darkred')
    
    # Text box 
    text_str = (f"Total corpus as of {df.iloc[-1].name.date()} is \${corpus:,.2f}\nand {title} {crypto:.4f}"
                f"@ ${df.iloc[-1]['Close']:.2f} = \${crypto * df.iloc[-1]['Close']:,.2f}")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs[0].text(0.3, 0.95, text_str, transform=axs[0].transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        
    axs[0].legend(shadow=True, fancybox=True, loc='upper left')
    
    # NVTS plot
    axs[1].plot(df.loc[df['overbought'].first_valid_index():]['nvts'], label='NVTS', color='grey')
    axs[1].plot(df.loc[df['overbought'].first_valid_index():]['overbought'], label = 'overbought', color = 'red')
    axs[1].plot(df.loc[df['overbought'].first_valid_index():]['oversold'], label = 'oversold', color = 'green')
    axs[1].legend(shadow=True, fancybox=True, loc='upper left')

    # Hide x labels and tick labels for all but bottom plot.
    for ax in axs:
        ax.label_outer()
   
    plt.margins(x=0)
    plt.tight_layout()
    plt.show()


def analyze_and_plot(coin:str, pair:str, hist_df:pd.DataFrame, nvt_df:pd.DataFrame)-> plt.Axes:
    ''' Returns a plot after downloading, cleaning and running a simulation '''
    coin_nvt_df = get_nvt_df(nvt_df, coin)
    clean_nvt_df = pre_process_nvt_data(coin_nvt_df)
    clean_hist_df = pre_process_hist_data(hist_df)
    merged_df = pd.merge(clean_nvt_df, clean_hist_df, left_index=True, right_index=True, how='left')
    signal_df = dynamic_nvts(merged_df)
    corpus, crypto, buys, sells = trade(signal_df)
    plot(signal_df, coin, buys, sells, corpus, crypto)


def ml_example(pair:str) -> plt.Axes:
    ''' Takes a coin-USD pair and returns an example training & test
        split with random values as the prediction '''
    btc_hist = yf.Ticker(pair).history(period="max")
    train_set = btc_hist[:int(btc_hist.shape[0]*.80)] 
    test_set = btc_hist[int(btc_hist.shape[0]*.80):].copy()
    test_set['Predict'] = test_set['Close'] * np.random.uniform(low=0.1, high=1.1)
    mse = round(mean_squared_error(test_set['Close'], test_set['Predict']),4)
    rmse = round(sqrt(mse),4)

    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    ax.plot(btc_hist.index,btc_hist['Close'], color='blue', label='Close price')
    ax.plot(train_set['Close'], color='green', alpha=0.5, linewidth=5, label='training data')
    ax.plot(test_set['Predict'], color='red', alpha=1, linewidth=1, label='predictions')

    # Text box 
    text_str = (f"Mean Square Error = {mse}\n Root Mean Square Error = {rmse}")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.annotate(text_str, xy=(0.3, 0.9), xycoords='axes fraction', bbox=props)

    plt.title(pair)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    pass
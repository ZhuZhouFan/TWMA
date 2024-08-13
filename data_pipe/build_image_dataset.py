import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse

def construct_feature_and_label(stock:str, 
                                date:str, 
                                lag_order:int, 
                                horizon_list:list, 
                                date_array:np.array, 
                                seq_data_path:str,
                                fig_path:str, 
                                fig_type:str = 'ohlc'):
    raw_df = pd.read_csv(f'{seq_data_path}/{stock}.csv', index_col = 'date')
    mpf_df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'], index = date_array)
    mpf_df['Open'] = raw_df['open']
    mpf_df['High'] = raw_df['high']
    mpf_df['Close'] = raw_df['close']
    mpf_df['Low'] = raw_df['low']
    mpf_df['Volume'] = raw_df['volume']
    mpf_df[f'MA{lag_order}'] = raw_df['close'].rolling(lag_order).mean()
    mpf_df.index = pd.to_datetime(mpf_df.index.values)
    
    previous_date = date_array[np.where(date_array == date)[0].item() - lag_order + 1]
    figure_df = mpf_df.loc[previous_date:date, :].copy()
    
    no_trade_count = (figure_df['Volume'] == 0).sum()
    non_existing_count = figure_df['Volume'].isna().sum()
    if max(no_trade_count, non_existing_count) > round(lag_order/2):
        return None
    
    base_price = figure_df.loc[previous_date, 'Close']
    base_volume = figure_df.loc[previous_date, 'Volume']
    if (base_volume == 0) or np.isnan(base_volume):
        for j in range(round(lag_order/2) + 1):
            alternative_date = date_array[np.where(date_array == date)[0].item() - lag_order + 1 + j]
            base_volume = figure_df.loc[alternative_date, 'Volume']
            if base_volume > 0:
                break
    
    if np.isnan(base_volume) or np.isnan(base_price):
        return None
    
    figure_df[['Open', 'High', 'Low', 'Close', f'MA{lag_order}']] = figure_df[['Open', 'High', 'Low', 'Close', f'MA{lag_order}']] / base_price
    figure_df['Volume'] = figure_df['Volume'] / base_volume
    
    file_save_path = f'{fig_path}/{fig_type}/{date}/{stock}'
    if not os.path.exists(file_save_path):
        os.makedirs(file_save_path)
    
    candle_color = mpf.make_marketcolors(up = 'g',
                      down='r',
                      edge = 'inherit',
                      wick = 'inherit',
                      volume = 'inherit')
    background_sty = mpf.make_mpf_style(marketcolors = candle_color,
                                    figcolor = '(0.82, 0.83, 0.85)',
                                    gridcolor='(0.82, 0.83, 0.85)')
    fig = mpf.figure(style = background_sty,
                     figsize = (16, 16),
                     facecolor = (0.82, 0.83, 0.85))
    ax1 = fig.add_axes([0.0, 0.2, 1, 0.80])
    ax2 = fig.add_axes([0.0, 0.0, 1, 0.20], sharex = ax1)
    ap = mpf.make_addplot(figure_df[[f'MA{lag_order}']], ax = ax1)
    mpf.plot(figure_df, 
             ax = ax1,
             volume = ax2, 
             addplot = ap, 
             style = background_sty,
             type = fig_type)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_yticks([])
    ax1.set_ylabel(None)
    ax2.set_ylabel(None)
    
    figure_file_path = f'{file_save_path}/I{lag_order}.jpeg'
    
    if os.path.exists(figure_file_path):
        pass
    else:
        plt.savefig(figure_file_path, dpi = 1024/16)
    plt.close()
    
    for horizon in horizon_list:
        target_location = np.where(date_array == date)[0].item() + horizon
        label_file_path = f'{file_save_path}/R{horizon}.npy'
        if (target_location >= date_array.shape[0]) or (os.path.exists(label_file_path)):
            continue
        else:
            target_date = date_array[target_location]
            ret = mpf_df.loc[target_date, 'Close']/mpf_df.loc[date, 'Close'] - 1
            if np.isnan(ret):
                ret = 0
            np.save(label_file_path, ret)
            
def sub_job(date:str,
            index_components_df:pd.DataFrame,
            lag_order:int,
            horizon_list:list,
            date_array:np.array,
            seq_data_path:str,
            fig_path:str,
            fig_type:str):
    onlist_stock_list = np.unique(index_components_df.loc[date, :].values).tolist()
    for stock in onlist_stock_list:
        try:
            construct_feature_and_label(stock, 
                                        date,
                                        lag_order,
                                        horizon_list,
                                        date_array,
                                        seq_data_path,
                                        fig_path, 
                                        fig_type)
        except Exception as e:
            print(stock, date, e)
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lag-order', type=int, default=5,
                        help='Number of trading days included in each image')
    parser.add_argument('--start-time', type=str, default='2014-01-01',
                        help='Dataset start time')
    parser.add_argument('--end-time', type=str, default='2023-05-10',
                        help='Dataset end time')
    parser.add_argument('--worker', type=int, default=50,
                        help='Number of cpus used in this program.')
    args = parser.parse_args()
    
    data_root = 'Your/Data/Path/Here'
    seq_data_path = f'{data_root}/kline_day'
    fig_path = 'Your/Image/Path/Here'
    index_df = pd.read_csv(f'{data_root}/index_kline_day/000852.XSHG.csv', index_col = 'date')
    date_array = index_df.index.values

    lag_order = args.lag_order
    horizon_list = [1, 5, 20]
    fig_type = 'ohlc'

    zz800_df = pd.read_csv(f'{data_root}/index_components/000906.XSHG.csv', index_col='date')
    zz1000_df = pd.read_csv(f'{data_root}/index_components/000852.XSHG.csv', index_col='date')
    index_components_df = pd.concat([zz800_df, zz1000_df], axis = 1)
    index_components_df = index_components_df.loc[args.start_time:args.end_time, :]
    selected_date_array = index_components_df.index.values

    Parallel(n_jobs=args.worker)(delayed(sub_job)
            (date, index_components_df, lag_order, horizon_list, date_array, seq_data_path, fig_path, fig_type)
            for date in tqdm(selected_date_array))
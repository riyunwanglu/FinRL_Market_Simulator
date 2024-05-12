import os
import torch
import pandas as pd

TEN = torch.Tensor


def _add_norm(tensor_dicts, n_days=5):
    pass

    '''get avg and the vam(the mean of var)'''
    price_avgs = torch.stack([d['price'].mean(dim=0) for d in tensor_dicts])
    price_vams = torch.stack([(d['price'] ** 2).mean(dim=0, keepdim=True) for d in tensor_dicts])
    volume_avgs = torch.stack([d['volume'].mean(dim=0) for d in tensor_dicts])
    volume_vams = torch.stack([(d['volume'] ** 2).mean(dim=0, keepdim=True) for d in tensor_dicts])
    value_avgs = torch.stack([d['value'].mean(dim=0) for d in tensor_dicts])
    value_vams = torch.stack([(d['value'] ** 2).mean(dim=0, keepdim=True) for d in tensor_dicts])

    '''save avg and std for normalization using previous day data'''

    def get_std(_vam, _avg):
        return torch.sqrt(_vam - _avg ** 2)

    for i_day in range(len(tensor_dicts)):
        i, j = (0, n_days) if i_day <= n_days else (i_day - n_days, i_day)
        tensor_dict = tensor_dicts[i_day]

        price_avg = price_avgs[i:j].mean()
        price_std = get_std(price_vams[i:j].mean(), price_avg)
        tensor_dict['price_norm'] = price_avg.item(), price_std.item()

        volume_avg = volume_avgs[i:j].mean()
        volume_std = get_std(volume_vams[i:j].mean(), volume_avg)
        tensor_dict['volume_norm'] = volume_avg.item(), volume_std.item()

        value_avg = value_avgs[i:j].mean()
        value_std = get_std(value_vams[i:j].mean(), value_avg)
        tensor_dict['value_norm'] = value_avg.item(), value_std.item()
    return tensor_dicts


def _add_share_style(tensor_dicts, n_days=5, data_dir='./share_mkt_equ_daily_zhongzheng500'):
    from shares_config import SharesZhongZheng500
    from shares_classify import download_market_equity_daily

    '''get mkt_equity_daily_df'''
    if not os.path.exists(data_dir):
        download_market_equity_daily(
            share_symbols=SharesZhongZheng500,
            data_dir=data_dir,
            beg_date='20101031',
            end_date='20201031',
        )

    share_symbols = [tensor_dict['share_symbol'] for tensor_dict in tensor_dicts]
    share_symbols = list(set(share_symbols))
    assert len(share_symbols) == 1
    share_symbol = share_symbols[0]
    csv_path = f'{data_dir}/{share_symbol}.csv'
    mkt_equity_daily_df = pd.read_csv(csv_path)
    mkt_equity_daily_df = mkt_equity_daily_df.set_index('date')

    turnover_rates = []
    neg_mkt_values = []
    for tensor_dict in tensor_dicts:
        trade_date = tensor_dict['trade_date']
        turnover_rates.append(mkt_equity_daily_df.loc[trade_date, 'turnover_rate'])
        neg_mkt_values.append(mkt_equity_daily_df.loc[trade_date, 'neg_mkt_value'])
    turnover_rates = torch.tensor(turnover_rates)
    neg_mkt_values = torch.tensor(neg_mkt_values)

    for i_day in range(len(tensor_dicts)):
        i, j = (0, n_days) if i_day <= n_days else (i_day - n_days, i_day)
        tensor_dict = tensor_dicts[i_day]

        tensor_dict['turnover_rate'] = turnover_rates[i:j].mean()
        tensor_dict['neg_mkt_value'] = neg_mkt_values[i:j].mean()
    return tensor_dicts

import chinese_calendar
import datetime
 
def get_tradeday(beg_date, end_date):
    start = datetime.datetime.strptime(beg_date, '%Y%m%d') # 将字符串转换为datetime格式
    end = datetime.datetime.strptime(end_date, '%Y%m%d')
    # 获取指定范围内工作日列表
    lst = chinese_calendar.get_workdays(start,end)
    expt = []
    # 找出列表中的周六，周日，并添加到空列表
    for time in lst:
        if time.isoweekday() == 6 or time.isoweekday() == 7:
            expt.append(time)
    # 将周六周日排除出交易日列表
    for time in expt:
        lst.remove(time)
    date_list = [item.strftime('%Y%m%d') for item in lst] #列表生成式，strftime为转换日期格式
    return date_list

def get_trade_dates(beg_date: str = '2022.09.01', end_date: str = '2022.09.15') -> [str]:
    from ideadata.stock.trade_calendar import TradeCalendar
    cal_df = TradeCalendar().get_trade_cal(beg_date, end_date)
    cal_df = cal_df[cal_df.is_open == 1]

    trade_dates = [item.date for item in cal_df.itertuples()]
    return trade_dates


def get_share_dicts_by_day(share_dir='./shares_data_by_day', share_symbol='000525.sz',
                           beg_date='20220901', end_date='20220930',
                           n_levels=5, device=None):
    if device is None:
        gpu_id = 0
        device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

    '''convert feather to tensor by day'''
    
    trade_dates = get_tradeday(beg_date=beg_date, end_date=end_date)

    tensor_dicts = []
    for trade_date in trade_dates:
        file_path = f"{share_dir}/{share_symbol}/{trade_date}/snapshot.csv"
        tensor_dict = get_snapshot_tensor_dict(trade_date, file_path=file_path, device=device, n_levels=n_levels)
        tensor_dict['trade_date'] = trade_date
        tensor_dict['share_symbol'] = share_symbol
        tensor_dicts.append(tensor_dict)
    return tensor_dicts

def get_snapshot_tensor_dict(trade_date:str, file_path: str, device, n_levels: int = 5):
    df = pd.read_csv(file_path)
    df = df.rename(columns = {'Time':'UpdateTime', 'Match':'LastPrice'})

    def get_tensor(ary):
        return torch.tensor(ary, dtype=torch.float32, device=device)

    assert df.shape[0] == 4800 + 2

    """get data for building tensor_dict"""
    '''get delta volume'''
    volume = get_tensor(df['Volume'].values)
    volume[1:] = torch.diff(volume, n=1)  # delta volume
    torch.nan_to_num_(volume, nan=0)

    '''get delta turnover (value)'''
    value = get_tensor(df['Turnover'].values)
    value[1:] = torch.diff(value, n=1)  # delta turnover (value)
    torch.nan_to_num_(value, nan=0)

    '''get last price'''
    last_price = get_tensor(df['LastPrice'].ffill().values)  # last price
    last_price[last_price == 0] = last_price[last_price > 0][-1]

    '''fill nan in ask_prices and ask_volumes'''
    ask_prices = get_tensor(df[[f'AskPrice{i}' for i in range(1, n_levels + 1)]].values).T
    assert ask_prices.shape == (n_levels, len(df))
    for i in range(n_levels):
        prev_price = ask_prices[i - 1] if i > 0 else last_price
        ask_prices[i] = fill_zero_and_nan_with_tensor(ask_prices[i], prev_price + 0.01)
    ask_volumes = get_tensor(df[[f'AskVol{i}' for i in range(1, n_levels + 1)]].values).T
    torch.nan_to_num_(ask_volumes, nan=0)

    '''fill nan in bid_prices and bid_volumes'''
    bid_prices = get_tensor(df[[f'BidPrice{i}' for i in range(1, n_levels + 1)]].values).T
    assert bid_prices.shape == (n_levels, len(df))
    for i in range(n_levels):
        prev_price = bid_prices[i - 1] if i > 0 else last_price
        bid_prices[i] = fill_zero_and_nan_with_tensor(bid_prices[i], prev_price - 0.01)
    bid_volumes = get_tensor(df[[f'BidVol{i}' for i in range(1, n_levels + 1)]].values).T
    torch.nan_to_num_(bid_volumes, nan=0)

    return {'last_price': last_price, 'volume': volume, 'value': value,
            'ask_prices': ask_prices, 'ask_volumes': ask_volumes,
            'bid_prices': bid_prices, 'bid_volumes': bid_volumes}


def fill_zero_and_nan_with_tensor(src: TEN, dst: TEN) -> TEN:
    fill_bool = torch.logical_or(torch.isnan(src), src == 0)
    src[fill_bool] = dst[fill_bool]
    return src


'''unit tests'''


def test_get_trade_dates():
    trade_dates = get_trade_dates(beg_date='2022.09.01', end_date='2022.09.15')
    print(trade_dates)
    assert trade_dates == ['2022-09-01', '2022-09-02', '2022-09-05', '2022-09-06', '2022-09-07',
                           '2022-09-08', '2022-09-09', '2022-09-13', '2022-09-14', '2022-09-15']


def test_csv_to_tensor_dict():
    gpu_id = 0
    share_dir = './shares_data_by_day_zhongzheng500'
    share_symbol = '000525_XSHE'
    trade_date = '2022-09-01'
    n_levels = 5

    csv_path = f"{share_dir}/{share_symbol}/{trade_date}.csv"
    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    tensor_dict = get_snapshot_tensor_dict(csv_path=csv_path, device=device, n_levels=n_levels)
    for k, v in tensor_dict.items():
        print(f"{k} {v.shape}")
    assert tensor_dict['price'].shape == (4800 + 2,)
    assert tensor_dict['volume'].shape == (4800 + 2,)
    assert tensor_dict['value'].shape == (4800 + 2,)
    assert tensor_dict['ask_prices'].shape == (n_levels, 4800 + 2)
    assert tensor_dict['ask_volumes'].shape == (n_levels, 4800 + 2)
    assert tensor_dict['bid_prices'].shape == (n_levels, 4800 + 2)
    assert tensor_dict['bid_volumes'].shape == (n_levels, 4800 + 2)


def test_get_share_dicts_by_day():
    gpu_id = 0
    share_dir = 'share_daily_zhongzheng500'
    share_symbol = '000525_XSHE'
    n_levels = 5
    n_days = 5
    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

    tensor_dicts = get_share_dicts_by_day(share_dir=share_dir, share_symbol=share_symbol,
                                          beg_date='2022-09-01', end_date='2022-09-30',
                                          n_levels=n_levels, device=device)
    for tensor_dict in tensor_dicts:
        trade_date = tensor_dict['trade_date']
        share_symbol = tensor_dict['share_symbol']
        print(f"trade_date {trade_date}    share_symbol {share_symbol}")
        for k, v in tensor_dict.items():
            if isinstance(v, str):
                continue
            v_info = v.shape if isinstance(v, TEN) else v
            print(f"    {k} {v_info}")

    tensor_dicts = _add_norm(tensor_dicts=tensor_dicts, n_days=n_days)
    for tensor_dict in tensor_dicts:
        trade_date = tensor_dict['trade_date']
        print(f"trade_date {trade_date}")
        for k in ('price_norm', 'volume_norm', 'value_norm'):
            avg, std = tensor_dict[k]
            print(f"    {k:16}    avg {avg:10.3e}    std {std:10.3e}")

    tensor_dicts = _add_share_style(tensor_dicts=tensor_dicts, n_days=n_days,
                                    data_dir='./share_mkt_equ_daily_zhongzheng500')
    for tensor_dict in tensor_dicts:
        trade_date = tensor_dict['trade_date']
        turnover_rate = tensor_dict['turnover_rate']
        neg_mkt_value = tensor_dict['neg_mkt_value']
        print(f"trade_date {trade_date}    turnover_rate {turnover_rate:10.3e}    neg_mkt_value {neg_mkt_value:10.3e}")

    assert isinstance(tensor_dicts, list)
    assert isinstance(tensor_dicts[0], dict)


if __name__ == '__main__':
    # test_get_trade_dates()
    # test_csv_to_tensor_dict()
    test_get_share_dicts_by_day()

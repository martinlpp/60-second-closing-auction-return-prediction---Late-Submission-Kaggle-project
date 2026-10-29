import pandas as pd
import numpy as np


def apply_feature_engineering(df):

    grouper = df.groupby(['stock_id', 'date_id'])

    df['diff_wap'] = grouper['wap'].diff()
    df['diff_wap_60'] = grouper['wap'].diff(6)
    df['diff_wap_300'] = grouper['wap'].diff(30)
    
    mean_wap_60 = grouper['wap'].transform(lambda x : x.rolling(6).mean())
    mean_wap_300 = grouper['wap'].transform(lambda x : x.rolling(30).mean())
    mean_ref_60 = grouper['reference_price'].transform(lambda x : x.rolling(6).mean())
    mean_ref_300 = grouper['reference_price'].transform(lambda x : x.rolling(30).mean())
    
    df['diff_ref_wap'] = df['reference_price'] - df['wap']
    df['diff_ref_wap_60'] = mean_ref_60 - mean_wap_60
    df['diff_ref_wap_300'] = mean_ref_300 - mean_wap_300

    # Price book spread & depths
    df['spread_price_book'] = df['ask_price'] - df['bid_price']
    df['book_depth'] = df['ask_size'] + df['bid_size']
    df['imbalance_book_depth'] = df['bid_size'] - df['ask_size']
    df['imbalance_book_flag_ratio'] = (df['bid_size'] - df['ask_size']) / (df['ask_size'] + df['bid_size'] + 1e-8)
    
    df['auction_depth'] = df['imbalance_size'] + df['matched_size']
    df['imbalance_auction_depth'] = df['imbalance_size'] 
    df['imbalance_auction_flag_ratio'] = (df['imbalance_buy_sell_flag'] * df['imbalance_size']) / (df['imbalance_size'] + df['matched_size'] + 1e-8)

    df['diff_book_ratio'] = grouper['imbalance_book_flag_ratio'].diff(1)
    df['diff_book_ratio_60'] = grouper['imbalance_book_flag_ratio'].diff(6)
    df['diff_auction_ratio'] = grouper['imbalance_auction_flag_ratio'].diff(1)
    df['diff_auction_ratio_60'] = grouper['imbalance_auction_flag_ratio'].diff(6)
    df['diff_spread'] = grouper['spread_price_book'].diff(1)
    df['diff_spread_60'] = grouper['spread_price_book'].diff(6)
    df['diff_book_depth'] = grouper['book_depth'].diff(1)
    df['diff_book_depth_60'] = grouper['book_depth'].diff(6)
    df['diff_auction_depth'] = grouper['imbalance_auction_depth'].diff(1)
    df['diff_auction_depth_60'] = grouper['imbalance_auction_depth'].diff(6)

    # Rolling price & size
    df['ask_price_60'] = grouper['ask_price'].transform(lambda x: x.rolling(6).mean())
    df['ask_price_300'] = grouper['ask_price'].transform(lambda x: x.rolling(30).mean())
    df['bid_price_60'] = grouper['bid_price'].transform(lambda x: x.rolling(6).mean())
    df['bid_price_300'] = grouper['bid_price'].transform(lambda x: x.rolling(30).mean())
    df['spread_price_book_60'] = df['ask_price_60'] - df['bid_price_60']
    df['spread_price_book_300'] = df['ask_price_300'] - df['bid_price_300']

    df['ask_size_60'] = grouper['ask_size'].transform(lambda x: x.rolling(6).mean())
    df['ask_size_300'] = grouper['ask_size'].transform(lambda x: x.rolling(30).mean())
    df['bid_size_60'] = grouper['bid_size'].transform(lambda x: x.rolling(6).mean())
    df['bid_size_300'] = grouper['bid_size'].transform(lambda x: x.rolling(30).mean())
    df['imbalance_size_60'] = grouper['imbalance_size'].transform(lambda x: x.rolling(6).mean())

    df['book_depth_60'] = df['ask_size_60'] + df['bid_size_60']
    df['book_depth_300'] = df['ask_size_300'] + df['bid_size_300']
    df['imbalance_book_depth_60'] = df['bid_size_60'] - df['ask_size_60']
    df['imbalance_book_depth_300'] = df['bid_size_300'] - df['ask_size_300']

    df['imbalance_book_flag_ratio_60'] = (df['bid_size_60'] - df['ask_size_60']) / (df['ask_size_60'] + df['bid_size_60'] + 1e-8)
    df['imbalance_book_flag_ratio_300'] = (df['bid_size_300'] - df['ask_size_300']) / (df['ask_size_300'] + df['bid_size_300'] + 1e-8)
    
    df['imbalance_size_300'] = grouper['imbalance_size'].transform(lambda x: x.rolling(30).mean())
    df['matched_size_60'] = grouper['matched_size'].transform(lambda x: x.rolling(6).mean())
    df['matched_size_300'] = grouper['matched_size'].transform(lambda x: x.rolling(30).mean())

    df['auction_depth_60'] = df['imbalance_size_60'] + df['matched_size_60']
    df['auction_depth_300'] = df['imbalance_size_300'] + df['matched_size_300']
    
    df['signed_imbalance_size'] = df['imbalance_buy_sell_flag'] * df['imbalance_size']
    df['signed_imbalance_size_60'] = grouper['signed_imbalance_size'].transform(lambda x: x.rolling(6).mean())
    df['signed_imbalance_size_300'] = grouper['signed_imbalance_size'].transform(lambda x: x.rolling(30).mean())

    df['imbalance_auction_flag_ratio_60'] = df['signed_imbalance_size_60'] / (df['imbalance_size_60'] + df['matched_size_60'] + 1e-8)
    df['imbalance_auction_flag_ratio_300'] = df['signed_imbalance_size_300'] / (df['imbalance_size_300'] + df['matched_size_300'] + 1e-8)
    
    df['mid_price'] = (df['ask_price'] + df['bid_price']) / 2

    # Price pressures & spreads
    df['price_pressure'] = df['wap'] - df['mid_price']
    df['ask_ref_spread'] = df['ask_price'] - df['reference_price']
    df['ref_bid_spread'] = df['reference_price'] - df['bid_price']
    df['ref_spread_asymmetry'] = df['ask_ref_spread'] / (df['ref_bid_spread'] + 1e-8)

    # Market information
    market_imbalance_total_book_depth = df['imbalance_book_depth'].abs().groupby([df['date_id'], df['seconds_in_bucket']]).transform('sum')
    df['imbalance_market_book_depth_ratio'] = df['imbalance_book_depth'] / (market_imbalance_total_book_depth + 1e-8)
    
    market_imbalance_total_book_depth_60 = df['imbalance_book_depth_60'].abs().groupby([df['date_id'], df['seconds_in_bucket']]).transform('sum')
    df['imbalance_market_book_depth_ratio_60'] = df['imbalance_book_depth_60'] / (market_imbalance_total_book_depth_60 + 1e-8)
    
    market_imbalance_total_auction_depth = df.groupby(['date_id', 'seconds_in_bucket'])['imbalance_auction_depth'].transform('sum')
    df['imbalance_market_auction_depth_ratio'] = df['imbalance_auction_depth'] / (market_imbalance_total_auction_depth + 1e-8)
    
    market_imbalance_total_auction_depth_60 = df.groupby(['date_id', 'seconds_in_bucket'])['imbalance_size_60'].transform('sum')
    df['imbalance_market_auction_depth_ratio_60'] = df['imbalance_size_60'] / (market_imbalance_total_auction_depth_60 + 1e-8)
    
    market_total_book_depth = df.groupby(['date_id', 'seconds_in_bucket'])['book_depth'].transform('sum')
    df['market_book_depth_ratio'] = df['book_depth'] / (market_total_book_depth + 1e-8)
    
    market_total_book_depth_60 = df.groupby(['date_id', 'seconds_in_bucket'])['book_depth_60'].transform('sum')
    df['market_book_depth_ratio_60'] = df['book_depth_60'] / (market_total_book_depth_60 + 1e-8)
    
    market_total_auction_depth = df.groupby(['date_id', 'seconds_in_bucket'])['matched_size'].transform('sum') + df.groupby(['date_id', 'seconds_in_bucket'])['imbalance_size'].transform('sum')
    df['market_auction_depth_ratio'] = df['auction_depth'] / (market_total_auction_depth + 1e-8)
    
    market_total_auction_depth_60 = df.groupby(['date_id', 'seconds_in_bucket'])['auction_depth_60'].transform('sum')
    df['market_auction_depth_ratio_60'] = df['auction_depth_60'] / (market_total_auction_depth_60 + 1e-8)

    # Weights diffs & ratios
    df['diff_liquidity_weights'] = df['market_book_depth_ratio'] - df['market_auction_depth_ratio']
    df['diff_liquidity_weights_60'] = df['market_book_depth_ratio_60'] - df['market_auction_depth_ratio_60']
    df['ratio_liquidity_weights'] = df['market_book_depth_ratio'] / (df['market_auction_depth_ratio'] + 1e-8)

    # Market flags
    market_bid = df.groupby(['date_id', 'seconds_in_bucket'])['bid_size'].transform('sum')
    market_ask = df.groupby(['date_id', 'seconds_in_bucket'])['ask_size'].transform('sum')
    df['market_book_flag_ratio'] = (market_bid - market_ask) / (market_bid + market_ask + 1e-8)
    
    market_bid_60 = df.groupby(['date_id', 'seconds_in_bucket'])['bid_size_60'].transform('sum')
    market_ask_60 = df.groupby(['date_id', 'seconds_in_bucket'])['ask_size_60'].transform('sum')
    df['market_book_flag_ratio_60'] = (market_bid_60 - market_ask_60) / (market_bid_60 + market_ask_60 + 1e-8)
    
    df['excess_flag_ratio'] = df['imbalance_book_flag_ratio'] - df['market_book_flag_ratio']
    df['excess_flag_ratio_60'] = df['imbalance_book_flag_ratio_60'] - df['market_book_flag_ratio_60']
    
    market_signed_imb = df.groupby(['date_id', 'seconds_in_bucket'])['signed_imbalance_size'].transform('sum')
    df['market_auction_flag_ratio'] = market_signed_imb / (market_total_auction_depth + 1e-8)
    df['excess_auction_flag_ratio'] = df['imbalance_auction_flag_ratio'] - df['market_auction_flag_ratio']
    
    market_signed_imb_60 = df.groupby(['date_id', 'seconds_in_bucket'])['signed_imbalance_size_60'].transform('sum')
    df['market_auction_flag_ratio_60'] = market_signed_imb_60 / (market_total_auction_depth_60 + 1e-8)
    df['excess_auction_flag_ratio_60'] = df['imbalance_auction_flag_ratio_60'] - df['market_auction_flag_ratio_60']

    # Market WAPs
    wap_x_auction_w = df['wap'] * df['market_auction_depth_ratio']
    df['market_wap_auction_w'] = wap_x_auction_w.groupby([df['date_id'], df['seconds_in_bucket']]).transform('sum')
    df['wap_rel_market'] = df['wap'] - df['market_wap_auction_w']
    
    wap60_x_auction_w60 = mean_wap_60 * df['market_auction_depth_ratio_60']
    df['market_wap_auction_w_60'] = wap60_x_auction_w60.groupby([df['date_id'], df['seconds_in_bucket']]).transform('sum')
    df['wap_rel_market_60'] = mean_wap_60 - df['market_wap_auction_w_60']
    
    wap_x_book_w = df['wap'] * df['market_book_depth_ratio']
    df['market_wap_book_w'] = wap_x_book_w.groupby([df['date_id'], df['seconds_in_bucket']]).transform('sum')
    df['wap_rel_market_book'] = df['wap'] - df['market_wap_book_w']
    
    wap60_x_book_w60 = mean_wap_60 * df['market_book_depth_ratio_60']
    df['market_wap_book_w_60'] = wap60_x_book_w60.groupby([df['date_id'], df['seconds_in_bucket']]).transform('sum')
    df['wap_rel_market_book_60'] = mean_wap_60 - df['market_wap_book_w_60']

    # Causal Z-scores
    def add_causal_z(dframe, cols, window=20, min_days=3):
        daily = dframe.groupby(['stock_id', 'date_id'])[cols].agg(['mean', 'std']).sort_index()
        rolling_stats = (
            daily.groupby(level=0)
            .rolling(window=window, min_periods=min_days)
            .mean()
            .droplevel(0)
            .groupby(level=0)
            .shift(1)
        )
        idx = pd.MultiIndex.from_frame(dframe[['stock_id', 'date_id']])
        for c in cols:
            m = rolling_stats[(c, 'mean')].reindex(idx).to_numpy(dtype='float32')
            s = rolling_stats[(c, 'std')].reindex(idx).to_numpy(dtype='float32')
            dframe[f'{c}_z'] = ((dframe[c].to_numpy(dtype='float32') - m) / (s + 1e-8))
        return dframe

    colums_z_add = [
        'imbalance_book_flag_ratio', 'diff_ref_wap', 'spread_price_book',
        'diff_auction_ratio', 'imbalance_auction_flag_ratio', 'excess_flag_ratio',
        'imbalance_book_depth', 'imbalance_market_book_depth_ratio', 'wap_rel_market',
        'matched_size', 'book_depth', 'imbalance_size'
    ]
    df = add_causal_z(df, colums_z_add)

    # Auction price spreads & diffs
    df['diff_near_wap'] = df['near_price'] - df['wap']
    df['diff_far_wap'] = df['far_price'] - df['wap']
    df['diff_near_far'] = df['near_price'] - df['far_price']
    df['diff_near_ref'] = df['near_price'] - df['reference_price']
    df['auction_book_depth_ratio'] = df['auction_depth'] / (df['auction_depth'] + df['book_depth'] + 1e-8)
    
    g = df.groupby(['stock_id', 'date_id'])
    df['diff_wap_rel_market'] = g['wap_rel_market'].diff(1)
    df['diff_wap_rel_market_60'] = g['wap_rel_market'].diff(6)
    df['diff_excess_flag_ratio'] = g['excess_flag_ratio'].diff(1)
    df['diff_excess_flag_ratio_60'] = g['excess_flag_ratio'].diff(6)
    df['diff_ref_wap_10'] = g['diff_ref_wap'].diff(1)
    df['diff_ref_wap_change_60'] = g['diff_ref_wap'].diff(6)
    df['diff_near_price'] = g['near_price'].diff(1)
    df['diff_near_price_60'] = g['near_price'].diff(6)
    df['diff_near_wap_10'] = g['diff_near_wap'].diff(1)
    df['diff_near_wap_60'] = g['diff_near_wap'].diff(6)

    return df

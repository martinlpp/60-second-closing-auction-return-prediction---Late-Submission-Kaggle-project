import sys
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

from features import apply_feature_engineering

features = ['stock_id', 'seconds_in_bucket', 'reference_price', 'matched_size',
       'far_price', 'near_price', 'wap', 'diff_wap',
       'diff_wap_60', 'diff_wap_300', 'diff_ref_wap', 'diff_ref_wap_60',
       'diff_ref_wap_300', 'spread_price_book', 'book_depth', 'imbalance_book_depth',
       'imbalance_book_flag_ratio', 'auction_depth', 'imbalance_auction_depth',
       'imbalance_auction_flag_ratio', 'spread_price_book_60',
       'spread_price_book_300', 'imbalance_book_depth_60', 'imbalance_book_depth_300',
       'imbalance_book_flag_ratio_60', 'imbalance_book_flag_ratio_300','imbalance_size_60',
       'imbalance_size_300',
       'imbalance_auction_flag_ratio_60', 'imbalance_auction_flag_ratio_300',
       'diff_book_ratio', 'diff_book_ratio_60', 'diff_auction_ratio',
       'diff_auction_ratio_60', 'diff_spread', 'diff_spread_60', 
       'diff_book_depth', 'diff_book_depth_60', 'diff_auction_depth',
       'diff_auction_depth_60', 'mid_price', 'imbalance_market_book_depth_ratio', 
        'imbalance_market_book_depth_ratio_60', 'imbalance_market_auction_depth_ratio',
        'imbalance_market_auction_depth_ratio_60', 'market_book_depth_ratio',
        'market_book_depth_ratio_60', 'market_auction_depth_ratio',
        'market_auction_depth_ratio_60', 'market_book_flag_ratio', 'market_book_flag_ratio_60',
        'excess_flag_ratio', 'excess_flag_ratio_60', 'wap_rel_market', 'wap_rel_market_60']

z_features = [
    'imbalance_book_flag_ratio_z', 'diff_ref_wap_z', 'spread_price_book_z',
    'diff_auction_ratio_z', 'imbalance_auction_flag_ratio_z', 'excess_flag_ratio_z',
    'imbalance_book_depth_z', 'imbalance_market_book_depth_ratio_z', 'wap_rel_market_z',
    'matched_size_z', 'book_depth_z', 'imbalance_size_z',
]

new_auction_and_diff_features = [
    'diff_near_wap', 'diff_far_wap', 'diff_near_far', 'diff_near_ref',
    'auction_book_depth_ratio',
    'diff_wap_rel_market', 'diff_wap_rel_market_60',
    'diff_excess_flag_ratio', 'diff_excess_flag_ratio_60',
    'diff_ref_wap_10', 'diff_ref_wap_change_60',
    'diff_near_price', 'diff_near_price_60',
    'diff_near_wap_10', 'diff_near_wap_60'
]

extra_features = [
    'diff_liquidity_weights', 'diff_liquidity_weights_60', 'ratio_liquidity_weights',
    'market_auction_flag_ratio', 'excess_auction_flag_ratio',
    'wap_rel_market_book',
    'price_pressure', 'ask_ref_spread', 'ref_bid_spread', 'ref_spread_asymmetry'
]

features = features + z_features + new_auction_and_diff_features + extra_features

target = 'target'

params = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'device': 'gpu',
    
    'learning_rate': 0.015,        
    'n_estimators': 2500,          
    
    'num_leaves': 63,             
    'max_depth': -1,             
    'min_child_samples': 300, 
    
    'subsample': 0.7,              
    'colsample_bytree': 0.5,       
    'subsample_freq': 1,
    
    'reg_alpha': 0.1,              
    'reg_lambda': 1.0,
    
    'random_state': 42,
    'n_jobs': -1
}


if __name__ == "__main__":
    train_df = pd.read_csv('/kaggle/input/competitions/optiver-trading-at-the-close/train.csv')
    train_df_byid = train_df.sort_values(['stock_id', 'date_id', 'seconds_in_bucket'])
    train_df_byid = apply_feature_engineering(train_df_byid)

    # Validation split (80/20)
    """
    split_date = int(train_df_byid["date_id"].max())*0.8
    train_mask = train_df_byid["date_id"] <= split_date
    valid_mask = train_df_byid["date_id"] > split_date

    train_set = train_df_byid[train_mask]
    valid_set = train_df_byid[valid_mask]

    train_clean = train_set.dropna(subset=[target])
    valid_clean = valid_set.dropna(subset=[target])

    X_train, y_train = train_clean[features], train_clean[target]
    X_val, y_val = valid_clean[features], valid_clean[target]

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=100)]
    )

    y_pred = model.predict(X_val)
    # we center the predictions to preserve the zero-sum property of the target (which is tied to no-arbitrage conditions)
    y_pred = y_pred - pd.Series(y_pred * valid_clean['market_auction_depth_ratio'], index=X_val.index).groupby([valid_clean['date_id'], valid_clean['seconds_in_bucket']]).transform('sum')
    score = mean_absolute_error(y_val, y_pred)
    print(f"Validation MAE : {score:.5f}")
    """

    print("Train on 100% data...")
    train_clean = train_df_byid.dropna(subset=['target'])
    X_all, y_all = train_clean[features], train_clean['target']

    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(X_all, y_all)

    sys.path.append('/kaggle/input/competitions/optiver-trading-at-the-close/')
    import optiver2023

    env = optiver2023.make_env()
    iter_test = env.iter_test()

    # keep in memory for rolling
    history_df = train_df_byid.copy()

    print("Real-time predict...")
    for (test, revealed_targets, sample_prediction) in iter_test:
        
        # new chunk
        history_df = pd.concat([history_df, test], ignore_index=True)
        
        # feature engineering
        processed_history = apply_feature_engineering(history_df)
        
        # past chunk 
        test_processed = processed_history.tail(len(test)).copy()
        
        # Predict
        preds = final_model.predict(test_processed[features])
        
        # we center the predictions to preserve the zero-sum property of the target (which is tied to no-arbitrage conditions)
        w_auction = test_processed['market_auction_depth_ratio'].to_numpy()
        shift = pd.Series(preds * w_auction, index=test_processed.index).groupby(
            [test_processed['date_id'], test_processed['seconds_in_bucket']]
        ).transform('sum').to_numpy()
        
        preds_neutral = preds - shift
        
        # Send
        sample_prediction['target'] = preds_neutral
        env.predict(sample_prediction)

    print("Submission completed successfully!")

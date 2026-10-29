# Optiver - Trading at the Close

Solution for the Kaggle **Optiver - Trading at the Close** competition (predicting 60-second closing auction returns relative to a synthetic market index).

## Results

- **Validation MAE (80/20 temporal split):** `5.91804` (vs `6.06005` naive zero baseline, `+0.14201` gain)
- **Private Leaderboard MAE:** `5.4757` (single LightGBM model)

## Approach

### 1. Feature Engineering (`src/features.py`)
All features are built strictly from order book and auction book snapshots:
- **Price Pressure & Spreads:** `price_pressure` (`wap - mid_price`), reference price location within the bid-ask spread (`ask_ref_spread`, `ref_bid_spread`, `ref_spread_asymmetry`), and near/far auction price spreads.
- **Book & Auction Liquidity:** Order book depth (`bid_size + ask_size`), auction depth (`imbalance_size + matched_size`), and signed imbalance ratios.
- **Temporal Dynamics:** 10s (`diff(1)`), 60s (`rolling(6)` / `diff(6)`), and 300s (`rolling(30)` / `diff(30)`) rolling averages and variations per stock and date.
- **Cross-Sectional Market Features:** Stock liquidity weights (`market_book_depth_ratio`, `market_auction_depth_ratio`), synthetic market WAPs, and idiosyncratic excess imbalances relative to the market.
- **Causal Daily Z-Scores:** Rolling 20-day mean and standard deviation shifted by 1 day (`shift(1)`) to normalize regime shifts without data leakage.

### 2. Model (`src/train_and_submit.py`)
Single `LGBMRegressor` trained with L1 loss (`regression_l1`) on 92 microstructure features. For final submission, the model is trained on 100% of the training set and uses an in-memory `history_df` buffer during `iter_test` to compute rolling windows accurately in real time.

### 3. Post-Processing (Zero-Sum Centering)
Because the target measures stock return relative to a weighted market index, the cross-sectional weighted sum of targets is zero by construction. We center raw predictions at each `(date_id, seconds_in_bucket)` using `market_auction_depth_ratio` weights to preserve the zero-sum property of the target (tied to market neutrality and no-arbitrage conditions).

## Repository Structure

```text
├── notebooks/
│   └── optiver_trading_at_the_close.ipynb   # Original Kaggle notebook
├── src/
│   ├── features.py                          # Feature engineering pipeline
│   └── train_and_submit.py                  # Training, validation & Kaggle API loop
├── requirements.txt
└── README.md
```

# (French bellow...)
# Optiver - Trading at the Close

Solution for the Kaggle **Optiver - Trading at the Close** competition (predicting 60-second closing auction returns relative to a synthetic market index).

## Results

- **Validation MAE (80/20 temporal split):** `5.91804` (vs `6.06005` naive zero baseline, `+0.14201` gain)
- **Private Leaderboard MAE:** `5.4757` (single LightGBM model)

## Approach

### 1. Feature Engineering (`src/features.py`)
All features are built strictly from order book and auction book snapshots:
- **Order Book & Auction Flag Ratios:** Normalized directional imbalance ratios for the limit order book `(bid_size - ask_size) / (bid_size + ask_size)` and the auction book `(signed_imbalance_size) / (imbalance_size + matched_size)`, capture order flow direction and supply/demand asymmetry.
- **Price Pressure & Spreads:** `price_pressure` (`wap - mid_price`), reference price location within the bid-ask spread (`ask_ref_spread`, `ref_bid_spread`, `ref_spread_asymmetry`), and near/far auction price spreads.
- **Book & Auction Liquidity:** Order book depth (`bid_size + ask_size`), auction depth (`imbalance_size + matched_size`), and signed imbalance magnitudes.
- **Temporal Dynamics:** 10s (`diff(1)`), 60s (`rolling(6)` / `diff(6)`), and 300s (`rolling(30)` / `diff(30)`) rolling averages and variations per stock and date.
- **Cross-Sectional & Excess Features:** Stock liquidity weights (`market_book_depth_ratio`, `market_auction_depth_ratio`), synthetic market WAPs, and idiosyncratic excess flag ratios relative to market-wide order flow (`excess_flag_ratio`).
- **Causal Daily Z-Scores:** Rolling 20-day mean and standard deviation shifted by 1 day (`shift(1)`) to normalize regime shifts without data leakage.

### 2. Model & Potential Improvements (`src/train_and_submit.py`)
Single `LGBMRegressor` trained with L1 loss (`regression_l1`) on 92 microstructure features. For final submission, the model is trained on 100% of the training set and uses an in-memory `history_df` buffer during `iter_test` to compute rolling windows accurately in real time.

**Next Steps / Ensembling:** Performance could be further boosted by blending/stacking diverse model families—specifically combining LightGBM with XGBoost, CatBoost, and a Deep Learning architecture (e.g., 1D-CNN or Transformer on tabular sequences) to capture non-linear cross-feature interactions and smooth out single-model variance.

### 3. Post-Processing (Zero-Sum Centering)
Because the target measures stock return relative to a weighted market index, the cross-sectional weighted sum of targets is zero by construction. We center raw predictions at each `(date_id, seconds_in_bucket)` using `market_auction_depth_ratio` weights to preserve the zero-sum property of the target (tied to market neutrality and no-arbitrage conditions).

# French

Projet pour la compétition Kaggle **Optiver - Trading at the Close** (prédiction des rendements à 60 secondes des enchères de clôture par rapport à un indice synthétique).

## Résultats

- **MAE en validation (split temporel 80/20) :** `5.91804` (contre `6.06005` pour la baseline zéro, soit un gain de `+0.14201`)
- **MAE Private Leaderboard :** `5.4757` (modèle LightGBM unique)

## Démarche

### 1. Feature Engineering (`src/features.py`)
Toutes les variables sont construites à partir des données du carnet d'ordres et du carnet d'enchères :
- **Ratios d'imbalance (Flag Ratios) :** Calcul de ratios normalisés pour mesurer le déséquilibre d'asymétrie entre l'offre et la demande, à la fois sur le carnet d'ordres `(bid_size - ask_size) / (bid_size + ask_size)` et sur le carnet d'enchères `(signed_imbalance_size) / (imbalance_size + matched_size)`.
- **Pression sur les prix & Spreads :** Indicateur de pression `price_pressure` (`wap - mid_price`), positionnement du prix de référence dans le spread bid-ask (`ask_ref_spread`, `ref_bid_spread`, `ref_spread_asymmetry`) et écarts de prix d'enchères (near/far prices).
- **Profondeur de marché & Liquidité :** Profondeur du carnet d'ordres (`bid_size + ask_size`), profondeur de l'enchère (`imbalance_size + matched_size`) et volumes d'imbalance signés.
- **Dynamique temporelle :** Moyennes mobiles et variations sur 10s (`diff(1)`), 60s (`rolling(6)` / `diff(6)`) et 300s (`rolling(30)` / `diff(30)`), calculées par action et par jour.
- **Indicateurs transversaux (Cross-Sectional & Excess) :** Poids de liquidité par action (`market_book_depth_ratio`, `market_auction_depth_ratio`), WAP moyen du marché et déséquilibres spécifiques par rapport à la tendance globale du marché (`excess_flag_ratio`).
- **Z-Scores journaliers causaux :** Normalisation par moyenne et écart-type glissants sur 20 jours avec un décalage d'un jour (`shift(1)`) pour s'adapter aux changements de régime sans fuite de données (data leakage).

### 2. Modélisation & Pistes d'amélioration (`src/train_and_submit.py`)
Un modèle `LGBMRegressor` entraîné avec une fonction de perte L1 (`regression_l1`) sur 92 variables de microstructure. Pour la soumission finale, le modèle est entraîné sur 100 % des données et utilise un buffer `history_df` en mémoire pendant l'exécution d'API `iter_test` afin de recalculer les fenêtres glissantes en temps réel.

**Pistes d'amélioration / Ensembling :** La performance globale pourrait être renforcée en combinant plusieurs familles de modèles, notamment en associant LightGBM à XGBoost, CatBoost et à une architecture de Deep Learning (type 1D-CNN ou Transformer pour données tabulaires séquentielles) pour mieux capturer les interactions non linéaires et réduire la variance d'un modèle unique.

### 3. Post-traitement (Centrage Zero-Sum)
La cible mesurant le rendement relatif d'une action par rapport à un indice pondéré du marché, la somme pondérée des cibles est nulle par construction. Les prédictions brutes sont donc recentrées à chaque horodatage `(date_id, seconds_in_bucket)` en utilisant les poids `market_auction_depth_ratio` pour respecter la neutralité du marché et les conditions de non-arbitrage.

## Repository Structure

```text
├── notebooks/
│   └── optiver_trading_at_the_close.ipynb   # Notebook Kaggle d'origine
├── src/
│   ├── features.py                          # Pipeline de feature engineering
│   └── train_and_submit.py                  # Entraînement, validation & boucle API Kaggle
├── requirements.txt
└── README.md

```text
├── notebooks/
│   └── optiver_trading_at_the_close.ipynb   # Original Kaggle notebook
├── src/
│   ├── features.py                          # Feature engineering pipeline
│   └── train_and_submit.py                  # Training, validation & Kaggle API loop
├── requirements.txt
└── README.md


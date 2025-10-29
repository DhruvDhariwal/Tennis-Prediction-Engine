# Tennis Prediction Engine (Leak-Proof & Ranked) 🎾

## Overview 📖

This project implements a robust machine learning system to predict the winner of ATP tennis matches. It uses historical match data, player information, and weekly ATP rankings to train an XGBoost model.

**Key Goals & Features:**

* **🚫 Leak-Proof Design:** Features are engineered strictly using information available *before* the start of each match. This prevents data leakage and ensures the model generalizes to future, unseen matches.
* **🕰️ Chronological Processing:** All data processing, feature updates (like Elo ratings), and rolling statistics are calculated in strict chronological order.
* **📊 Comprehensive Features:** The model incorporates various predictive signals:
    * **Elo & Surface-Specific Elo:** Captures overall and surface-adjusted player strength. K-factor varies by match level (ATP, Challenger, Futures).
    * **ATP Rankings & Points:** Uses official weekly rankings via efficient "as-of" lookups.
    * **Head-to-Head (H2H):** Direct historical matchup results.
    * **Rolling Form:** Recent win rate (last N matches).
    * **Rolling Serve Stats:** Average ace rate, double fault rate, and service points won rate over recent matches.
    * **Static Attributes:** Differences in height and age (where available).
    * **Surface Encoding:** One-hot encoding for Hard, Clay, Grass, Carpet.
* **↔️ Order-Invariant Features:** Features are designed as differences (e.g., `elo_diff = player_A_elo - player_B_elo`). This ensures `P(A wins B)` is approximately equal to `1 - P(B wins A)`.
* **🚀 Efficient Ranking Lookup:** Pre-processes millions of weekly ranking rows into an optimized dictionary for near-instantaneous lookup using binary search (`bisect`).
* **⚙️ XGBoost Model:** Utilizes the efficient gradient boosting algorithm (`hist` tree method) for training.
* **📅 Clear Train/Test Split:** Uses a hard chronological split (e.g., train up to 2023, test on 2024) to rigorously evaluate performance.
* **🔮 Prediction Utility:** Provides a `predict_match` function to score future matchups using the saved model and player state.

---

## Features Used 📈

The model trains on the following features, calculated as differences between Player A and Player B for a given match:

* `rank_diff`: Difference in ATP Rank just before the match.
* `points_diff`: Difference in ATP Points just before the match.
* `elo_diff`: Difference in global Elo rating.
* `surf_elo_diff`: Difference in surface-specific Elo rating (Hard, Clay, Grass, Carpet).
* `roll_winrate_diff`: Difference in win percentage over the last `ROLL_N` matches.
* `matches_played_diff`: Difference in total matches played historically.
* `wins_diff`: Difference in total matches won historically.
* `h2h_diff`: Difference in head-to-head wins (`A wins over B` - `B wins over A`).
* `ace_rate_diff`: Difference in rolling average ace rate (aces / service points).
* `df_rate_diff`: Difference in rolling average double fault rate (DFs / service points).
* `srv_winrate_diff`: Difference in rolling average service points won rate.
* `height_diff`: Difference in player height (cm). *Currently set to 0 in prediction if not available.*
* `age_diff`: Difference in player age (years). *Currently set to 0 in prediction if not available.*
* `surface_is_hard`: 1 if surface is Hard, 0 otherwise.
* `surface_is_clay`: 1 if surface is Clay, 0 otherwise.
* `surface_is_grass`: 1 if surface is Grass, 0 otherwise.
* `surface_is_carpet`: 1 if surface is Carpet, 0 otherwise.

---

## Data Requirements 💾

This project relies on several CSV datasets, typically sourced from repositories like Jeff Sackmann's tennis_atp:

1.  **Match Data:** Yearly CSV files containing match results. The notebook expects files named like:
    * `atp_matches_YYYY.csv` (Main ATP Tour)
    * `atp_matches_qual_chall_YYYY.csv` (Qualifying & Challenger Tours)
    * `atp_matches_futures_YYYY.csv` (Futures / ITF Tour)
    * Where `YYYY` is the year (e.g., 2019, 2020...).
2.  **Player Data:** A single CSV file mapping player IDs to names, handedness, height, etc.
    * `atp_players.csv`
3.  **Ranking Data:** CSV files containing weekly ATP rankings. The notebook expects:
    * `atp_rankings_00s.csv` (Rankings from 2000-2009)
    * `atp_rankings_10s.csv` (Rankings from 2010-2019)
    * `atp_rankings_20s.csv` (Rankings from 2020-2023)
    * `atp_rankings_current.csv` (Rankings from 2024 onwards - used for testing/prediction state)
    * **Format:** Each ranking file should have columns (preferably without headers): `ranking_date`, `rank`, `player_id`, `points`.

**File Location:**

* All data files should reside within the directory specified by the `BASE_DIR` variable in the first code cell of the notebook. You **must** update this path to match your local setup.

---

## Methodology 🔬

1.  **Loading & Cleaning:** Loads match data for specified training and testing years. Normalizes column names and data types. Removes walkovers, retirements, and matches with missing essential data (player IDs, date). Drops columns that would cause data leakage (e.g., `score`, `minutes`, point-level stats).
2.  **Ranking Pre-processing:** Loads all relevant ranking files. Filters them by date. Creates an optimized lookup dictionary: `{player_id: ([sorted_dates], [(rank, points)])}`.
3.  **Feature Engineering:**
    * Iterates through matches **strictly chronologically**.
    * For each match, retrieves the **most recent ranking** for both players *before* the match date using the `bisect` module on the pre-processed lookup (O(log N) complexity).
    * Retrieves the current Elo, surface Elo, H2H record, and rolling stats history for both players from the `PlayerState`.
    * Calculates all the difference-based features listed above.
    * Stores the feature row twice: once for `(Player A, Player B, Target=1 if A won)` and once mirrored for `(Player B, Player A, Target=0 if A won)`.
    * **Post-Match Updates:** After creating the feature row, updates Elo, surface Elo, H2H counts, and rolling statistics history for both players involved.
4.  **Training:**
    * Uses the engineered features (`X_train`, `y_train`) and optional sample weights based on tournament level (`w_train`).
    * Trains an XGBoost classifier (`binary:logistic` objective, `hist` tree method).
    * Uses the test set (`X_test`, `y_test`) **only for early stopping** to prevent overfitting, mimicking a real-world validation scenario.
5.  **Evaluation:**
    * Calculates standard classification metrics on the held-out test set: Accuracy, AUC, LogLoss, Brier Score.
    * Performs an **order-invariance check** to ensure `P(A wins B) ≈ 1 - P(B wins A)`.
    * Visualizes feature importances (Gain) and model calibration (Reliability Curve).

---

## Setup & Usage 💻

**Prerequisites:**

* Python (>= 3.8 recommended)
* Required libraries:
    ```
    pandas>=1.3
    numpy>=1.20
    xgboost>=2.0
    scikit-learn>=1.0
    matplotlib>=3.5
    ```

**Installation:**

1.  Clone or download this repository/notebook.
2.  Download the required ATP datasets (matches, players, rankings) and place them in a single directory.
3.  Install the required Python libraries:
    ```bash
    pip install pandas numpy xgboost scikit-learn matplotlib
    ```

**Configuration:**

1.  **`BASE_DIR`:** Open the notebook and **update the `BASE_DIR` variable** in the first code cell to the absolute path of the directory containing your downloaded CSV data files.
2.  **`TRAIN_YEARS` / `TEST_YEARS`:** Adjust the years used for training and testing if desired. Ensure they are chronologically distinct.
3.  **`ROLL_N`:** Modify the window size for rolling statistics if needed.

**Running the Notebook:**

1.  Launch Jupyter Notebook or Jupyter Lab.
2.  Open the `Tennis_Prediction_Engine.ipynb` file.
3.  Execute the cells sequentially ("Run All"). The notebook will:
    * Load and clean the data.
    * Build the ranking lookup.
    * Construct features.
    * Train the XGBoost model.
    * Evaluate the model.
    * Save the trained model (`tennis_xgb_model.json`) and the final player state (`player_state_after_2024.pkl`).
    * Demonstrate the `predict_match` function.

---

## Prediction Function (`predict_match`) 🔮

After running the notebook once to train and save the model/state, you can use the final cells to predict future or hypothetical matches.

* The cell titled **"LOAD MODEL + PLAYER STATE..."** loads the `tennis_xgb_model.json` and `player_state_after_2024.pkl` files.
* The cell titled **"PREDICT A FUTURE MATCH..."** defines and uses the `predict_match` function.

**Usage:**

```python
# Example: Predict Djokovic vs Alcaraz on Hard court, using rankings/state as of today
result = predict_match("Novak Djokovic", "Carlos Alcaraz", surface="Hard", match_date="today")
print(result)

# Example: Predict Sinner vs De Minaur on Clay, using rankings/state as of 2025-05-15
result_clay = predict_match("Jannik Sinner", "Alex De Minaur", surface="Clay", match_date="2025-05-15")
print(result_clay)

# Players can also be specified by their ATP player ID
result_ids = predict_match(104925, 200005, surface="Grass") # Example IDs
print(result_ids)

**Inputs:**

* `player1`, `player2`: Player names (case-insensitive, first last) or integer ATP player IDs.
* `surface`: 'Hard', 'Clay', 'Grass', or 'Carpet' (defaults to 'Hard').
* `match_date`: 'today' (uses current date) or a 'YYYY-MM-DD' string to get the correct historical ranking for that date.

**Output:**

A dictionary containing:

* Player names.
* Surface and match date used.
* Predicted win probabilities (`p1_win_prob`, `p2_win_prob`).
* Implied decimal odds (`p1_decimal_odds`, `p2_decimal_odds`).

---

## Output Files 📦

Running the notebook generates two essential files in the same directory as the notebook:

1.  **`tennis_xgb_model.json`**: The trained XGBoost model saved in JSON format.
2.  **`player_state_after_2024.pkl`**: A Python pickle file containing the serialized state required for predictions:
    * `players`: Dictionary mapping `player_id` to their final `PlayerState` (Elo, surface Elo, rolling stats history).
    * `h2h`: Dictionary storing head-to-head counts.
    * `rankings`: The optimized ranking lookup dictionary.
    * `config`: Metadata like feature names, Elo constants, etc.

---

## Potential Extensions ✨

* Incorporate richer point-level or summary stats (break points saved/converted, return points won %) via rolling windows.
* Add features for travel fatigue (time zones crossed, days since last match).
* Hyperparameter tuning for XGBoost using `TimeSeriesSplit` strictly on the training data.
* Use the `atp_players.csv` more extensively to fill `height_diff` and `age_diff` reliably during prediction.
* Implement more rigorous adversarial leakage checks (e.g., training on shuffled labels).

## Acknowledgements 🙏

This project heavily relies on the comprehensive and meticulously maintained ATP tennis datasets provided by **Jeff Sackmann** ([https://github.com/JeffSackmann](https://github.com/JeffSackmann)). Special thanks are extended for making such valuable data publicly available, which was instrumental in developing and training this prediction engine.
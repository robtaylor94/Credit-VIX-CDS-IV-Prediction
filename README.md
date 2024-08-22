This is the repo counterpart to the paper 'Evaluating Credit VIX (CDS IV) Prediction Methods with Incremental Batch Learning'

- **Evaluating CDS Implied Vol Prediction Methods with Incremental Batch Learning:**  
  This empirical research project assesses the effectiveness of a custom Attention-Gated Recurrent Unit (ATTN-GRU), Support Vector Machine (SVM), and Gradient Boosting (LightGBM) in predicting an Implied Volatility Index (iTraxx/Cboe Europe Main 1-Month Volatility Index) of aggregated five-year CDS contracts on European corporate debt.

## Repository Structure
- `Feature Engineering and Selection.ipynb`: Notebook for Exploratory Data Analysis (EDA) and feature engineering.
- `Test_and_Eval.ipynb`: Notebook for model testing and evaluation.
- `features.pkl`: Preprocessed feature matrix for all timesteps up to 09/08/2024.
- `features (experimentation set).pkl`: Feature matrix for preliminary experimentation.
- `models.py`: Contains the implementations of preprocessing steps, machine learning models (Naive Estimate, SVM, LightGBM, ATTN-GRU), and shared functions.
- `utils.py`: Utility functions for time-series tests, model testing loops, and output handling.

## Citation
Robert Taylor, "Evaluating Credit VIX (CDS IV) Prediction Methods with Incremental Batch Learning,", Queen Mary University of London, Aug 2024.

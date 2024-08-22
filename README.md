This is the repo counterpart to the paper 'Evaluating Credit VIX (CDS IV) Prediction
Methods with Incremental Batch Learning'

› Evaluating CDS Implied Vol Prediction Methods with Incremental Batch Learning: This empirical research project assesses the effectiveness of a custom Attention-Gated Recurrent Unit (ATTN-GRU), Support Vector Machine (SVM), and Gradient Boosting (LightGBM) in predicting an Implied Volatility Index (iTraxx/Cboe Europe Main 1-Month Volatility Index) of aggregated five-year CDS contracts on European corporate debt.

› Feature Engineering and Selection.ipynb: EDA Notebook that performs basic Feature Engineering.
› Test_and_Eval.ipynb: Testing and Evaluation Notebook.
› features (experimentation set).pkl: Serialised feature matrix used for cusory experimentation.
› features.pkl: Serialised feature matrix including all timesteps up to 09/08/2024
› models.py: Preprocessing, Shared Functions, Naive Estimate, SVM, LightGBM and ATTN-GRU Classes.
› utils.py: Time-Series tests, Testing Loop, Output Functions.

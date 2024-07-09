import numpy as np
import pandas as pd
import tensorflow as tf
import itertools
import lightgbm as lgb
from keras.models import Model
from keras.layers import Dense, GRU, Dropout, Bidirectional, LayerNormalization, Input, Add, Conv1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer, mean_squared_error
from sklearn.svm import SVR

class DataPreparation:
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.original_dates = None

    def create_sequences(self, X, y, is_3d=False):
        # handle dates
        if isinstance(X, pd.DataFrame):
            original_dates = X.index
        else:
            original_dates = pd.RangeIndex(start=0, stop=len(X), step=1)

        # print original dimensions of X and y pre-transformation
        print(f"X dims pre-sequiencing: {X.shape}")
        print(f"y dims pre-sequencing: {y.shape}")

        X_values = np.array(X)
        y_values = y.reshape(-1, 1) if isinstance(y, np.ndarray) else np.array(y).reshape(-1, 1)
        X_with_y = np.hstack((X_values, y_values))

        # print dimensions after stacking X and y
        print(f"stacked shape: {X_with_y.shape}")

        if X_with_y.shape[0] <= self.sequence_length:
            print("not enough observations to create any sequences.")
            return pd.DataFrame(), pd.Series()  # return empty df and series

        if is_3d:
            n_sequences = X_with_y.shape[0] - self.sequence_length + 1
            X_seq = np.array([X_with_y[i:i + self.sequence_length, :-1] for i in range(n_sequences)])
            y_seq = np.array([X_with_y[i + self.sequence_length - 1, -1] for i in range(n_sequences)])

            # log the dimensions of input and output for 3D
            print(f"X sequenced dim: {X_seq.shape}")
            print(f"y dim: {y_seq.shape}")

            X_seq_df = X_seq  # direct use as array
            y_seq_df = pd.Series(y_seq, index=original_dates[self.sequence_length - 1:])
        else:
            X_seq, y_seq = [], []
            for i in range(len(X_with_y) - self.sequence_length + 1):
                X_seq.append(X_with_y[i:i + self.sequence_length, :-1].flatten())
                y_seq.append(X_with_y[i + self.sequence_length - 1, -1])

            X_seq_df = pd.DataFrame(X_seq, index=original_dates[self.sequence_length - 1:])
            y_seq_df = pd.Series(y_seq, index=original_dates[self.sequence_length - 1:])

        return X_seq_df, y_seq_df

    def scale_data(self, X, y):
        # convert X to numpy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X_values = X.values
            X_index = X.index  #pPreserve index for later use
        else:
            X_values = X
            X_index = None  # handling case when X doesn't have an index

        # dimensionality based scaling (TFT_GRU etc)
        if X_values.ndim == 3:
            n_samples, n_steps, n_features = X_values.shape
            X_reshaped = X_values.reshape(n_samples * n_steps, n_features)
            X_scaled = self.scaler_X.fit_transform(X_reshaped)
            X_scaled = X_scaled.reshape(n_samples, n_steps, n_features)
        elif X_values.ndim == 2:
            X_scaled = self.scaler_X.fit_transform(X_values)
        else:
            raise ValueError("Unsupported data shape for scaling: X must be either 2D or 3D")

        # scaling
        y_values = y.values.reshape(-1, 1) if isinstance(y, pd.Series) else y.reshape(-1, 1)
        y_scaled = self.scaler_y.fit_transform(y_values).flatten()

        # return scaled data with original index
        if isinstance(X, pd.DataFrame):
            X_scaled = pd.DataFrame(X_scaled, index=X_index)
        if isinstance(y, pd.Series):
            y_scaled = pd.Series(y_scaled, index=y.index)

        return X_scaled, y_scaled


    def inverse_transform_y(self, y):
        if isinstance(y, pd.Series):
            y_inv = self.scaler_y.inverse_transform(y.values.reshape(-1, 1)).flatten()
            return pd.Series(y_inv, index=y.index)
        else:
            return pd.Series(self.scaler_y.inverse_transform(y.reshape(-1, 1)).flatten())

    def get_forecast_date(self):
        return self.original_dates[-1] + pd.Timedelta(days=1)

class WalkForwardValidation:
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length

    def validate(self, model, X, y):
        errors = []
        n_samples = len(X)

        for i in range(self.sequence_length, n_samples + 1):
            X_train = X[:i]
            y_train = y[:i]
            X_test = X[i-1:i]
            y_test = y[i-1:i]

            yield (X_train, y_train), (X_test, y_test)

class AR_SVM:
    """autoregressive SVM, featured extensively in vol literature"""
    def __init__(self, time_steps=15, n_splits=5):
        self.time_steps = time_steps
        self.param_grid = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.2, 0.5]
        }
        self.scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        self.best_svr = None
        self.scaler = MinMaxScaler()
        self.validator = WalkForwardValidation(n_splits)

    def prepare_training_data(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y

    def fit(self, X, y):
        X_scaled, y = self.prepare_training_data(X, y)
        errors = self.validator.validate(GridSearchCV(SVR(), self.param_grid, scoring=self.scorer, cv=2, verbose=1, n_jobs=-1), X_scaled, y)
        print(f"walk-forward validation errors: {errors}")
        print(f"mean validation error: {np.mean(errors)}")

        grid_search = GridSearchCV(SVR(), self.param_grid, scoring=self.scorer, cv=2, verbose=1, n_jobs=-1)
        grid_search.fit(X_scaled, y)
        self.best_svr = grid_search.best_estimator_
        print(f"best params: {grid_search.best_params_}")

    def predict(self, X):
        if not self.best_svr:
            raise AttributeError("model has not been fitted yet.")
        X_scaled = self.scaler.transform(X)
        return self.best_svr.predict(X_scaled)

class Naive:
    def __init__(self):
        self.last_y = None  # only need to store the last observed y as prediction of y_T+1

    def fit(self, X, y):
        self.last_y = y[-1, 0] if y.ndim > 1 else y[-1]

    def predict(self):
        if self.last_y is None:
            raise ValueError("model has not been fitted yet.")
        return np.array([self.last_y])[0]

class SVM:
    def __init__(self, sequence_length=15, feature_selection=False, n_features=10):
        self.sequence_length = sequence_length
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.param_grid = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [5, 10, 20],
            'epsilon': [0.05, 0.1, 0.2],
            'gamma': ['scale', 'auto', 0.05, 0.1, 0.2]
        }
        self.scorer = make_scorer(mean_squared_error, greater_is_better=False)
        self.best_svr = None
        self.feature_selector = None
        self.data_prep = DataPreparation(self.sequence_length)
        self.validator = WalkForwardValidation(sequence_length=self.sequence_length)
        self.X_seq = None  # store the last sequence for testing or prediction

    def fit(self, X: np.ndarray, y: np.ndarray, params: dict = None):
        if params:
            self.set_params(params)

        X_seq, y_seq = self.data_prep.create_sequences(X, y, is_3d=False)
        self.X_seq = X_seq  # store the sequenced data for potential future predictions

        best_score = float('inf')
        best_params = None

        param_combinations = list(itertools.product(
            self.param_grid['kernel'],
            self.param_grid['C'],
            self.param_grid['epsilon'],
            self.param_grid['gamma']
        ))

        try:
            for kernel, C, epsilon, gamma in param_combinations:
                total_score = 0
                count = 0

                for (X_train, y_train), (X_test, y_test) in self.validator.validate(self, X_seq, y_seq):
                    if len(X_train) == 0 or len(X_test) == 0:
                        print('Empty training or testing set, skipping.')
                        continue

                    X_train_scaled, y_train_scaled = self.data_prep.scale_data(X_train, y_train)
                    X_test_scaled, _ = self.data_prep.scale_data(X_test, y_test)

                    if self.feature_selection:
                        self.feature_selector = SelectKBest(f_regression, k=self.n_features)
                        X_selected = self.feature_selector.fit_transform(X_train_scaled, y_train_scaled)
                    else:
                        X_selected = X_train_scaled

                    model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
                    model.fit(X_selected, y_train_scaled.ravel())

                    y_pred = model.predict(X_test_scaled)
                    score = mean_squared_error(y_test, y_pred)

                    total_score += score
                    count += 1

                avg_score = total_score / count if count > 0 else float('inf')

                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {'kernel': kernel, 'C': C, 'epsilon': epsilon, 'gamma': gamma}
                    self.best_svr = model  # update the best model within the parameter testing loop

            self.last_params = best_params
            print(f'Best params found: {best_params} with score {best_score}')
            if self.best_svr is not None:
                print("Model is properly fitted and ready for prediction.")
            else:
                print("Failed to fit any model.")

        except Exception as e:
            print(f"An error occurred during fit: {e}")

    def predict(self):
        if not self.best_svr:
            print("Model has not been fitted yet or failed to fit properly.")
            return np.array([np.nan])[0]  # return a single NaN

        try:
            X_seq = self.X_seq.iloc[-1].values.reshape(1, -1)
            X_seq_scaled = self.data_prep.scaler_X.transform(X_seq)

            if self.feature_selection and self.feature_selector is not None:
                X_selected = self.feature_selector.transform(X_seq_scaled)
            else:
                X_selected = X_seq_scaled

            forecast_scaled = self.best_svr.predict(X_selected)
            forecast = self.data_prep.inverse_transform_y(forecast_scaled)[0]

            return np.array([forecast])[0]  # return a single numpy value

        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.array([np.nan])[0]  # return a single NaN

class TFT_GRU:
    def __init__(self, time_steps=40, sequence_length=4):
        self.time_steps = time_steps
        self.sequence_length = sequence_length
        self.model = None
        self.data_prep = DataPreparation(self.sequence_length)
        self.validator = WalkForwardValidation(sequence_length=self.sequence_length)

    def build_model(self, input_shape, sequence_length):
        print(f"Building model with input shape: {input_shape} and sequence length: {sequence_length}")
        inputs = Input(shape=(sequence_length, input_shape))
        
        x = Conv1D(filters=64, kernel_size=2, padding='same')(inputs)
        print(f"Shape after Conv1D: {x.shape}")
        attention = self.attention_mechanism(x, head_size=16, num_heads=4)
        print(f"Shape after attention: {attention.shape}")
        
        attention = Conv1D(filters=64, kernel_size=1, padding='same')(attention)
        
        x = Add()([x, attention])
        x = LayerNormalization()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # residual connection
        residual = Conv1D(filters=64, kernel_size=1, padding='same')(inputs)
        x = Add()([x, residual])
        x = LayerNormalization()(x)
        
        x = Bidirectional(GRU(64, return_sequences=True))(x)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(GRU(32, return_sequences=False))(x)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(32, activation='selu')(x)
        output = Dense(1)(x)

        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

    def attention_mechanism(self, inputs, head_size, num_heads):
        query = Dense(head_size * num_heads)(inputs)
        key = Dense(head_size * num_heads)(inputs)
        value = Dense(head_size * num_heads)(inputs)
        query = tf.split(query, num_heads, axis=-1)
        key = tf.split(key, num_heads, axis=-1)
        value = tf.split(value, num_heads, axis=-1)

        attention_outputs = [
            tf.matmul(tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(head_size, tf.float32)), axis=-1), v)
            for q, k, v in zip(query, key, value)
        ]

        concat_attention = tf.concat(attention_outputs, axis=-1)
        return Dense(inputs.shape[-1])(concat_attention)

    def fit(self, X, y):
        X_seq, y_seq = self.data_prep.create_sequences(X, y, is_3d=True)
        print(f"Data prepared: X_seq shape {X_seq.shape}, y_seq shape {y_seq.shape}")
        self.X_seq = X_seq  # store the sequenced data for potential future predictions

        self.X_train = X_seq
        self.y_train = y_seq

        if self.model is None:
            self.build_model(X.shape[1], self.sequence_length)

        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00001)

        total_score = 0
        count = 0

        for (X_train, y_train), (X_test, y_test) in self.validator.validate(self, X_seq, y_seq):
            if len(X_train) == 0 or len(X_test) == 0:
                print('Empty training or testing set, skipping.')
                continue

            print(f"Current data types before scaling: X_train type: {type(X_train)}, y_train type: {type(y_train)}")
            print(f"Current data shapes before scaling: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

            # fit scaler on the training data
            X_train_scaled, y_train_scaled = self.data_prep.scale_data(X_train, y_train)

            # ensure X_test and y_test are numpy arrays
            X_test_reshaped = X_test.values.reshape(-1, X_test.shape[-1]) if isinstance(X_test, pd.DataFrame) else X_test.reshape(-1, X_test.shape[-1])
            y_test_reshaped = y_test.values.reshape(-1, 1) if isinstance(y_test, pd.Series) else y_test.reshape(-1, 1)

            X_test_scaled = self.data_prep.scaler_X.transform(X_test_reshaped).reshape(X_test.shape)
            y_test_scaled = self.data_prep.scaler_y.transform(y_test_reshaped).reshape(-1)

            print(f"Shapes for model fitting: X_train_scaled={X_train_scaled.shape}, y_train_scaled={y_train_scaled.shape}")
            self.model.fit(
                X_train_scaled, y_train_scaled,
                epochs=100, batch_size=64, verbose=1,
                callbacks=[early_stopping, reduce_lr]
            )

            y_pred_scaled = self.model.predict(X_test_scaled)
            y_pred = self.data_prep.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

            y_test_original = self.data_prep.scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
            score = mean_squared_error(y_test_original, y_pred)

            total_score += score
            count += 1

        avg_score = total_score / count if count > 0 else float('inf')
        print(f"Mean validation error: {avg_score}")

        # fit the model on whole chunk
        X_seq_scaled, y_seq_scaled = self.data_prep.scale_data(X_seq, y_seq)
        self.model.fit(
            X_seq_scaled, y_seq_scaled,
            epochs=100, batch_size=64, verbose=1,
            callbacks=[early_stopping, reduce_lr]
        )

    def predict(self):
        if self.model is None:
            raise AttributeError("Model has not been fitted yet.")

        X_input = self.X_train[-1:]
        print(f"Predicting with last input shape: {X_input.shape}")

        X_input_scaled = self.data_prep.scaler_X.transform(X_input.reshape(-1, X_input.shape[-1])).reshape(X_input.shape)
        print(f"Scaled input shape: {X_input_scaled.shape}")
        
        forecast_scaled = self.model.predict(X_input_scaled)
        print(f"Raw scaled forecast: {forecast_scaled}")
        
        forecast = self.data_prep.inverse_transform_y(forecast_scaled.reshape(-1, 1)).flatten()
        print(f"Final forecast: {forecast}")
        
        if np.isnan(forecast).any():
            print("Warning: NaN values in forecast")
        
        return forecast[0]  # return a single numpy value

class LGBM:
    def __init__(self, sequence_length=4):
        self.sequence_length = sequence_length
        self.model = None
        self.data_prep = DataPreparation(self.sequence_length)
        self.validator = WalkForwardValidation(sequence_length=self.sequence_length)

    def fit(self, X, y):
        X_seq, y_seq = self.data_prep.create_sequences(X, y, is_3d=False)
        print(f"Data prepared: X_seq shape {X_seq.shape}, y_seq shape {y_seq.shape}")
        self.X_seq = X_seq  # store the sequenced data for future predictions

        best_params = None
        best_score = float('inf')

        param_grid = {
            'num_leaves': [15, 17, 19,],
            'learning_rate': [0.035, 0.04],
            'max_depth': [-1],
            'min_data_in_leaf': [1, 5, 10],
            'feature_fraction': [0.9],
            'min_gain_to_split': [0.01],
            'bagging_fraction': [0.9],
            'bagging_freq': [1]
        }
        
        param_combinations = list(itertools.product(
            param_grid['num_leaves'],
            param_grid['learning_rate'],
            param_grid['max_depth'],
            param_grid['min_data_in_leaf'],
            param_grid['feature_fraction'],
            param_grid['min_gain_to_split'],
            param_grid['bagging_fraction'],
            param_grid['bagging_freq']
        ))

        for num_leaves, learning_rate, max_depth, min_data_in_leaf, feature_fraction, min_gain_to_split, bagging_fraction, bagging_freq in param_combinations:
            params = {
                'objective': 'regression',
                'num_leaves': num_leaves,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'min_data_in_leaf': min_data_in_leaf,
                'feature_fraction': feature_fraction,
                'min_gain_to_split': min_gain_to_split,
                'bagging_fraction': bagging_fraction,
                'bagging_freq': bagging_freq,
                'metric': 'rmse',
                'verbosity': -1
            }

            total_score = 0
            count = 0

            for (X_train, y_train), (X_test, y_test) in self.validator.validate(self, X_seq, y_seq):
                if len(X_train) == 0 or len(X_test) == 0:
                    print('Empty training or testing set, skipping.')
                    continue

                X_train_scaled, y_train_scaled = self.data_prep.scale_data(X_train, y_train)

                X_test_scaled = self.data_prep.scaler_X.transform(X_test.to_numpy().reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
                y_test_scaled = self.data_prep.scaler_y.transform(y_test.to_numpy().reshape(-1, 1)).reshape(-1)

                train_data = lgb.Dataset(X_train_scaled, label=y_train_scaled)
                valid_data = lgb.Dataset(X_test_scaled, label=y_test_scaled, reference=train_data)

                # train with early stopping and logging
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=500,
                    valid_sets=[valid_data],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=20),
                        lgb.log_evaluation(10)
                    ]
                )

                y_pred_scaled = model.predict(X_test_scaled, num_iteration=model.best_iteration)
                y_pred = self.data_prep.inverse_transform_y(y_pred_scaled)
                y_test_original = self.data_prep.inverse_transform_y(y_test_scaled)
                
                score = mean_squared_error(y_test_original, y_pred)

                total_score += score
                count += 1

            avg_score = total_score / count if count > 0 else float('inf')

            if avg_score < best_score:
                best_score = avg_score
                best_params = params

        print(f'Best params found: {best_params} with score {best_score}')
        
        # scale the entire dataset for final training
        X_seq_scaled, y_seq_scaled = self.data_prep.scale_data(X_seq, y_seq)

        train_data = lgb.Dataset(X_seq_scaled, label=y_seq_scaled)

        # retrain with best params on all data
        if best_params:
            self.model = lgb.train(
                best_params,
                train_data,
                num_boost_round=500,
                callbacks=[
                    lgb.log_evaluation(10)
                ]
            )

    def predict(self):
        if self.model is None:
            raise AttributeError("Model has not been fitted yet.")

        X_input = self.X_seq.iloc[-1].to_numpy().reshape(1, -1)
        print(f"Predicting with last input shape: {X_input.shape}")

        X_input_scaled = self.data_prep.scaler_X.transform(X_input)
        forecast_scaled = self.model.predict(X_input_scaled)
        forecast = self.data_prep.inverse_transform_y(forecast_scaled)
        return forecast[0]  # return a single numpy value
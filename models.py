import numpy as np
import pandas as pd
import tensorflow as tf
import itertools
import lightgbm as lgb
import math
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.svm import SVR
from tensorflow.keras.layers import Input, Conv1D, Add, LayerNormalization, Dense, Dropout, Bidirectional, GRU, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.activations import swish

def uniform_noise(features):
    noise = np.random.uniform(-0.02, 0.02, features.shape) * features
    noisy_features = features + noise
    return noisy_features

class DataPreparation:
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def create_sequences(self, X, y, is_3d=False):
        if isinstance(X, pd.DataFrame):
            original_dates = X.index
        else:
            original_dates = pd.RangeIndex(start=0, stop=len(X), step=1)

        X_values = np.array(X)
        y_values = y.reshape(-1, 1) if isinstance(y, np.ndarray) else np.array(y).reshape(-1, 1)
        X_with_y = np.hstack((X_values, y_values))

        if X_with_y.shape[0] <= self.sequence_length:
            print("Not enough observations to create any sequences.")
            return pd.DataFrame(), pd.Series()

        if is_3d:
            n_sequences = X_with_y.shape[0] - self.sequence_length + 1
            X_seq = np.array([X_with_y[i:i + self.sequence_length, :-1] for i in range(n_sequences)])
            y_seq = np.array([X_with_y[i + self.sequence_length - 1, -1] for i in range(n_sequences)])
            X_seq_df = X_seq
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
        if isinstance(X, pd.DataFrame):
            X_values = X.values
            X_index = X.index
        else:
            X_values = X
            X_index = None

        if X_values.ndim == 3:
            n_samples, n_steps, n_features = X_values.shape
            X_reshaped = X_values.reshape(n_samples * n_steps, n_features)
            X_scaled = self.scaler_X.fit_transform(X_reshaped)
            X_scaled = X_scaled.reshape(n_samples, n_steps, n_features)
        elif X_values.ndim == 2:
            X_scaled = self.scaler_X.fit_transform(X_values)
        else:
            raise ValueError("Unsupported data shape for scaling: X must be either 2D or 3D")

        y_values = y.values.reshape(-1, 1) if isinstance(y, pd.Series) else y.reshape(-1, 1)
        y_scaled = self.scaler_y.fit_transform(y_values).flatten()

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

    def validate(self, X, y):
        n_samples = len(X)
        for i in range(self.sequence_length, n_samples):
            X_train, y_train = X[:i], y[:i]
            X_test, y_test = X[i:i+1], y[i:i+1]
            yield (X_train, y_train), (X_test, y_test)

class SVM:
    def __init__(self, sequence_length=15, feature_selection=False, n_features=10, reg=1.0):
        self.sequence_length = sequence_length
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.reg = reg
        self.param_grid = {
            'kernel': ['poly', 'rbf', 'sigmoid'],
            'epsilon': [0.05, 0.1, 0.15],
            'gamma': ['scale', 'auto', 0.1, 0.15, 0.2]
        }
        self.best_svr = None
        self.feature_selector = None
        self.data_prep = DataPreparation(self.sequence_length)
        self.validator = WalkForwardValidation(self.sequence_length)
        self.X_seq = None
        self.best_params = None

    def fit(self, X, y, params=None):
        if params:
            self.set_params(params)
        X_seq, y_seq = self.data_prep.create_sequences(X, y, is_3d=False)
        self.X_seq = X_seq
        best_score = float('inf')
        param_combinations = list(itertools.product(
            self.param_grid['kernel'],
            self.param_grid['epsilon'],
            self.param_grid['gamma']
        ))

        if self.feature_selection:
            self.feature_selector = SelectKBest(f_regression, k=self.n_features)
            self.feature_selector.fit(self.data_prep.scale_data(X_seq, y_seq)[0], y_seq)

        best_params = None
        for kernel, epsilon, gamma in param_combinations:
            total_score = 0
            count = 0
            for (X_train, y_train), (X_test, y_test) in self.validator.validate(X_seq, y_seq):
                if len(X_train) == 0 or len(X_test) == 0:
                    continue

                X_train_noised = uniform_noise(X_train)
                X_train_scaled, y_train_scaled = self.data_prep.scale_data(X_train_noised, y_train)

                # No noise application for test data
                X_test_scaled, _ = self.data_prep.scale_data(X_test, y_test)

                if self.feature_selection:
                    X_selected = self.feature_selector.transform(X_train_scaled)
                else:
                    X_selected = X_train_scaled

                model = SVR(kernel=kernel, epsilon=epsilon, gamma=gamma, C=self.reg)
                model.fit(X_selected, y_train_scaled.ravel())
                y_pred = model.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, y_pred)
                total_score += mae
                count += 1
                print(f"Kernel: {kernel}, Epsilon: {epsilon}, Gamma: {gamma}, Step {count} Validation MAE: {mae}")

            avg_mae = total_score / count if count > 0 else float('inf')
            if avg_mae < best_score:
                best_score = avg_mae
                best_params = {'kernel': kernel, 'epsilon': epsilon, 'gamma': gamma, 'C': self.reg}
                self.best_svr = model

        self.best_params = best_params
        print(f'Best params found: {self.best_params} with MAE {best_score}')

        # final train with best model
        if self.best_params:
            X_seq_noised = uniform_noise(X_seq)
            X_seq_scaled, y_seq_scaled = self.data_prep.scale_data(X_seq_noised, y_seq)
            if self.feature_selection:
                X_seq_scaled = self.feature_selector.transform(X_seq_scaled)
            self.best_svr.fit(X_seq_scaled, y_seq_scaled)

    def predict(self):
        if not self.best_svr:
            raise ValueError("Model has not been fitted yet or failed to fit properly.")
        X_seq = self.X_seq.iloc[-1].values.reshape(1, -1)
        X_seq_scaled = self.data_prep.scaler_X.transform(X_seq)
        X_selected = self.feature_selector.transform(X_seq_scaled) if self.feature_selection and self.feature_selector is not None else X_seq_scaled
        forecast_scaled = self.best_svr.predict(X_selected)
        forecast = self.data_prep.inverse_transform_y(forecast_scaled)[0]
        return np.array([forecast])[0]

class LGBM:
    def __init__(self, sequence_length=4, lr=0.005, reg=0.01):
        "default params were generally the best in training/OoS"
        self.sequence_length = sequence_length
        self.learning_rate = lr
        self.regularization_param = reg
        self.model = None
        self.data_prep = DataPreparation(self.sequence_length)
        self.validator = WalkForwardValidation(self.sequence_length)
        self.best_params = None

    def fit(self, X, y):
        X_seq, y_seq = self.data_prep.create_sequences(X, y, is_3d=False)
        self.X_seq = X_seq
        best_score = float('inf')
        self.param_grid = {
            'num_leaves': [75, 100, 125],
            'learning_rate': [self.learning_rate],
            'max_depth': [-1, 5, 10],
            'min_data_in_leaf': [10, 20, 30],
            'feature_fraction': [0.4, 0.5, 0.6],
            'min_gain_to_split': [self.regularization_param],
            'bagging_fraction': [0.9],
            'bagging_freq': [1]
        }

        param_combinations = list(itertools.product(
            self.param_grid['num_leaves'],
            self.param_grid['learning_rate'],
            self.param_grid['max_depth'],
            self.param_grid['min_data_in_leaf'],
            self.param_grid['feature_fraction'],
            self.param_grid['min_gain_to_split'],
            self.param_grid['bagging_fraction'],
            self.param_grid['bagging_freq']
        ))

        best_params = None
        for params in param_combinations:
            lgb_params = dict(zip(self.param_grid.keys(), params))
            total_score = 0
            count = 0

            for (X_train, y_train), (X_test, y_test) in self.validator.validate(X_seq, y_seq):
                if len(X_train) == 0 or len(X_test) == 0:
                    continue

                # apply uniform noise to the training data
                X_train_noised = uniform_noise(X_train)
                X_train_scaled, y_train_scaled = self.data_prep.scale_data(X_train_noised, y_train)

                X_test_scaled, y_test_scaled = self.data_prep.scale_data(X_test, y_test)

                train_data = lgb.Dataset(X_train_scaled, label=y_train_scaled)
                valid_data = lgb.Dataset(X_test_scaled, label=y_test_scaled, reference=train_data)

                model = lgb.train(
                    lgb_params,
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

                mae = mean_absolute_error(y_test_original, y_pred)
                total_score += mae
                count += 1
                print(f"Params: {lgb_params}, Step {count} Validation MAE: {mae}")

            avg_mae = total_score / count if count > 0 else float('inf')
            if avg_mae < best_score:
                best_score = avg_mae
                best_params = lgb_params
                self.model = model

        self.best_params = best_params
        print(f'Best params found: {self.best_params} with MAE {best_score}')

        # final train with best model
        if self.best_params:
            X_seq_noised = uniform_noise(X_seq)
            X_seq_scaled, y_seq_scaled = self.data_prep.scale_data(X_seq_noised, y_seq)
            train_data = lgb.Dataset(X_seq_scaled, label=y_seq_scaled)
            self.model = lgb.train(
                self.best_params,
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
        X_input_scaled = self.data_prep.scaler_X.transform(X_input)
        forecast_scaled = self.model.predict(X_input_scaled)
        forecast = self.data_prep.inverse_transform_y(forecast_scaled)
        return forecast[0]

class TFT_GRU:
    def __init__(self, time_steps=40, sequence_length=4, learning_rate=0.07, clipnorm=0.1):
        self.time_steps = time_steps
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.clipnorm = clipnorm
        self.model = None
        self.data_prep = DataPreparation(self.sequence_length)
        self.validator = WalkForwardValidation(self.sequence_length)

    def build_model(self, input_shape, sequence_length):
        inputs = Input(shape=(sequence_length, input_shape))

        adjusted_inputs = Conv1D(filters=64, kernel_size=2, padding='same', activation=swish)(inputs)
        attention = self.attention_layer(adjusted_inputs, head_size=16, num_heads=4)
        attention = Conv1D(filters=64, kernel_size=2, padding='same', activation=swish)(attention)
        x = Add()([adjusted_inputs, attention])
        x = LayerNormalization()(x)
        x = Dense(64, activation=swish)(x)
        x = Dropout(0.2)(x)
        x = Add()([adjusted_inputs, x])
        x = LayerNormalization()(x)

        x = Bidirectional(GRU(64, activation=swish, recurrent_activation='sigmoid', return_sequences=True))(x)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(GRU(32, activation=swish, recurrent_activation='sigmoid', return_sequences=False))(x)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)

        x = Dense(32, activation=swish)(x)
        output = Dense(1)(x)

        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=self.clipnorm)
        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile(optimizer=optimizer, loss='mean_absolute_error')

    def attention_layer(self, inputs, head_size, num_heads):
        def multi_head_attention(query_key_value):
            query, key, value = query_key_value
            query = tf.split(query, num_heads, axis=-1)
            key = tf.split(key, num_heads, axis=-1)
            value = tf.split(value, num_heads, axis=-1)

            attention_outputs = []
            for q, k, v in zip(query, key, value):
                attention_score = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(head_size, tf.float32))
                attention_weights = tf.nn.softmax(attention_score, axis=-1)
                attention_output = tf.matmul(attention_weights, v)
                attention_outputs.append(attention_output)

            concat_attention = tf.concat(attention_outputs, axis=-1)
            return concat_attention

        query = Dense(head_size * num_heads)(inputs)
        key = Dense(head_size * num_heads)(inputs)
        value = Dense(head_size * num_heads)(inputs)

        attention_output = Lambda(multi_head_attention)([query, key, value])
        output = Dense(inputs.shape[-1], activation=swish)(attention_output)

        return output

    def fit(self, X, y):
        X_seq, y_seq = self.data_prep.create_sequences(X, y, is_3d=True)
        self.X_seq = np.array(X_seq)
        if self.model is None:
            self.build_model(X_seq.shape[-1], self.sequence_length)

        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=0.001)

        for i, ((X_train, y_train), (X_test, y_test)) in enumerate(self.validator.validate(X_seq, y_seq)):
            if len(X_train) == 0 or len(X_test) == 0:
                continue

            # apply noise to the training data
            X_train_noised = uniform_noise(X_train)
            X_train_scaled, y_train_scaled = self.data_prep.scale_data(X_train_noised, y_train)
            
            X_test_scaled, y_test_scaled = self.data_prep.scale_data(X_test, y_test)

            self.model.fit(
                X_train_scaled, y_train_scaled,
                validation_data=(X_test_scaled, y_test_scaled),
                epochs=15, batch_size=64, verbose=1,
                callbacks=[early_stopping, reduce_lr]
            )

            y_pred_scaled = self.model.predict(X_test_scaled)
            y_pred = self.data_prep.inverse_transform_y(pd.Series(y_pred_scaled.flatten())).to_numpy().flatten()
            y_test_original = y_test.to_numpy().flatten()
            mae = mean_absolute_error(y_test_original, y_pred)
            print(f"Step {i+1} Validation MAE: {mae}")

        # final train
        X_seq_noised = uniform_noise(X_seq)
        X_seq_scaled, y_seq_scaled = self.data_prep.scale_data(X_seq_noised, y_seq)
        self.model.fit(
            X_seq_scaled, y_seq_scaled,
            epochs=15, batch_size=64, verbose=1,
            callbacks=[early_stopping, reduce_lr]
        )

    def predict(self):
        if self.model is None:
            raise AttributeError("Model has not been fitted yet.")
        X_input = self.X_seq[-1].reshape(1, self.sequence_length, -1)
        X_input_scaled = self.data_prep.scaler_X.transform(X_input.reshape(-1, X_input.shape[-1])).reshape(1, self.sequence_length, -1)
        forecast_scaled = self.model.predict(X_input_scaled)
        forecast = self.data_prep.inverse_transform_y(forecast_scaled)
        return forecast[0]

class Naive:
    def __init__(self, **kwargs):
        self.last_y = None

    def fit(self, X, y):
        self.last_y = y[-1, 0] if y.ndim > 1 else y[-1]

    def predict(self):
        if self.last_y is None:
            raise ValueError("Model has not been fitted yet.")
        return np.array([self.last_y])[0]

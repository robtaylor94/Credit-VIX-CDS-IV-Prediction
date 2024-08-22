import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import itertools
import lightgbm as lgb
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR

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
            'num_leaves': [50, 100, 150],
            'learning_rate': [self.learning_rate],
            'max_depth': [-1, 5, 10],
            'min_data_in_leaf': [5, 15, 25],
            'feature_fraction': [0.4, 0.6, 0.8],
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

class ATTN_GRU(nn.Module):
    def __init__(self, sequence_length=5, learning_rate=0.07, clipnorm=0.1):
        super(ATTN_GRU, self).__init__()
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.clipnorm = clipnorm
        self.model_built = False
        self.data_prep = DataPreparation(self.sequence_length)
        self.validator = WalkForwardValidation(self.sequence_length)
        self.optimizer = None

    def build_model(self, input_shape):
        self.conv1 = nn.Conv1d(in_channels=input_shape, out_channels=64, kernel_size=2, padding='same')
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2, padding='same')

        self.query_dense = nn.Linear(64, 64)
        self.key_dense = nn.Linear(64, 64)
        self.value_dense = nn.Linear(64, 64)
        self.output_dense = nn.Linear(64, 64)

        self.layernorm1 = nn.LayerNorm(64)
        self.layernorm2 = nn.LayerNorm(64)

        self.fc1 = nn.Linear(64, 64)
        self.dropout1 = nn.Dropout(0.2)

        self.gru1 = nn.GRU(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        self.layernorm3 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(0.2)
        self.gru2 = nn.GRU(input_size=128, hidden_size=32, batch_first=True, bidirectional=True)
        self.layernorm4 = nn.LayerNorm(64)
        self.dropout3 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        self.model_built = True

    def attention_layer(self, inputs, head_size, num_heads):
        batch_size, seq_len, _ = inputs.size()

        # Apply linear projections to generate query, key, and value
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Split the query, key, and value into multiple heads
        query = query.view(batch_size, seq_len, num_heads, head_size).permute(0, 2, 1, 3)
        key = key.view(batch_size, seq_len, num_heads, head_size).permute(0, 2, 1, 3)
        value = value.view(batch_size, seq_len, num_heads, head_size).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_size, dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)

        concat_attention = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)

        # Apply a final linear layer with Swish activation to match the original input size
        output = self.output_dense(concat_attention)
        output = F.silu(output)  # Swish activation function

        return output

    def forward(self, x):
        if not self.model_built:
            self.build_model(input_shape=x.shape[2])

        x = x.transpose(1, 2)
        x1 = F.silu(self.conv1(x)).transpose(1, 2)

        # attention mechanism
        attention_output = self.attention_layer(x1, head_size=16, num_heads=4)
        attention_output = F.silu(self.conv2(attention_output.transpose(1, 2)).transpose(1, 2))

        # residual connections and normalisation
        x1 = self.add_and_norm(x1, attention_output, self.layernorm1)
        x2 = F.silu(self.fc1(x1))
        x2 = self.dropout1(x2)
        x2 = self.add_and_norm(x1, x2, self.layernorm2)

        x3, _ = self.gru1(x2)
        x3 = self.layernorm3(x3)
        x3 = self.dropout2(x3)
        x4, _ = self.gru2(x3)
        x4 = self.layernorm4(x4)
        x4 = self.dropout3(x4)

        x5 = F.silu(self.fc2(x4))
        output = self.fc3(x5)

        return output[:, -1, :].squeeze(-1)

    def add_and_norm(self, x1, x2, layernorm):
        x = x1 + x2
        x = layernorm(x)
        return x

    def fit(self, X, y, batch_size=32, epochs=50, patience=4):
        X_seq, y_seq = self.data_prep.create_sequences(X, y, is_3d=True)
        self.X_seq = X_seq

        if X_seq.size == 0 or y_seq.size == 0:
            print("Sequence creation failed. Not enough data points.")
            return

        if not self.model_built:
            self.build_model(input_shape=X_seq.shape[-1])

        for i, ((X_train, y_train), (X_test, y_test)) in enumerate(self.validator.validate(X_seq, y_seq)):
            if len(X_train) == 0 or len(X_test) == 0:
                continue

            X_train_noised = uniform_noise(X_train)
            X_train_scaled, y_train_scaled = self.data_prep.scale_data(X_train_noised, y_train)
            X_test_scaled, y_test_scaled = self.data_prep.scale_data(X_test, y_test)

            y_train_scaled = y_train_scaled.to_numpy().flatten()
            y_test_scaled = y_test_scaled.to_numpy().flatten()

            train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train_scaled, dtype=torch.float32))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            best_val_loss = np.inf
            patience_counter = 0

            self.train()
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    self.optimizer.zero_grad()
                    output = self.forward(batch_X)
                    loss = nn.functional.l1_loss(output, batch_y)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), self.clipnorm)
                    self.optimizer.step()
                    epoch_loss += loss.item()

                print(f'Step {i+1} Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}')

                # Validation
                self.eval()
                with torch.no_grad():
                    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
                    y_pred_scaled = self.forward(X_test_tensor).cpu().numpy().flatten()
                    y_pred = self.data_prep.inverse_transform_y(pd.Series(y_pred_scaled)).to_numpy().flatten()
                    y_test_original = y_test.to_numpy().flatten()
                    val_mae = mean_absolute_error(y_test_original, y_pred)

                print(f"Step {i+1} Validation MAE: {val_mae}")

                if val_mae < best_val_loss:
                    best_val_loss = val_mae
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1} for Step {i+1}. Best Validation MAE: {best_val_loss}")
                    break

        # final training with all data
        X_seq_noised = uniform_noise(X_seq)
        X_seq_scaled, y_seq_scaled = self.data_prep.scale_data(X_seq_noised, y_seq)
        y_seq_scaled = y_seq_scaled.to_numpy().flatten()
        train_dataset = TensorDataset(torch.tensor(X_seq_scaled, dtype=torch.float32), torch.tensor(y_seq_scaled, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_final_loss = best_val_loss
        patience_counter = 0

        self.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                output = self.forward(batch_X)
                loss = nn.functional.l1_loss(output, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.clipnorm)
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f'Final Training Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}')

            if epoch_loss < best_final_loss:
                best_final_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping during final training at epoch {epoch+1}. Best Final Loss: {best_final_loss/len(train_loader)}")
                break

    def predict(self):
        if not self.model_built:
            raise AttributeError("Model has not been fitted yet.")
        X_input = self.X_seq[-1].reshape(1, self.sequence_length, -1)
        X_input_scaled = self.data_prep.scaler_X.transform(X_input.reshape(-1, X_input.shape[-1])).reshape(1, self.sequence_length, -1)
        X_input_tensor = torch.tensor(X_input_scaled, dtype=torch.float32)

        with torch.no_grad():
            forecast_scaled = self.forward(X_input_tensor).cpu().numpy()

        forecast = self.data_prep.inverse_transform_y(pd.Series(forecast_scaled.flatten())).to_numpy()

        return forecast[0]

class Naive:
    def __init__(self, **kwargs):
        self.last_y = None

    def fit(self, X, y):
        self.last_y = y[-1, 0] if y.ndim > 1 else y[-1]

    def predict(self, X=None):
        if self.last_y is None:
            raise ValueError("Model has not been fitted yet.")
        return np.array([self.last_y])[0]

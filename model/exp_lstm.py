from torch import nn
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os


class LSTMModel(nn.Module):
    def __init__(self, num_lstm_layers, input_dim, hidden_dim, output_dim, device, **kwargs):
        super(LSTMModel, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.lstm_layers = nn.LSTM(input_dim, hidden_dim, num_lstm_layers, batch_first=True,
                                   dropout=0.2 if num_lstm_layers > 1 else 0)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.device = device

    def forward(self, x):
        H0 = torch.randn(self.num_lstm_layers, x.size(0), self.hidden_dim, device=self.device).requires_grad_()
        C0 = torch.randn(self.num_lstm_layers, x.size(0), self.hidden_dim, device=self.device).requires_grad_()
        out, (H, C) = self.lstm_layers(x, (H0.detach(), C0.detach()))
        out = self.bn1(out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.bn3(out)
        out = self.fc3(out)
        return out


class LSTMBaseline(nn.Module):
    def __init__(self, num_lstm_layers, input_dim, hidden_dim, output_dim, device, **kwargs):
        super(LSTMBaseline, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.lstm_layers = nn.LSTM(input_dim, hidden_dim, num_lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.device = device

    def forward(self, x):
        H0 = torch.randn(self.num_lstm_layers, x.size(0), self.hidden_dim, device=self.device).requires_grad_()
        C0 = torch.randn(self.num_lstm_layers, x.size(0), self.hidden_dim, device=self.device).requires_grad_()
        out, (H, C) = self.lstm_layers(x, (H0.detach(), C0.detach()))
        output = self.fc(out[:, -1, :])
        return output


class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


class Exp_LSTM():
    def __init__(self, args, **kwargs):
        super(Exp_LSTM, self).__init__(**kwargs)
        self.args = args
        self.num_lstm_layers = args.num_lstm_layers
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.device = torch.device(self.args.device)
        self.root_path = args.root_path
        self.stock_name = args.stock_name
        self.num_steps = args.num_steps
        self.batch_size = args.batch_size
        if args.feature_list == "all_data":
            self.feature_list = ['open', 'high', 'low', 'close', 'volume', 'sma_5', 'sma_10', 'sma_30', 'dema', 'kama',
                                 'wma',
                                 'trima', 'adx', 'rsi', 'cci', 'macd[0]', 'mom', 'roc', 'ad', 'obv', 'score',
                                 'neg_count',
                                 'neu_count', 'pos_count']  # 24
        elif args.feature_list == "all_data without senti":
            self.feature_list = ['open', 'high', 'low', 'close', 'volume', 'sma_5', 'sma_10', 'sma_30', 'dema', 'kama',
                                 'wma',
                                 'trima', 'adx', 'rsi', 'cci', 'macd[0]', 'mom', 'roc', 'ad', 'obv']  # 20
        elif args.feature_list == "all_data without TA":
            self.feature_list = ['open', 'high', 'low', 'close', 'volume', 'score', 'neg_count',
                                 'neu_count', 'pos_count']  # 9
        elif args.feature_list == "raw_data":
            self.feature_list = ['open', 'high', 'low', 'close', 'volume']  # 5
        self.input_dim = len(self.feature_list)
        self.learning_rate = args.learning_rate
        self.num_epochs = args.num_epochs
        self.checkpoints_path = args.checkpoints_path
        self.result_path = args.result_path

    def _build_model(self):
        return LSTMBaseline(self.num_lstm_layers, self.input_dim, self.hidden_dim, self.output_dim, self.device)

    def _get_date(self):
        df = pd.read_csv(os.path.join(self.root_path, f"{self.stock_name}"))
        scaler = StandardScaler()
        scaler.fit(df[self.feature_list].values)
        scaled_data = scaler.transform(df[self.feature_list].values)
        num_datas = len(scaled_data)
        X, Y = [], []
        for i in range(num_datas):
            end_idx = i + self.num_steps
            if end_idx > num_datas - 1:
                break
            seq_x, seq_y = scaled_data[i: end_idx], scaled_data[end_idx, 3]
            X.append(seq_x)
            Y.append(seq_y)

        num_seqs = len(X)
        X_train = X[:int(0.7 * num_seqs)]
        Y_train = Y[:int(0.7 * num_seqs)]
        X_train = torch.tensor(X_train).float()
        Y_train = torch.tensor(Y_train).float().view(-1, 1)
        dataset_train = TensorDataset(X_train, Y_train)
        data_loader_train = DataLoader(dataset_train, self.batch_size, shuffle=True)

        X_valid = X[int(0.7 * num_seqs): int((0.7 + 0.1) * num_seqs)]
        Y_valid = Y[int(0.7 * num_seqs): int((0.7 + 0.1) * num_seqs)]
        X_valid = torch.tensor(X_valid).float()
        Y_valid = torch.tensor(Y_valid).float().view(-1, 1)
        dataset_valid = TensorDataset(X_valid, Y_valid)
        data_loader_valid = DataLoader(dataset_valid, self.batch_size, shuffle=True)

        X_test = torch.tensor(X).float()
        Y_test = torch.tensor(Y).float().view(-1, 1)
        dataset_test = TensorDataset(X_test, Y_test)
        data_loader_test = DataLoader(dataset_test, self.batch_size, shuffle=False)
        st = self.num_steps
        ed = len(dataset_test)
        test_time_array = np.array(df["date"])[st: st + ed]

        return data_loader_train, data_loader_valid, data_loader_test, test_time_array

    @staticmethod
    def calc_correct(pred_tensor, x_tensor, y_tensor):
        pred_list = pred_tensor.tolist()
        x_lists = x_tensor.tolist()
        y_list = y_tensor.tolist()
        pred_label_list = []
        truth_label_list = []
        for x_list, pred, y in zip(x_lists, pred_list, y_list):
            if y > x_list[-1]:
                truth_label_list.append(0)
            elif y == x_list[-1]:
                truth_label_list.append(1)
            elif y < x_list[-1]:
                truth_label_list.append(2)

            if pred > x_list[-1]:
                pred_label_list.append(0)
            elif pred == x_list[-1]:
                pred_label_list.append(1)
            elif pred < x_list[-1]:
                pred_label_list.append(2)
        correct = 0
        for pred_label, truth_label in zip(pred_label_list, truth_label_list):
            if pred_label == truth_label:
                correct += 1
        return correct

    def train(self, store_flag: str):
        folder_path = os.path.join(self.checkpoints_path, store_flag)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        model = self._build_model()
        model = model.to(self.device)
        data_loader_train, _, _, _ = self._get_date()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        for epoch in range(self.num_epochs):

            model.train()
            train_loss_sum = 0
            for x_batch, y_batch in data_loader_train:
                if isinstance(x_batch, list):
                    x_batch = [x_1.to(self.device) for x_1 in x_batch]
                else:
                    x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch.view(-1, 1))
                train_loss_sum += loss.item()
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}, Train Loss: {train_loss_sum}')
        torch.save(model.state_dict(), folder_path + "/checkpoints.pth")

    def test(self, store_flag: str):
        ckpt_path = os.path.join(self.checkpoints_path, store_flag) + "/checkpoints.pth"
        folder_path = os.path.join(self.result_path, store_flag)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        model = self._build_model()
        model.load_state_dict(torch.load(ckpt_path))
        model = model.to(self.device)
        _, _, data_loader_test, test_time_array = self._get_date()
        criterion = torch.nn.MSELoss()
        preds, trues = [], []
        test_loss_sum = 0
        correct_preds = 0
        total_preds = 0
        for x_batch, y_batch in data_loader_test:
            if isinstance(x_batch, list):
                x_batch = [x_1.to(self.device) for x_1 in x_batch]
            else:
                x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            y_pred = model(x_batch)
            correct_preds += self.calc_correct(y_pred, x_batch, y_batch)
            total_preds += y_pred.size(0)
            loss = criterion(y_pred, y_batch.view(-1, 1))
            preds.extend(y_pred.cpu().detach().tolist())
            trues.extend(y_batch.cpu().detach().tolist())
            test_loss_sum += loss.item()
        preds = np.array(preds)
        trues = np.array(trues)
        np.save(folder_path + "/pred.npy", preds)
        np.save(folder_path + "/true.npy", trues)
        np.save(folder_path + "/time.npy", test_time_array)

    def eval(self, store_flag: str):
        folder_path = os.path.join(self.result_path, store_flag)
        pred_path = folder_path + "/pred.npy"
        true_path = folder_path + "/true.npy"

        preds = np.load(pred_path)
        trues = np.load(true_path)
        pred_value, true_value = [], []
        print(preds.shape, trues.shape)
        for i in range(preds.shape[0]):
            # for j in range(data2.shape[1]):
            true_value.append(trues[i][0])
            pred_value.append(preds[i][0])
        plt.figure(figsize=(20, 12))
        plt.plot(true_value, label="true")
        plt.plot(pred_value, label="pred")
        plt.axvline(x=0.7 * len(true_value))
        plt.legend()
        plt.show()

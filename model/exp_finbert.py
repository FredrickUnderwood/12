import numpy as np
from sklearn.metrics import mean_squared_error
from transformers import BertTokenizer, BertForSequenceClassification, pipeline, BertModel
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd


class FinBERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, stock_name, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stock_name = stock_name

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        prompt = f"Please analyze news headline: '{text}' influence on {self.stock_name}"

        encoding = self.tokenizer.encode_plus(
            prompt,
            add_special_tokens=True,
            max_length=int(self.max_len),
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(labels, dtype=torch.float)
        }


class FinBERT(torch.nn.Module):
    def __init__(self, base_model, dropout_rate=0.1):
        super(FinBERT, self).__init__()
        self.bert = base_model
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.regression_head = torch.nn.Linear(self.bert.config.hidden_size, 1)  # For volatility prediction

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        volatility_pred = self.regression_head(pooled_output).squeeze(-1)

        return volatility_pred


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


class Exp_FinBERT():
    def __init__(self, stock_name):
        self.sentiment_data_path = "../data/news_data/sentiment_data.csv"
        self.raw_data_path = "../data/raw_data/"
        self.preprocessed_raw_data_path = "../data/preprocessed_raw_data/"
        self.finbert_finetune_data_path = "../data/sentiment_finetune_data/"
        self.model_config_path = "./finbert_model"
        self.result_path = "./results/"
        self.stock_name = stock_name
        self.device = f"cuda:0"
        self.max_len = 128
        self.batch_size = 16
        self.num_epochs = 3
        self.finbert_checkpoints = "../finbert_checkpoints/"

    @staticmethod
    def convert_to_datetime(date_str, time_str):
        if time_str.startswith('24'):
            time_str = '00' + time_str[2:]
            date_str = pd.to_datetime(date_str) + pd.Timedelta(days=1)
            datetime_str = f"{date_str.strftime('%Y/%m/%d')} {time_str}"
        else:
            datetime_str = f"{date_str} {time_str}"

        datetime_obj = pd.to_datetime(datetime_str, format="%Y/%m/%d %H:%M:%S")
        return datetime_obj

    def raw_data_preprocessing(self):
        # df_sentiment_data = pd.read_csv(self.sentiment_data_path, encoding="utf-8")
        df_raw_data = pd.read_csv(os.path.join(self.raw_data_path, self.stock_name))

        if not os.path.exists(self.preprocessed_raw_data_path):
            os.makedirs(self.preprocessed_raw_data_path)

        list_price = list(df_raw_data["close"])
        scaler = StandardScaler()
        scaler.fit(np.array(list_price))
        scaled_list_price = scaler.transform(np.array(list_price))
        list_date = list(df_raw_data["date"])

        thirtyMinutes_vol = []
        sixtyMinutes_vol = []
        twoHours_vol = []
        fourHours_vol = []

        thirtyMinute_status = []
        sixtyMinute_status = []
        twoHours_status = []
        fourHours_status = []

        vol_lists = [thirtyMinutes_vol, sixtyMinutes_vol, twoHours_vol, fourHours_vol]
        status_lists = [thirtyMinute_status, sixtyMinute_status, twoHours_status, fourHours_status]
        time_range_list = [30, 60, 120, 240]
        max_time_range = max(time_range_list)

        for i, (vol_list, status_list) in enumerate(zip(vol_lists, status_lists)):
            for index in range(len(scaled_list_price)):
                if (index + max_time_range < len(scaled_list_price)):
                    vol_raw = scaled_list_price[index + time_range_list[i]] / scaled_list_price[index] - 1.0
                    vol = round(vol_raw, 6)
                    if vol > 0:
                        status = 1
                    elif vol == 0:
                        status = 0
                    else:
                        status = -1
                    vol_list.append(vol)
                    status_list.append(status)

        scaled_list_price = scaled_list_price[:len(thirtyMinutes_vol)]
        list_date = list_date[:len(thirtyMinutes_vol)]
        df_preprocessed_raw_data = pd.DataFrame({
            "date": list_date,
            "price": scaled_list_price,
            "thirtyMinutes_vol": thirtyMinutes_vol,
            "sixtyMinutes_vol": sixtyMinutes_vol,
            "twoHours_vol": twoHours_vol,
            "fourHours_vol": fourHours_vol,
            "thirtyMinute_status": thirtyMinute_status,
            "sixtyMinute_status": sixtyMinute_status,
            "twoHours_status": twoHours_status,
            "fourHours_status": fourHours_status
        })
        df_preprocessed_raw_data.to_csv(self.preprocessed_raw_data_path + self.stock_name, index=False)

    def finetune_data_preprocessing(self):
        if not os.path.exists(self.finbert_finetune_data_path):
            os.makedirs(self.finbert_finetune_data_path)
        df_preprocessed_raw_data = pd.read_csv(self.preprocessed_raw_data_path + self.stock_name, encoding="utf-8")
        df_sentiment_data = pd.read_csv(self.sentiment_data_path, encoding="utf-8")
        # df_preprocessed_raw_data = df_preprocessed_raw_data.drop("price", axis=1)
        df_finbert_finetune_data = pd.merge(df_sentiment_data, df_preprocessed_raw_data, on="date")
        df_finbert_finetune_data.to_csv(self.finbert_finetune_data_path + self.stock_name, index=False)

    def finetune(self, store_flag, split_date):
        df_finbert_finetune_data = pd.read_csv(self.finbert_finetune_data_path + self.stock_name, encoding="utf-8")
        df_finbert_finetune_data_train = df_finbert_finetune_data[df_finbert_finetune_data["date"] < split_date]
        df_finbert_finetune_data_valid = df_finbert_finetune_data[df_finbert_finetune_data["date"] >= split_date]
        print(str(len(df_finbert_finetune_data_train) / len(df_finbert_finetune_data)) + " data for train.")
        label_list = df_finbert_finetune_data.columns[6:7]
        for label in label_list:
            if "vol" in label:
                base_model = BertModel.from_pretrained("../model/finbert_model")
                model = FinBERT(base_model)
                model.to(self.device)
                tokenizer = BertTokenizer.from_pretrained("../model/finbert_model/tokenizer_config.json")
                optimizer = AdamW(model.parameters(), lr=2e-5)
                loss_fn = torch.nn.MSELoss()

                train_texts = df_finbert_finetune_data_train["headline"].tolist()
                train_labels = df_finbert_finetune_data_train[label].tolist()

                valid_texts = df_finbert_finetune_data_valid["headline"].tolist()
                valid_labels = df_finbert_finetune_data_valid[label].tolist()

                texts = df_finbert_finetune_data["headline"].tolist()
                labels = df_finbert_finetune_data[label].tolist()

                train_dataset = FinBERTDataset(train_texts, train_labels, tokenizer,
                                               self.stock_name.replace(".csv", ""), self.max_len)
                valid_dataset = FinBERTDataset(valid_texts, valid_labels, tokenizer,
                                               self.stock_name.replace(".csv", ""), self.max_len)
                test_dataset = FinBERTDataset(texts, labels, tokenizer,
                                              self.stock_name.replace(".csv", ""), self.max_len)
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

                for epoch in range(self.num_epochs):
                    model.train()
                    losses = []
                    for batch in train_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['label'].to(self.device)
                        optimizer.zero_grad()

                        pred = model(input_ids=input_ids, attention_mask=attention_mask)
                        loss = loss_fn(pred, labels)

                        losses.append(loss.item())

                        loss.backward()
                        optimizer.step()
                    avg_loss = sum(losses) / len(losses)
                    print(f"Train: label: {label}, epoch: {epoch}, avg_loss: {avg_loss}")

                    model.eval()
                    losses = []
                    valid_label = []
                    valid_pred = []

                    with torch.no_grad():
                        for batch in valid_loader:
                            input_ids = batch['input_ids'].to(self.device)
                            attention_mask = batch['attention_mask'].to(self.device)
                            labels = batch['label'].to(self.device)

                            preds = model(input_ids=input_ids, attention_mask=attention_mask)
                            loss = loss_fn(preds, labels)
                            losses.append(loss.item())

                            valid_label.extend(labels.cpu().numpy())
                            valid_pred.extend(preds.cpu().numpy())

                    mse = mean_squared_error(valid_label, valid_pred)
                    transformed_valid_label = [1 if x > 0.001 else (-1 if x < -0.001 else 0) for x in valid_label]
                    transformed_valid_pred = [1 if x > 0.001 else (-1 if x < -0.001 else 0) for x in valid_pred]
                    correct = 0
                    for i in range(len(transformed_valid_label)):
                        if transformed_valid_label[i] == transformed_valid_pred[i]:
                            correct += 1
                    acc = round(correct / len(transformed_valid_label), 4)
                    print(f"Valid correction is {acc}")

                    print(f"Valid: label: {label}, epoch: {epoch}, mse: {mse}")
                if not os.path.exists(self.finbert_checkpoints):
                    os.makedirs(self.finbert_checkpoints)
                torch.save(model.state_dict(),
                           self.finbert_checkpoints + store_flag + "_" + self.stock_name.replace(".csv",
                                                                                                 "") + "_" + label + ".pth")

                model.eval()
                test_label = []
                test_pred = []
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)

                    preds = model(input_ids=input_ids, attention_mask=attention_mask)

                    test_label.extend(labels.detach().cpu().numpy())
                    test_pred.extend(preds.detach().cpu().numpy())
                mse = mean_squared_error(test_label, test_pred)
                transformed_test_label = [1 if x > 0.001 else (-1 if x < -0.001 else 0) for x in test_label]
                transformed_test_pred = [1 if x > 0.001 else (-1 if x < -0.001 else 0) for x in test_pred]
                correct = 0
                for i in range(len(transformed_test_label)):
                    if transformed_test_label[i] == transformed_test_pred[i]:
                        correct += 1
                acc = round(correct / len(transformed_test_label), 4)
                print(f"Test correction is {acc}")
                print(f"Test: label: {label}, mse: {mse}")

    def predict(self, store_flag, best_label):
        df_finbert_finetune_data = pd.read_csv(self.finbert_finetune_data_path + self.stock_name, encoding="utf-8")
        time_range_list = [30, 60, 120, 240]
        vol_list = list(df_finbert_finetune_data.columns[3:7])
        label_index = vol_list.index(best_label)
        best_time_range = time_range_list[label_index]

        texts = df_finbert_finetune_data["headline"].tolist()
        dates = df_finbert_finetune_data["date"].tolist()
        prices = df_finbert_finetune_data["price"].tolist()
        labels = df_finbert_finetune_data[best_label].tolist()
        scaler = StandardScaler()
        scaler.fit(np.array(prices))
        scaled_prices = list(scaler.transform(np.array(prices)))

        base_model = BertModel.from_pretrained("../model/finbert_model")
        model = FinBERT(base_model)
        model_dict_path = self.finbert_checkpoints + store_flag + "_" + self.stock_name.replace(".csv",
                                                                                                "") + "_" + best_label + ".pth"
        model.load_state_dict(torch.load(model_dict_path))
        model.to(self.device)
        tokenizer = BertTokenizer.from_pretrained("../model/finbert_model/tokenizer_config.json")

        test_dataset = FinBERTDataset(texts, labels, tokenizer,
                                      self.stock_name.replace(".csv", ""), self.max_len)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        model.eval()
        test_label, test_pred = [], []
        for batch in test_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            preds = model(input_ids=input_ids, attention_mask=attention_mask)

            test_label.extend(labels.detach().cpu().numpy())
            test_pred.extend(preds.detach().cpu().numpy())

        for i in range(len(test_pred)):
            test_pred[i] = (test_pred[i] + 1) * scaled_prices[i]
        finbert_result_path = self.result_path + store_flag
        if not os.path.exists(finbert_result_path):
            os.makedirs(finbert_result_path)
        finbert_pred_path = self.result_path + store_flag + "/finbert_pred.csv"
        df_finbert_pred = pd.DataFrame({
            "date": dates,
            "finbert_pred": test_pred
        })
        df_finbert_pred_mean = df_finbert_pred.groupby("date", as_index=False)["finbert_pred"].mean()
        new_date = df_finbert_pred_mean["date"].tolist()
        new_finbert_pred = df_finbert_pred_mean["finbert_pred"].tolist()
        new_date = new_date[best_time_range:]
        new_finbert_pred = new_finbert_pred[:len(new_date)]
        df_finbert_pred_final = pd.DataFrame({
            "date": new_date,
            "finbert_pred": new_finbert_pred
        })
        df_finbert_pred_final.to_csv(finbert_pred_path, index=False)

        plt.plot(scaled_prices, label="True")
        plt.plot(test_pred, label="Pred")
        plt.legend()
        plt.show()

    def finbert_pipeline(self, store_flag, split_date, best_label):
        self.finetune(store_flag=store_flag, split_date=split_date)
        self.predict(store_flag=store_flag, best_label=best_label)

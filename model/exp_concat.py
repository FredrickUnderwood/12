import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class Exp_Concat():
    def __init__(self):
        self.result_path = "./results"

    @staticmethod
    def find_longest_common_substring(list1, list2):
        max_length = 0
        start_index = 0
        end_index = 0

        # 遍历第一个列表中的所有子串
        for i in range(len(list1)):
            for j in range(i, len(list1)):
                sub_list = list1[i:j + 1]
                if sub_list in [list2[k:k + len(sub_list)] for k in range(len(list2) - len(sub_list) + 1)]:
                    if len(sub_list) > max_length:
                        max_length = len(sub_list)
                        start_index = i
                        end_index = j 

        return start_index, end_index

    def _get_date(self, informer_store_flag, lstm_store_flag, finbert_store_flag):

        informer_preds_path = self.result_path + "/" + informer_store_flag + "/" + "pred.npy"
        informer_trues_path = self.result_path + "/" + informer_store_flag + "/" + "true.npy"
        informer_times_path = self.result_path + "/" + informer_store_flag + "/" + "time.npy"

        lstm_preds_path = self.result_path + "/" + lstm_store_flag + "/" + "pred.npy"
        lstm_trues_path = self.result_path + "/" + lstm_store_flag + "/" + "true.npy"
        lstm_times_path = self.result_path + "/" + lstm_store_flag + "/" + "time.npy"

        informer_preds = np.load(informer_preds_path)
        informer_trues = np.load(informer_trues_path)
        informer_times = np.load(informer_times_path, allow_pickle=True)

        lstm_preds = np.load(lstm_preds_path)
        lstm_trues = np.load(lstm_trues_path)
        lstm_times = np.load(lstm_times_path, allow_pickle=True)

        informer_preds_values = []
        informer_trues_values = []
        informer_times_values = []

        for i in range(informer_preds.shape[0]):
            informer_preds_values.append(informer_preds[i][0][0])
            informer_trues_values.append(informer_trues[i][0][0])
            informer_times_values.append(informer_times[i])
        df_informer = pd.DataFrame({
            "date": informer_times_values,
            "true": informer_trues_values,
            "informer_pred": informer_preds_values
        })


        lstm_preds_values = []
        lstm_trues_values = []
        lstm_times_values = []

        for i in range(lstm_preds.shape[0]):
            lstm_preds_values.append(lstm_preds[i][0])
            lstm_trues_values.append(lstm_trues[i][0])
            lstm_times_values.append(lstm_times[i])
        df_lstm = pd.DataFrame({
            "date": lstm_times_values,
            "lstm_pred": lstm_preds_values
        })
        df_finbert = pd.read_csv(self.result_path + "/" + finbert_store_flag + "/finbert_pred.csv")

        df_concat = pd.merge(df_informer, df_lstm, on="date")
        df_concat = pd.merge(df_concat, df_finbert, on="date", how="left")
        df_mean_values = df_concat[["informer_pred", "lstm_pred"]].mean(axis=1)
        df_concat["finbert_pred"] = df_concat["finbert_pred"].fillna(df_mean_values)

        return np.array(df_concat["true"]), np.array(df_concat["informer_pred"]), np.array(df_concat["lstm_pred"]), np.array(df_concat["finbert_pred"])

    def evaluation(self, informer_store_flag, lstm_store_flag, finbert_store_flag, concat_store_flag, stock_name):
        trues_values, informer_preds_values, lstm_preds_values, finbert_preds_values = self._get_date(informer_store_flag, lstm_store_flag, finbert_store_flag)
        X = np.vstack((informer_preds_values, lstm_preds_values, finbert_preds_values)).T
        y = trues_values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        comb_model = LinearRegression()
        comb_model.fit(X_train, y_train)
        y_pred = comb_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        final_forecast = comb_model.predict(np.vstack((informer_preds_values, lstm_preds_values, finbert_preds_values)).T)
        print(f"ConcatModel's MSE: {mean_squared_error(final_forecast, trues_values)}")
        print(f"LSTM's MSE: {mean_squared_error(lstm_preds_values, trues_values)}")
        print(f"Informer's MSE: {mean_squared_error(informer_preds_values, trues_values)}")
        print(f"FinBERT's MSE: {mean_squared_error(finbert_preds_values, trues_values)}")

        concat_result_path = self.result_path + "/" + concat_store_flag + "/"
        if not os.path.exists(concat_result_path):
            os.makedirs(concat_result_path)

        plt.figure(figsize=(30, 18))
        plt.title("Prediction Data")
        st = int(0.7 * len(final_forecast))
        plt.plot(trues_values[st:], label="TrueValue")
        plt.plot(final_forecast[st:], label="ConcatModel_Pred")
        plt.plot(lstm_preds_values[st:], label="LSTM_Pred")
        plt.plot(informer_preds_values[st:], label="Informer_Pred")
        # plt.plot(finbert_preds_values[st:], label="FinBERT_Pred")
        plt.legend()
        plt.savefig(concat_result_path + f"{stock_name}_preds.png")
        plt.show()

        plt.figure(figsize=(30, 18))
        plt.title("All Data")
        plt.plot(trues_values, label="TrueValue")
        plt.plot(final_forecast, label="ConcatModel_Pred")
        plt.plot(lstm_preds_values, label="LSTM_Pred")
        plt.plot(informer_preds_values, label="Informer_Pred")
        # plt.plot(finbert_preds_values, label="FinBERT_Pred")
        plt.axvline(x=0.7 * len(final_forecast), color='red', linestyle='--', label="train/test", linewidth=3)
        plt.legend()
        plt.savefig(concat_result_path + f"{stock_name}_all.png")
        plt.show()

        concat_log_path = concat_result_path + "log.txt"
        if not os.path.exists(concat_log_path):
            with open(concat_log_path, 'a') as f:
                f.write("|Stock_name|ConcatModel's MSE|LSTM's MSE|FinBERT's MSE|\n")
                f.write(f"|{stock_name}|{mean_squared_error(final_forecast, trues_values)}|{mean_squared_error(lstm_preds_values, trues_values)}|{mean_squared_error(informer_preds_values, trues_values)}|{mean_squared_error(finbert_preds_values, trues_values)}|\n")
        else:
            with open(concat_log_path, 'a') as f:
                f.write(f"|{stock_name}|{mean_squared_error(final_forecast, trues_values)}|{mean_squared_error(lstm_preds_values, trues_values)}|{mean_squared_error(informer_preds_values, trues_values)}|{mean_squared_error(finbert_preds_values, trues_values)}|\n")







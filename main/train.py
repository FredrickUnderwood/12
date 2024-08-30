import pandas as pd

from model.Informer2020.exp.exp_informer import Exp_Informer
from model.exp_concat import Exp_Concat
from model.exp_lstm import Exp_LSTM
from model.exp_finbert import Exp_FinBERT
import os
import json
import argparse
import torch
import time



def get_model_config(model_type="informer"):
    config_root = "../config/"
    with open(os.path.join(config_root, "config.json"), "r") as file:
        model_config = json.load(file)
    if model_type == "informer":
        return model_config["Informer_config"]
    elif model_type == "lstm":
        return model_config["LSTM_config"]
    else:
        return model_config["Default"]

def get_model(model_type, stock_name):
    model = None
    if model_type == "informer":
        model_config = get_model_config(model_type)
        parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')
        for key, value in model_config.items():
            type_ = eval(value["type"])
            if "default" and "action" in value:
                parser.add_argument(f"--{key}", default=value["default"],action=value["action"], help=value["help"])
            elif "default" in value:
                parser.add_argument(f"--{key}", type=type_, default=value["default"],
                                    help=value["help"])
            elif "action" in value:
                parser.add_argument(f"--{key}", action=value["action"],
                                    help=value["help"])
            else:
                parser.add_argument(f"--{key}", type=type_, nargs=value["nargs"],
                                    help=value["help"])
        parser.set_defaults(data_path=stock_name + ".csv")
        args = parser.parse_args()
        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
        if args.use_gpu and args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]
        data_parser = {
            'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
            'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
            'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
            'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
            'WTH': {'data': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
            'ECL': {'data': 'ECL.csv', 'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
            'Solar': {'data': 'solar_AL.csv', 'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1]},
        }
        if args.data in data_parser.keys():
            data_info = data_parser[args.data]
            args.data_path = data_info['data']
            args.target = data_info['T']
            args.enc_in, args.dec_in, args.c_out = data_info[args.features]

        args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ', '').split(',')]
        args.detail_freq = args.freq
        args.freq = args.freq[-1:]

        Exp = Exp_Informer
        model = Exp(args)
    elif model_type == "lstm":
        model_config = get_model_config("lstm")
        parser = argparse.ArgumentParser(description='LSTM')
        for key, value in model_config.items():
            type_ = eval(value["type"])
            parser.add_argument(f"--{key}", type=type_, default=value["default"])
        parser.set_defaults(stock_name=stock_name + ".csv")
        args = parser.parse_args()
        Exp = Exp_LSTM
        model = Exp(args)
    elif model_type == "concat":
        Exp = Exp_Concat
        model = Exp()
    elif model_type == "finbert":
        Exp = Exp_FinBERT
        model = Exp(stock_name + '.csv')
    return model


if __name__ == "__main__":
    stock_names = ['C', 'CSCO', 'DWDP', 'GM', 'HPQ', 'IBM', 'INTC', 'JPM', 'KFT', 'KO', 'MCD',
                   'MMM', 'NKE', 'PGP', 'RTX', 'T', 'VZ']
    test_round = "0"
    model_type_1 = "informer"
    model_type_2 = "lstm"
    model_type_3 = "concat"
    model_type_4 = "finbert"


    for stock_name in stock_names:
        informer_store_flag = model_type_1 + "_test_" + test_round
        lstm_store_flag = model_type_2 + "_test_" + test_round
        concat_store_flag = model_type_3 + "_test_" + test_round
        finbert_store_flag = model_type_4 + "_test_" + test_round

        # prevent data leak in finbert training p
        train_ratio = 0.7
        raw_data_path = f"../data/raw_data/{stock_name}.csv"
        df = pd.read_csv(raw_data_path)
        split_date = df.iloc[int(train_ratio * len(df)), 0]


        init_time = time.time()
        infomer_model = get_model("informer", stock_name)
        infomer_model.train(setting=informer_store_flag)
        infomer_model.test(setting=informer_store_flag)
        infomer_model.eval(setting=informer_store_flag)
        torch.cuda.empty_cache()
        print(f"It costs {time.time() - init_time} seconds to finish {informer_store_flag}.")

        init_time = time.time()
        lstm_model = get_model("lstm", stock_name)
        lstm_model.train(store_flag=lstm_store_flag)
        lstm_model.test(store_flag=lstm_store_flag)
        lstm_model.eval(store_flag=lstm_store_flag)
        torch.cuda.empty_cache()
        print(f"It costs {time.time() - init_time} seconds to finish {lstm_store_flag}.")

        init_time = time.time()
        model = get_model("finbert", stock_name)
        model.finetune(finbert_store_flag, split_date=split_date)
        model.predict(finbert_store_flag, best_label="fourHours_vol")
        torch.cuda.empty_cache()
        print(f"It costs {time.time() - init_time} seconds to finish {finbert_store_flag}.")

        init_time = time.time()
        concat_model = get_model("concat", None)
        concat_model.evaluation(informer_store_flag=informer_store_flag, lstm_store_flag=lstm_store_flag, finbert_store_flag=finbert_store_flag, concat_store_flag=concat_store_flag, stock_name=stock_name)
        torch.cuda.empty_cache()
        print(f"It costs {time.time() - init_time} seconds to finish {concat_store_flag}.")


    # Test
    # for stock_name in stock_names:
    #     informer_store_flag = model_type_1 + "_test_" + test_round
    #     lstm_store_flag = model_type_2 + "_test_" + test_round
    #     concat_store_flag = model_type_3 + "_test_" + test_round
    #     finbert_store_flag = model_type_4 + "_test_" + test_round
    #     train_ratio = 0.7
    #     raw_data_path = f"../data/raw_data/{stock_name}.csv"
    #     df = pd.read_csv(raw_data_path)
    #     split_date = df.iloc[int(train_ratio * len(df)), 0]
    #     model = get_model("finbert", stock_name)
    #     # model.finetune(finbert_store_flag, split_date=split_date)
    #     model.predict(finbert_store_flag, best_label="twoHours_vol")
    #     concat_model = get_model("concat", None)
    #     concat_model.evaluation(informer_store_flag=informer_store_flag, lstm_store_flag=lstm_store_flag, finbert_store_flag=finbert_store_flag, concat_store_flag=concat_store_flag, stock_name=stock_name)
    #     break



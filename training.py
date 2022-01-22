import numpy as np
import pandas as pd

from data import preprocessing, validation
from optuna_training import train_lightgbm, train_catboost, train_svm

from utils import seed_everything, get_timestamp

import os, logging, argparse

TELEGRAM_SEND = True


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--n_splits', default=5, type=int)
    parser.add_argument('--n_trials', default=20, type=int)
    parser.add_argument('--modelname', default="svm", type=str)

    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    seed_everything(args.seed)

    # Get foldername
    foldername = f"{args.modelname}-{get_timestamp()}"
    os.mkdir(f"models/{foldername}")

    logging.basicConfig(filename=f'models/{foldername}/training.log', encoding='utf-8', level=logging.DEBUG)
    logging.info(f"STARTING TIME {foldername}")
    logging.info(f"ARGUMENTS {args} \n")

    train_df, num_columns, cat_columns, target = preprocessing(normalize_num="standard")

    if args.modelname == "lgbm":
        train_lightgbm(train_df, num_columns, cat_columns, target, validation, foldername, args)
    elif args.modelname == "cb":
        train_catboost(train_df, num_columns, cat_columns, target, validation, foldername, args)
    elif args.modelname == "svm":
        train_svm(train_df, num_columns, target, validation, foldername, args)

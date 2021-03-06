import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, scale
from sklearn.model_selection import StratifiedKFold

from target_encoder import smoothing_target_encoder


def preprocessing(encode_cat = "label",
                  normalize_num = None):
    train_df = pd.read_csv("data/train.csv")

    num_columns = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", 
                   "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
                   "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]

    target = "Cover_Type"

    # Reversing one-hot encoding
    train_df["Wilderness_Area"] = train_df[[f"Wilderness_Area{i}" for i in range(1, 5)]].idxmax(axis=1)
    train_df["Soil_Type"] = train_df[[f"Soil_Type{i}" for i in range(1, 41)]].idxmax(axis=1)

    cat_columns = ["Wilderness_Area", "Soil_Type"]

    # Feature engineering goes here

    # Cat encodings
    if encode_cat == "label":
        for column in cat_columns:
            le = LabelEncoder()
            le.fit(train_df[column])
            train_df[column] = le.transform(train_df[column]).astype(int)
    elif encode_cat == "target":
        # Will need 1-hot encoded target if target is multiclass
        onehot_target = pd.get_dummies(train_df[target])
        onehot_target_columns = list(onehot_target.columns)
        train_df[onehot_target_columns] = onehot_target

        for column in cat_columns:
            for target_column in onehot_target_columns:
                new_column = f"{column}_{target_column}"
                train_df[new_column] = smoothing_target_encoder(train_df, column, target_column, weight = 300)
                num_columns.append(new_column)

        train_df.drop(onehot_target_columns, axis = 1, inplace = True)
        cat_columns = []

    # Convert num to float
    for column in num_columns:
        train_df[column] = train_df[column].astype(float)

    # Normalizing
    if normalize_num == "standard":
        for column in num_columns:
            train_df[column] = scale(train_df[column]).astype(float)

    # Final sanity check for everything
    if encode_cat == "label":
        for column in cat_columns:
            assert train_df[column].dtype == 'int', (column, train_df[column].dtype)

    for column in num_columns:
        assert train_df[column].dtype == 'float', (column, train_df[column].dtype)

    return train_df, num_columns, cat_columns, target


def validation(n_splits, seed):
    train_df = pd.read_csv("data/train.csv")
    labels = train_df["Cover_Type"]

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = splitter.split(labels, labels)
    return splits
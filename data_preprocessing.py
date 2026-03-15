# data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def load_and_preprocess_data(train_path, test_path=None, min_samples=2):
    train = pd.read_csv(train_path)
    train.columns = train.columns.str.strip()

    if test_path:
        test = pd.read_csv(test_path)
    else:
        train, test = train_test_split(
            train, test_size=0.2, stratify=train['label'], random_state=42
        )

    train, val = train_test_split(
        train, test_size=0.25, stratify=train['label'], random_state=42
    )

    # Remove rare diseases
    valid = train['Disease_name'].value_counts()
    valid = valid[valid >= min_samples].index

    train = train[train['Disease_name'].isin(valid)]
    val = val[val['Disease_name'].isin(valid)]
    test = test[test['Disease_name'].isin(valid)]

    disease_encoder = LabelEncoder()
    context_encoder = LabelEncoder()

    train['Disease_label'] = disease_encoder.fit_transform(train['Disease_name'])
    val['Disease_label'] = disease_encoder.transform(val['Disease_name'])
    test['Disease_label'] = disease_encoder.transform(test['Disease_name'])

    train['Context_label'] = context_encoder.fit_transform(train['context'])
    val['Context_label'] = context_encoder.transform(val['context'])
    test['Context_label'] = context_encoder.transform(test['context'])

    os.makedirs("saved_models", exist_ok=True)
    pickle.dump(disease_encoder, open("saved_models/disease_encoder.pkl", "wb"))
    pickle.dump(context_encoder, open("saved_models/context_encoder.pkl", "wb"))

    return train, val, test, disease_encoder, context_encoder

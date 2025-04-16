import pandas as pd
import numpy as np

def Handle_Duplicated_Values(data):
    return data.drop_duplicates()


def Handle_Null_Values(data, threshold=0.5):
    data = data.copy()
    null_percent = data.isnull().mean()

    # Drop columns with > threshold missing values
    to_drop = null_percent[null_percent > threshold].index.tolist()
    data.drop(columns=to_drop, inplace=True)

    for col in data.columns:
        if data[col].isnull().sum() == 0:
            continue

        if data[col].dtype == 'object':
            # Fill categorical with mode
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            # Use IQR to assess skewness
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            skewed = (data[col] > upper).mean() > 0.1 or (data[col] < lower).mean() > 0.1

            if skewed:
                # Skewed → use median
                data[col] = data[col].fillna(data[col].median())
            else:
                # Not skewed → use mean
                data[col] = data[col].fillna(data[col].mean())

    return data



def handle_object_columns(data):
    for col in data.select_dtypes(include='object').columns:
        sample = data[col].dropna().head(10).astype(str)

        try:
            pd.to_datetime(sample, errors='raise')
            data[col] = pd.to_datetime(data[col], errors='coerce')
            continue
        except:
            pass

        try:
            pd.to_numeric(sample, errors='raise')
            data[col] = pd.to_numeric(data[col], errors='coerce')
            continue
        except:
            pass

    return data


def preprocess(data):
    data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")

    if data.isnull().sum().any():
        data = Handle_Null_Values(data)

    if data.duplicated().any():
        data = Handle_Duplicated_Values(data)

    data = handle_object_columns(data)

    return data


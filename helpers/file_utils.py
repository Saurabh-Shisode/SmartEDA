import pandas as pd

def read_csv_safely(file_path):
    encodings_to_try = [None, 'utf-8', 'ISO-8859-1']
    last_exception = None

    for enc in encodings_to_try:
        try:
            if enc:
                return pd.read_csv(file_path, encoding=enc)
            else:
                return pd.read_csv(file_path)  # default (no encoding)
        except Exception as e:
            last_exception = e

    raise RuntimeError(f"❌ Failed to read CSV file with tried encodings: {encodings_to_try}. Last error: {last_exception}")

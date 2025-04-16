from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from joblib import dump
import pandas as pd
import numpy as np
import joblib


# --- Feature Engineering ---
def feature_engineering(df, target_col=None, unique_ratio_threshold=0.6, bin_threshold=0.05):
    df = df.copy()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)

    for col in numeric_cols:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > unique_ratio_threshold:
            near_zero_frac = (df[col] < df[col].quantile(0.05)).mean()

            if near_zero_frac > bin_threshold:
                df[f"is_{col}"] = (df[col] >= df[col].median()).astype(int)
                df.drop(columns=[col], inplace=True)
            else:
                df[f"{col}_bin"] = pd.qcut(df[col], q=4, labels=False, duplicates='drop')
                df.drop(columns=[col], inplace=True)
    return df

# --- Preprocessing ---
def remove_outliers_iqr(data, multiplier=1.5):
    numeric_cols = data.select_dtypes(include='number').columns
    mask = pd.Series(True, index=data.index)
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        mask &= data[col].between(lower, upper)
    return data[mask].reset_index(drop=True)

def preprocess_for_modeling(df, target_var):
    df = df.copy()
    df = remove_outliers_iqr(df)
    df = feature_engineering(df, target_col=target_var)

    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    df = df.dropna()
    X = df.drop(columns=[target_var])
    y = df[target_var]

    numeric_cols = X.select_dtypes(include='number').columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    joblib.dump(scaler, 'static/scaler.pkl')

    return X, y

# --- Evaluation ---
def evaluate_model(model, X_test, y_test, task_type):
    y_pred = model.predict(X_test)
    if task_type == 'classification':
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'labels': sorted(list(set(y_test)))
        }
    else:
        return {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }

# --- Training ---
def train_with_grid_search(model, param_grid, X, y, cv=5, scoring='accuracy'):
    grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_, grid.best_score_

# --- Main Run ---
def run_all_models(X, y, task_type='classification', models_to_run=None):
    results = {}
    best_model_obj = None
    best_score = -np.inf
    best_model_name = None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if task_type == 'classification':
        models = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [51, 101, 151, 199],
                    'max_depth': [None, 10, 20, 25],
                    'min_samples_split': [2, 5, 7],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 2, 10],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'SVM': {
                'model': SVC(),
                'params': {
                    'C': [0.01, 0.1, 1, 2, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            }
        }
        scoring = 'f1_weighted'
    else:  # regression
        models = {
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [51, 99, 151],
                    'max_depth': [None, 10, 20, 25],
                    'min_samples_split': [2, 5, 7],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'Linear Regression': {
                'model': LinearRegression(),
                'params': {}
            },
            'SVR': {
                'model': SVR(),
                'params': {
                    'C': [0.01, 0.1, 1, 2, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            }
        }
        scoring = 'r2'

    for name, config in models.items():
        if models_to_run and name not in models_to_run:
            continue

        best_model, best_params, best_score_cv = train_with_grid_search(config['model'], config['params'], X_train, y_train, scoring=scoring)
        metrics = evaluate_model(best_model, X_test, y_test, task_type)

        results[name] = {
            'best_params': best_params,
            'cross_val_score': best_score_cv,
            'test_metrics': metrics
        }

        key_score = metrics['f1'] if task_type == 'classification' else metrics['r2']
        if key_score > best_score:
            best_score = key_score
            best_model_name = name
            best_model_obj = best_model

    if best_model_obj:
        dump(best_model_obj, 'static/model.pkl')

    return results

def run_all_classification_models(X, y, models_to_run=None):
    return run_all_models(X, y, task_type='classification', models_to_run=models_to_run)

def run_all_regression_models(X, y, models_to_run=None):
    return run_all_models(X, y, task_type='regression', models_to_run=models_to_run)

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
import matplotlib.pyplot as plt

import io
import base64
import numpy as np
import seaborn as sns

# ---------------- UI SIDE (ZAHRA) ----------------
# 1. Backend sends dataframe columns to UI
# 2. UI shows dropdown with all columns, user selects target
# 3. UI sends selected target column name back to backend
# --------------------------------------------------


def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df


def get_dataset_info(df):
    return {
        "columns": df.columns.tolist(),
        "preview": df.head(10).to_dict(orient="records")
    }


# ------ Target validation ------
def validate_target_column(df, target_col):
    # Target must exist
    if target_col not in df.columns:
        return False, "Selected target column does not exist in the dataset."

    # Target must NOT be a unique ID / name
    if df[target_col].dtype == object and df[target_col].nunique() == df.shape[0]:
        return False, "Target cannot be a unique identifier or name column."

    # Target with too many text categories is invalid
    if df[target_col].dtype == object and df[target_col].nunique() > 20:
        return False, "Target has too many categories — cannot be predicted."

    # Target must have at least 2 distinct classes
    if df[target_col].nunique() < 2:
        return False, "Target column must have at least two classes."

    return True, "Target column is valid."


# ==================================================
# PROBLEM TYPE + TARGET ENCODING
# ==================================================

def detect_problem_type(df, target):
    """Detect whether the ML problem is regression or classification."""
    if pd.api.types.is_numeric_dtype(df[target]):
        return "regression"
    else:
        return "classification"


def encode_target_if_categorical(df, target):
    """
    Encode target if it's categorical and return:
    - modified df
    - LabelEncoder (or None)
    - detected problem_type
    """
    ptype = detect_problem_type(df, target)
    encoder = None

    if ptype == "classification":
        encoder = LabelEncoder()
        df[target] = encoder.fit_transform(df[target])

    return df, encoder, ptype


# ==================================================
# FEATURE ENGINEERING
# ==================================================

def feature_engineering(df, target):
    """
    Basic automatic feature engineering on X (all columns except target).

    - One-hot encode categorical features (low/medium cardinality).
    - Keep numeric features as they are.
    - Create simple pairwise interaction features between numeric columns
      (only if there aren't too many numeric columns).

    Returns:
        df_fe: dataframe with engineered features + original target column.
    """
    df = df.copy()

    # Separate X and y
    X = df.drop(columns=[target])
    y = df[target]

    # Identify numeric and categorical predictors
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # --- 1) One-hot encode categorical features (if any) ---
    # Only for reasonably small-cardinality columns to avoid explosion
    low_cardinality_cats = []
    for col in categorical_cols:
        if X[col].nunique() <= 20:  # threshold can be adjusted
            low_cardinality_cats.append(col)
        # if too many unique categories, we simply drop it
        # (or you can keep it raw if you prefer)
    if low_cardinality_cats:
        X = pd.get_dummies(X, columns=low_cardinality_cats, drop_first=True)

    # Recompute numeric_cols after one-hot encoding (all are numeric now)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # --- 2) Pairwise interaction features between numeric columns ---
    # To avoid too many new features, only do this if numeric_cols is small
    max_numeric_for_interactions = 6  # safety limit
    if len(numeric_cols) > 1 and len(numeric_cols) <= max_numeric_for_interactions:
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                c1 = numeric_cols[i]
                c2 = numeric_cols[j]
                inter_name = f"{c1}_x_{c2}"
                X[inter_name] = X[c1] * X[c2]

    # Re-attach target at the end
    df_fe = pd.concat([X, y], axis=1)

    return df_fe


# ==================================================
# TRAIN/TEST SPLIT (with feature engineering)
# ==================================================

def split_dataset_fixed(df, target):
    """
    Apply feature engineering, then split into train/test.
    """
    # Apply feature engineering to all data
    df_fe = feature_engineering(df, target)

    X = df_fe.drop(columns=[target])
    y = df_fe[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,   # 20% testing
        shuffle=True,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


# ==================================================
# AUTO MODEL SELECTION
# ==================================================

def auto_model_selection(X_train, X_test, y_train, y_test, problem_type):
    """
    Automatically trains multiple models and selects the best one.
    Returns: best_model, performance_dict
    """
    performance = {}
    models = []

    if problem_type == "classification":
        models = [
            ("Logistic Regression", LogisticRegression(max_iter=1000)),
            ("Decision Tree", DecisionTreeClassifier()),
            ("Random Forest", RandomForestClassifier())
        ]
        metric_func = accuracy_score
    else:  # regression
        models = [
            ("Linear Regression", LinearRegression()),
            ("Decision Tree Regressor", DecisionTreeRegressor()),
            ("Random Forest Regressor", RandomForestRegressor())
        ]
        metric_func = r2_score

    best_score = -float('inf')
    best_model = None

    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = metric_func(y_test, y_pred)
        performance[name] = score

        if score > best_score:
            best_score = score
            best_model = model

    return best_model, performance


# ==================================================
# METRICS & EVALUATION
# ==================================================

def evaluate_model_metrics(model, X_test, y_test, problem_type="classification"):
    results = {}

    if problem_type == "classification":
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None

        # Basic metrics
        results["accuracy"] = accuracy_score(y_test, y_pred)
        results["precision"] = precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        results["recall"] = recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        results["f1_score"] = f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        results["classification_report"] = classification_report(
            y_test, y_pred, output_dict=True
        )

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        results["confusion_matrix_plot"] = base64.b64encode(
            buf.read()
        ).decode('utf-8')

        # ROC/AUC (only for binary classification)
        if y_proba is not None and len(np.unique(y_test)) == 2:
            fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
            auc_score = roc_auc_score(y_test, y_proba[:, 1])
            results["roc_auc"] = auc_score
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC={auc_score:.2f}")
            ax.plot([0, 1], [0, 1], '--', color='gray')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            results["roc_curve_plot"] = base64.b64encode(
                buf.read()
            ).decode('utf-8')

    elif problem_type == "regression":
        y_pred = model.predict(X_test)

        # Basic regression metrics
        results["MAE"] = mean_absolute_error(y_test, y_pred)
        results["MSE"] = mean_squared_error(y_test, y_pred)
        results["RMSE"] = np.sqrt(results["MSE"])
        results["R2"] = r2_score(y_test, y_pred)

        # Predicted vs Actual plot
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'r--'
        )
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Predicted vs Actual")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        results["predicted_vs_actual_plot"] = base64.b64encode(
            buf.read()
        ).decode('utf-8')

        # Feature importance for tree-based models (optional)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(
                x=importances,
                y=[f"F{i}" for i in range(len(importances))],
                ax=ax
            )
            ax.set_title("Feature Importances")
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            results["feature_importance_plot"] = base64.b64encode(
                buf.read()
            ).decode('utf-8')

    return results


# ==================================================
# PREDICTION ON NEW DATA
# ==================================================

def predict_new_data(model, new_data, encoders=None):
    """
    Predict outcome for a new input.

    Parameters:
    - model: Trained ML model
    - new_data: dict, keys are feature names (AFTER feature engineering),
                values are user input
    - encoders: dict, feature_name -> LabelEncoder (for categorical features)

    Returns:
    - prediction: predicted value
    - proba: probabilities (if classification model supports predict_proba)
    """
    # Convert to DataFrame
    df_new = pd.DataFrame([new_data])

    # Encode categorical features if encoders provided
    if encoders:
        for col, encoder in encoders.items():
            if col in df_new.columns:
                df_new[col] = encoder.transform(df_new[col])

    # NOTE:
    # For a full production setup, the SAME feature_engineering steps
    # applied during training should also be applied here to df_new,
    # using the same columns and transformations.

    # Predict
    prediction = model.predict(df_new)

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df_new)

    return prediction[0], proba

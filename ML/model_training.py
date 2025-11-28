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
# 1. I  will send you the dataframe columns from the backend.
#    Example: ["Age", "Gender", "Salary", "Purchased"]
#
# 2. You must create a dropdown widget (select box) in the UI.
#    This dropdown MUST display all column names sent from the backend.
#
# 3. The user will choose ONE column as the target (label).
#    Example: User chooses "Purchased".
#
# 4. When the user selects the target column:
#       - You must send the selected target column name BACK to the backend.
#       - Backend will train the ML model using that column as y.
#
# 5. Important:
#       - Your dropdown should NOT guess the target.
#       - Your job is only to SHOW the columns and get user selection.
#
# 6. Your UI flow:
#       - Receive df columns from backend
#       - Display them in dropdown
#       - Wait for user to select target column
#       - Send the chosen column name to backend (API call)
#
# 7. Example (pseudo):
#       select_target = Dropdown(options=df_columns)
#       when user clicks "Train", send select_target.value to backend
#
# --------------------------------------------------

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def get_dataset_info(df):
    return {
        "columns": df.columns.tolist(),
        "preview": df.head(10).to_dict(orient="records")
    }

#zahra 3meli check enu target li user 3emela select valid bas tkun valid b3atili traget li na2eha
def validate_target_column(df, target_col):
    #  Target must exist
    if target_col not in df.columns:
        return False, "Selected target column does not exist in the dataset."

    #  Target must NOT be unique ID or text name
    if df[target_col].dtype == object and df[target_col].nunique() == df.shape[0]:
        return False, "Target cannot be a unique identifier or name column."

    # Target with too many text categories is invalid
    if df[target_col].dtype == object and df[target_col].nunique() > 20:
        return False, "Target has too many categories — cannot be predicted."

    #  Target must have at least 2 distinct classes
    if df[target_col].nunique() < 2:
        return False, "Target column must have at least two classes."

    return True, "Target column is valid."

##################################################################
####hed part lal auto detection iza classification aw regression
def detect_problem_type(df, target):
    """Detect whether the ML problem is regression or classification."""
    if pd.api.types.is_numeric_dtype(df[target]):
        return "regression"
    else:
        return "classification"
       

def encode_target_if_categorical(df, target):
    """Encode target if it's categorical and return encoder (or None)."""
    ptype = detect_problem_type(df, target)
    encoder = None

    if ptype == "classification":
        encoder = LabelEncoder()
        df[target] = encoder.fit_transform(df[target])
    
    return df, encoder, ptype

######Bas luser yf2oss split data call the fucntion
def split_dataset_fixed(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,   # 20% testing
        shuffle=True,
        random_state=42
    )
    
    return X_train, X_test, y_train, y_test



# ---- Now using auto model selection by default ----
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

######part 5 button lal metrics evaluation
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
        results["precision"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        results["recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        results["f1_score"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        results["classification_report"] = classification_report(y_test, y_pred, output_dict=True)

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        results["confusion_matrix_plot"] = base64.b64encode(buf.read()).decode('utf-8')

        # ROC/AUC
        if y_proba is not None and len(np.unique(y_test)) == 2:
            fpr, tpr, thresholds = roc_curve(y_test, y_proba[:,1])
            auc_score = roc_auc_score(y_test, y_proba[:,1])
            results["roc_auc"] = auc_score
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC={auc_score:.2f}")
            ax.plot([0,1],[0,1],'--', color='gray')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            results["roc_curve_plot"] = base64.b64encode(buf.read()).decode('utf-8')

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
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Predicted vs Actual")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        results["predicted_vs_actual_plot"] = base64.b64encode(buf.read()).decode('utf-8')

        # Feature importance for tree-based models
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            fig, ax = plt.subplots(figsize=(6,4))
            sns.barplot(x=importances, y=[f"F{i}" for i in range(len(importances))], ax=ax)
            ax.set_title("Feature Importances")
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            results["feature_importance_plot"] = base64.b64encode(buf.read()).decode('utf-8')

    return results

def predict_new_data(model, new_data, encoders=None):
    """
    Predict outcome for a new input.

    Parameters:
    - model: Trained ML model
    - new_data: dict, keys are feature names, values are user input
    - encoders: dict, feature_name -> LabelEncoder (optional for categorical features)

    Returns:
    - prediction: predicted value
    - proba: probabilities (if classification)
    """

    import numpy as np

    # Convert to DataFrame
    df_new = pd.DataFrame([new_data])

    # Encode categorical features if encoders provided
    if encoders:
        for col, encoder in encoders.items():
            if col in df_new.columns:
                df_new[col] = encoder.transform(df_new[col])

    # Predict
    prediction = model.predict(df_new)
    
    # Get probability if classification
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df_new)
    
    return prediction[0], proba

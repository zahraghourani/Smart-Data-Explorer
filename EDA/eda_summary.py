import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def show_duplicates_total(df):
    if df is None:
        return
    dup_rows = df.duplicated().sum()
    print("Hi", df[df.duplicated()])
    return dup_rows


def show_duplicates(df):
    if df is None:
        return None
    dupes = {}
    for col in df.columns:
        dupes[col] = df[col].duplicated().sum()
    return dupes


def show_missing_total(df):
    if df is None:
        return
    missing = df.isnull().sum().sum()
    return missing

def show_missing(df):
    if df is None:
        return None
    return {col: df[col].isnull().sum() for col in df.columns}

def show_overview(df):
    if df is None:
        return
    
    # Create popup window
    win = tk.Toplevel()
    win.title("Data Overview (First 10 Rows)")

    # Table (Treeview)
    tree = ttk.Treeview(win, show="headings")
    tree.pack(fill="both", expand=True)

    # Add columns to the table
    tree["columns"] = list(df.columns)

    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=120, anchor="center")

    # Show first 10 rows
    preview = df.head(10)

    for _, row in preview.iterrows():
        tree.insert("", "end", values=list(row))

def get_min_max(df):
    min_max_info = {}

    for col in df.columns:
        # If column is numeric or datetime, compute min/max normally
        try:
            min_val = df[col].min()
            max_val = df[col].max()
        except:
            min_val = None
            max_val = None

        # Convert to string to avoid Tkinter issues with NaN, timestamps, etc.
        min_max_info[col] = {
            "min": str(min_val),
            "max": str(max_val)
        }

    return min_max_info

def detect_outliers(df, col):
    if df is None or col not in df.columns:
        return {"total_outliers": "N/A"}

    series = df[col].copy()

    # STEP 1 — convert booleans manually (True=1, False=0)
    if series.dtype == "bool":
        series = series.astype(int)

    # STEP 2 — force everything to numeric
    numeric_series = pd.to_numeric(series, errors="coerce")

    # STEP 3 — drop non-numeric values
    numeric_series = numeric_series.dropna()

    if numeric_series.empty:
        return {
            "column": col,
            "total_outliers": "N/A",
            "outliers": [],
            "lower_bound": None,
            "upper_bound": None,
            "note": "non-numeric column — outliers not applicable"
        }

    # STEP 4 — compute IQR
    Q1 = numeric_series.quantile(0.25)
    Q3 = numeric_series.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = numeric_series[(numeric_series < lower) | (numeric_series > upper)]

    return {
        "column": col,
        "total_outliers": int(len(outliers)),
        "outliers": outliers.tolist(),
        "lower_bound": float(lower),
        "upper_bound": float(upper)
    }



def get_outliers_count(df):
    """
    Safe version: Returns outlier counts only for numeric columns.
    Non-numeric → 'N/A'
    """
    outliers_count = {}

    for col in df.columns:
        # convert to numeric safely
        series = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(series) == 0:
            outliers_count[col] = "N/A"
            continue

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        count = ((series < lower) | (series > upper)).sum()
        outliers_count[col] = int(count)

    return outliers_count
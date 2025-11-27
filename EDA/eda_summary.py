import tkinter as tk
from tkinter import ttk, messagebox

class DuplicateTable(tk.Frame):
    def __init__(self, parent, df):
        super().__init__(parent)

        tree = ttk.Treeview(self, columns=("Column", "Duplicates"), show="headings")
        tree.heading("Column", text="Column Name")
        tree.heading("Duplicates", text="Number of Duplicates")
        tree.pack(fill="both", expand=True)

        if df is not None:
            for col in df.columns:
                dup_count = df[col].duplicated().sum()
                tree.insert("", "end", values=(col, dup_count))

def create_duplicates_table(parent,df):
    if df is None:
        return None

    frame = tk.Frame(parent)
    tree = ttk.Treeview(frame, columns=("Column", "Duplicates"), show="headings")
    tree.heading("Column", text="Column Name")
    tree.heading("Duplicates", text="Number of Duplicates")
    tree.pack(fill="both", expand=True)

    # Insert rows
    for col in df.columns:
        dup_count = df[col].duplicated().sum()
        tree.insert("", "end", values=(col, dup_count))
    
    return frame

def show_missing(df):
    if df is None:
        messagebox.showwarning("No Data", "Load a CSV first.")
        return

    win = tk.Toplevel()
    win.title("Missing Values Overview")

    tree = ttk.Treeview(win, columns=("Column", "Missing"), show="headings")
    tree.heading("Column", text="Column Name")
    tree.heading("Missing", text="Missing Values")
    tree.pack(fill="both", expand=True)

    missing = df.isnull().sum()

    for col in df.columns:
        tree.insert("", "end", values=(col, missing[col]))


def show_overview(df):
    if df is None:
        messagebox.showwarning("No Data", "Load a CSV first.")
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

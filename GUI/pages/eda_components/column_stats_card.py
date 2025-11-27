import customtkinter as ctk
from .base_card import BaseCard

class ColumnStatsCard(BaseCard):
    """Column-wise statistics table"""

    def __init__(self, parent):
        super().__init__(parent, title="Column-Wise Stats Table")

        self.table_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent"
        )
        self.table_frame.pack(fill="both", expand=True, padx=20, pady=(0,10))


    def update(self, df):
        for w in self.table_frame.winfo_children():
            w.destroy()

        if df is None:
            ctk.CTkLabel(
                self.table_frame,
                text="No dataset loaded",
                font=("Poppins", 14)
            ).pack(anchor="w")
            return

        cols = list(df.columns)

        # --- Compute stats ---
        missing = df.isna().sum()
        duplicates = df.duplicated().sum()    # full rows (simplified)
        unique_counts = df.nunique()

        # Dummy outliers (static values)
        outliers = {col: 0 for col in cols}

        # --- Table Header ---
        headers = ["Column Name", "Missing Values", "Duplicate Rows",
                   "Unique Values", "Outliers"]

        for col_idx, h in enumerate(headers):
            ctk.CTkLabel(
                self.table_frame, text=h,
                font=("Poppins", 14, "bold")
            ).grid(row=0, column=col_idx, padx=10, pady=5)

        # --- Data Rows ---
        for row_idx, col in enumerate(cols):
            values = [
                col,
                missing[col],
                duplicates,
                unique_counts[col],
                outliers[col]
            ]

            for col_idx, v in enumerate(values):
                ctk.CTkLabel(
                    self.table_frame, text=str(v),
                    font=("Poppins", 13)
                ).grid(row=row_idx + 1, column=col_idx,
                       padx=10, pady=4)

        for col_idx in range(len(headers)):
            self.table_frame.grid_columnconfigure(col_idx, weight=1)

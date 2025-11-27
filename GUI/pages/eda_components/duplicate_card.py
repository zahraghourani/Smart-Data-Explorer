import customtkinter as ctk
from .base_card import BaseCard

class DuplicateCard(BaseCard):
    """Duplicate counts per column — same design as ColumnStatsCard."""

    def __init__(self, parent):
        super().__init__(parent, title="Duplicate Rows per Column")

        # Scrollable table frame (same style as ColumnStatsCard)
        self.table_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent"
        )
        self.table_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))

    def update(self, df):
        # clear old rows
        for w in self.table_frame.winfo_children():
            w.destroy()

        if df is None:
            ctk.CTkLabel(
                self.table_frame,
                text="No dataset loaded",
                font=("Poppins", 14)
            ).pack(anchor="w")
            return

        # ---- Compute duplicate counts ----
        duplicate_counts = df.apply(lambda col: col.duplicated().sum())

        # ---- Header ----
        headers = ["Column Name", "Duplicate Values"]

        for col_idx, h in enumerate(headers):
            ctk.CTkLabel(
                self.table_frame,
                text=h,
                font=("Poppins", 14, "bold")
            ).grid(row=0, column=col_idx, padx=10, pady=5)

        # ---- Rows ----
        for row_idx, col in enumerate(df.columns):
            values = [
                col,
                int(duplicate_counts[col])
            ]

            for col_idx, v in enumerate(values):
                ctk.CTkLabel(
                    self.table_frame,
                    text=str(v),
                    font=("Poppins", 13)
                ).grid(row=row_idx + 1, column=col_idx,
                       padx=10, pady=4)

        # allow columns to stretch
        for col_idx in range(len(headers)):
            self.table_frame.grid_columnconfigure(col_idx, weight=1)

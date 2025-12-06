from EDA.eda_summary import *
import customtkinter as ctk
import tkinter as tk
from .base_card import BaseCard

class ColumnStatsCard(BaseCard):
    """Column-wise statistics table"""

    def __init__(self, parent):
        super().__init__(parent, title="Column-Wise Stats Table")

        # self.header_frame = ctk.CTkFrame(self, fg_color="white")
        # self.header_frame.pack(fill="x", padx=20)

        # ----- CONTAINER FRAME -----
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=20, pady=(0,10))

        # ----- CANVAS (for horizontal scroll) -----
        self.canvas = tk.Canvas(
            container,
            bg="white",
            highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # ----- SCROLLBAR -----
        self.v_scroll = ctk.CTkScrollbar(
            container, orientation="vertical",
            command=self.canvas.yview
        )
        self.v_scroll.grid(row=0, column=1, sticky="ns")

        self.h_scroll = ctk.CTkScrollbar(
            container,
            orientation="horizontal",
            command=self.canvas.xview
        )
        self.h_scroll.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(
            yscrollcommand=self.v_scroll.set,
            xscrollcommand=self.h_scroll.set
        )

        # ----- INNER FRAME INSIDE CANVAS -----
        self.inner = ctk.CTkFrame(self.canvas, fg_color="white")
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw"
        )

        # auto-expand
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # resize scroll region
        self.inner.bind("<Configure>", self._update_scroll_region)


    def update(self, df):
        # clear previous widgets
        for widget in self.inner.winfo_children():
            widget.destroy()

        if df is None:
            ctk.CTkLabel(
                self.inner,
                text="No dataset loaded",
                font=("Poppins", 14)
            ).pack(anchor="w")
            return

        cols = list(df.columns)

        # --- Compute stats ---
        missing = show_missing(df)
        duplicates = show_duplicates(df)
        unique_counts = df.nunique()
        min_max = get_min_max(df)
        # Replace placeholder with actual outlier counts
        outliers = {}
        for col in cols:
            result = detect_outliers(df, col)
            if isinstance(result, dict) and "total_outliers" in result:
                outliers[col] = result["total_outliers"]
            else:
                outliers[col] = "N/A"


        # --- Header ---
        headers = ["Column Name", "Missing Values", "Duplicate Rows",
                   "Unique Values", "Min", "Max", "Outliers"]
        
        for col_idx, h in enumerate(headers):
            ctk.CTkLabel(
                self.inner,
                text=h,
                font=("Poppins", 14, "bold"),
                width=150,
                anchor="w"
            ).grid(row=0, column=col_idx, padx=10, pady=8)

        # --- Rows ---
        for row_idx, col in enumerate(cols):
            values = [
                col,
                missing[col],
                duplicates[col],
                unique_counts[col],
                min_max[col]["min"],
                min_max[col]["max"],
                outliers[col]
            ]

            for col_idx, v in enumerate(values):
                ctk.CTkLabel(
                    self.inner, text=str(v),
                    font=("Poppins", 13),
                    width=150,
                    anchor="w"
                ).grid(
                    row=row_idx + 1,
                    column=col_idx,
                    padx=10,
                    pady=3,
                    sticky="w"
                )

        # stretch columns
        for col_idx in range(len(headers)):
            self.inner.grid_columnconfigure(col_idx, weight=1)


    def _update_scroll_region(self, event=None):
        """Update canvas scroll size."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

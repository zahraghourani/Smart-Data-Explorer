import customtkinter as ctk
import psutil

class ExtraSummaryCard(ctk.CTkFrame):
    """
    A compact dataset summary card:
    - Total Rows
    - Total Columns
    - Numeric / Categorical / Datetime / Boolean Count
    - Memory Usage
    """

    def __init__(self, parent):
        super().__init__(parent)

        self.title = ctk.CTkLabel(
            self,
            text="Dataset Summary",
            font=("Poppins", 18, "bold"),
            text_color="#53555D"
        )
        self.title.pack(anchor="w", padx=10, pady=(10, 5))

        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Labels (placeholders)
        self.rows_label = self.add_item("Total Rows:")
        self.cols_label = self.add_item("Total Columns:")
        self.num_label = self.add_item("Numeric Columns:")
        self.cat_label = self.add_item("Categorical Columns:")
        self.dt_label = self.add_item("Datetime Columns:")
        self.bool_label = self.add_item("Boolean Columns:")
        self.mem_label = self.add_item("Memory Usage:")

    def add_item(self, title):
        frame = ctk.CTkFrame(self.content, fg_color="transparent")
        frame.pack(fill="x", pady=2)

        label_title = ctk.CTkLabel(
            frame,
            text=title,
            width=180,
            anchor="w",
            font=("Poppins", 14, "bold")
        )
        label_title.pack(side="left")

        label_value = ctk.CTkLabel(
            frame,
            text="-",
            anchor="w",
            font=("Poppins", 14)
        )
        label_value.pack(side="left", padx=10)

        return label_value

    def update(self, df):
        if df is None:
            return

        # Compute counts
        n_rows, n_cols = df.shape

        numeric_cols = df.select_dtypes(include=["int", "float"]).shape[1]
        categorical_cols = df.select_dtypes(include=["object"]).shape[1]
        datetime_cols = df.select_dtypes(include=["datetime"]).shape[1]
        bool_cols = df.select_dtypes(include=["bool"]).shape[1]

        # Memory usage
        mem_bytes = df.memory_usage(deep=True).sum()
        mem_mb = mem_bytes / (1024 ** 2)

        # Update labels
        self.rows_label.configure(text=str(n_rows))
        self.cols_label.configure(text=str(n_cols))
        self.num_label.configure(text=str(numeric_cols))
        self.cat_label.configure(text=str(categorical_cols))
        self.dt_label.configure(text=str(datetime_cols))
        self.bool_label.configure(text=str(bool_cols))
        self.mem_label.configure(text=f"{mem_mb:.2f} MB")

import customtkinter as ctk
from .base_card import BaseCard

class SummaryCard(BaseCard):
    """
    Card that shows:
      - Dataset shape
      - Column types (short text)
      - Missing values summary
      - Duplicate summary
    """

    def __init__(self, parent):
        super().__init__(parent, title="Summary Panel")

        # container frame for the labels
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=(0, 15))

        # 1) Dataset shape
        shape_title = ctk.CTkLabel(content,
                                   text="Dataset Shape:",
                                   font=("Poppins", 14, "bold"))
        shape_title.grid(row=0, column=0, sticky="w", pady=2)

        self.shape_value = ctk.CTkLabel(content,
                                        text="-",
                                        font=("Poppins", 14))
        self.shape_value.grid(row=0, column=1, sticky="w", padx=10, pady=2)

        # 2) Column types (use a textbox with wrap – it can scroll later)
        col_title = ctk.CTkLabel(content,
                                 text="Column Info (Types):",
                                 font=("Poppins", 14, "bold"))
        col_title.grid(row=1, column=0, sticky="nw", pady=(8, 2))

        self.col_text = ctk.CTkTextbox(
            content,
            height=60,
            fg_color="#F7F7F7",
            wrap="word",
            font=("Poppins", 13)
        )
        self.col_text.grid(row=1, column=1, sticky="we", padx=10, pady=(4, 2))

        # 3) Missing values summary
        miss_title = ctk.CTkLabel(content,
                                  text="Missing Values Summary:",
                                  font=("Poppins", 14, "bold"))
        miss_title.grid(row=2, column=0, sticky="w", pady=(8, 2))

        self.miss_value = ctk.CTkLabel(content,
                                       text="-",
                                       font=("Poppins", 14))
        self.miss_value.grid(row=2, column=1, sticky="w", padx=10, pady=(8, 2))

        # 4) Duplicate summary
        dup_title = ctk.CTkLabel(content,
                                 text="Duplicate Summary:",
                                 font=("Poppins", 14, "bold"))
        dup_title.grid(row=3, column=0, sticky="w", pady=(8, 2))

        self.dup_value = ctk.CTkLabel(content,
                                      text="-",
                                      font=("Poppins", 14))
        self.dup_value.grid(row=3, column=1, sticky="w", padx=10, pady=(8, 2))

        # make second column stretch
        content.grid_columnconfigure(1, weight=1)

        # lock textbox as read-only
        self.col_text.insert("end", "")
        self.col_text.configure(state="disabled")

    # ------- public API: called by eda_page.py -------
    def update_from_overview(self, overview: dict):
        """
        overview is a dict like:
        {
          'shape': 'Rows: 1500, Columns: 12',
          'column_types': 'Age: int64, Income: float64, ...',
          'missing_summary': 'Total 145 (9.7%)',
          'duplicate_summary': 'Total 23 rows'
        }
        """

        self.shape_value.configure(text=overview.get("shape", "-"))

        self.col_text.configure(state="normal")
        self.col_text.delete("1.0", "end")
        self.col_text.insert("end", overview.get("column_types", ""))
        self.col_text.configure(state="disabled")

        self.miss_value.configure(text=overview.get("missing_summary", "-"))
        self.dup_value.configure(text=overview.get("duplicate_summary", "-"))

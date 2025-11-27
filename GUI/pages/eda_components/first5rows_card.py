import customtkinter as ctk
from .base_card import BaseCard
import tkinter as tk

class First5RowsCard(BaseCard):
    """Displays df.head() inside a small scrollable table."""

    def __init__(self, parent):
        super().__init__(parent, title="First 5 Rows Preview")

        # OUTER FRAME (holds canvas + scrollbars)
        container = ctk.CTkFrame(self, fg_color="transparent")
        # self.pack(fill="both", expand=True)
        container.pack(fill="both", expand=True)

        # CANVAS (for horizontal + vertical scroll)
        self.canvas = tk.Canvas(
            container,
            bg = "white",
            highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # self.canvas.pack(side="left", fill="both", expand=True)

        # SCROLLBARS
        # self.v_scroll = tk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        # self.h_scroll = tk.Scrollbar(container, orient="horizontal", command=self.canvas.xview)

        # self.v_scroll.pack(side="right", fill="y")
        # self.h_scroll.pack(side="bottom", fill="x")

        # self.canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)

        # SCROLLBARS (CTk look)
        self.v_scroll = ctk.CTkScrollbar(container, orientation="vertical",
                                        command=self.canvas.yview)
        self.v_scroll.grid(row=0, column=1, sticky="ns")

        self.h_scroll = ctk.CTkScrollbar(container, orientation="horizontal",
                                        command=self.canvas.xview)
        self.h_scroll.grid(row=1, column=0, sticky="ew")

        # connect canvas → scrollbars
        self.canvas.configure(
            yscrollcommand=self.v_scroll.set,
            xscrollcommand=self.h_scroll.set
        )

        # INNER FRAME inside canvas
        self.inner = ctk.CTkFrame(self.canvas, fg_color="white")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        # auto-expand correctly
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # update scroll region on resize
        self.inner.bind("<Configure>", self._update_scroll_region)

        # INTERNAL FRAME inside canvas
        # self.inner = ctk.CTkFrame(self.canvas, fg_color="white")
        # self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        # Update scroll region when resized
        # self.inner.bind("<Configure>", self._update_scroll_region)

        # Smooth scrolling
        # self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        # self.canvas.bind_all("<Shift-MouseWheel>", self._on_shift_mousewheel)

        # self.table_frame = ctk.CTkScrollableFrame(
        #     self,
        #     fg_color="transparent",
        #     height=160
        # )
        # self.table_frame.pack(fill="x", padx=20, pady=(0, 10))

        # 🔥 Stop scroll event from reaching the big page
        # Bind scroll to the table only (smooth & fast)
        # self.table_frame.bind("<Enter>", lambda e: self._bind_scroll_to_table())
        # self.table_frame.bind("<Leave>", lambda e: self._unbind_scroll_from_table())


    def update(self, df):
        # clear old content
        for widget in self.inner.winfo_children():
            widget.destroy()

        if df is None:
            ctk.CTkLabel(self.inner,
                         text="No dataset loaded",
                         font=("Poppins", 14)).pack(anchor="w")
            return

        head = df.head()
        cols = list(head.columns)

        # --- Header ---
        for col_idx, col_name in enumerate(cols):
            ctk.CTkLabel(
                self.inner,
                text=col_name,
                font=("Poppins", 14, "bold"),
                width=180,
                anchor="w"
            ).grid(row=0, column=col_idx, padx=10, pady=5, sticky="w")

        # --- Rows ---
        for row_idx, row in head.iterrows():
            for col_idx, value in enumerate(row):
                ctk.CTkLabel(
                    self.inner,
                    text=str(value),
                    font=("Poppins", 13),
                    width=180,
                    anchor="w"
                ).grid(row=row_idx + 1, column=col_idx, padx=10, pady=3, sticky="w")

        # make columns stretch
        for col_idx in range(len(cols)):
            self.inner.grid_columnconfigure(col_idx, weight=1)
    # Helper methods
    def _update_scroll_region(self, event=None):
        """Update canvas scroll size."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event):
        """Vertical scrolling with mouse wheel / touchpad."""
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_shift_mousewheel(self, event):
        """Horizontal scroll via SHIFT + wheel or touchpad."""
        direction = -1 if event.delta > 0 else 1
        self.canvas.xview_scroll(direction, "units")

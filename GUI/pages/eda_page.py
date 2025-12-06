import customtkinter as ctk
import tkinter as tk
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from GUI.utils.data_loader import DataStorage
from EDA.eda_summary import *
from GUI.pages.eda_components.summary_card import SummaryCard
from GUI.pages.eda_components.first5rows_card import First5RowsCard
from GUI.pages.eda_components.column_stats_card import ColumnStatsCard
from GUI.pages.eda_components.duplicate_card import DuplicateCard
from GUI.pages.eda_components.extra_summary_card import ExtraSummaryCard
from GUI.pages.eda_components.plot_histogram import plot_histogram
from GUI.pages.eda_components.plot_scatter import plot_scatter
from GUI.pages.eda_components.plot_heatmap import plot_heatmap


class EDAPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(fg_color="#FFFFFF")
        # ----- LEFT SIDEBAR -----
        sidebar = ctk.CTkFrame(self, width=150, fg_color="#E8F4E5", bg_color = "#FFFFFF")
        sidebar.pack(side="left", fill="y")

        # Create rotated images
        img_overview = self.create_vertical_text("Data Overview", font_size=16)
        img_viz = self.create_vertical_text("Visualizations", font_size=16)

        # Buttons with rotated images
        btn_data =ctk.CTkButton(
            sidebar,
            text="",           # we use image instead of text
            image=img_overview,
            width=40,
            height=160,
            fg_color="transparent",
            hover_color="#DDECE0",
            command=lambda: self.switch_tab("overview")
        )
        btn_data.pack(pady=10, padx = 4)

        btn_viz = ctk.CTkButton(
            sidebar,
            text="",
            image=img_viz,
            width=40,
            height=160,
            fg_color="transparent",
            hover_color="#DDECE0",
            command=lambda: self.switch_tab("viz")
        )
        btn_viz.pack(pady=10)

        self.sidebar_buttons = {}
        self.sidebar_buttons["overview"] = btn_data
        self.sidebar_buttons["viz"] = btn_viz
        
        # scrollable page (full page)
        self.scroll = ctk.CTkScrollableFrame(self, fg_color="#FFFFFF", bg_color="#FFFFFF")
        self.scroll._parent_canvas.bind("<Configure>", self._resize_scroll_width)
        self.scroll.pack(fill="both", expand=True, padx=10, pady=10)

        # Remove the grey canvas background inside CTkScrollableFrame
        canvas = self.scroll._parent_canvas
        canvas.configure(bg="#FFFFFF", highlightthickness=0)

        # layout: left + right columns
        self.scroll.grid_columnconfigure(0, weight=3)
        self.scroll.grid_columnconfigure(1, weight=2)

        self.left = ctk.CTkFrame(self.scroll, fg_color="transparent")
        self.left.grid(row=0, column=0, sticky="nwe", padx=(0, 10))

        self.right = ctk.CTkFrame(self.scroll, fg_color="transparent")
        self.right.grid(row=0, column=1, sticky="nwe", padx=(10, 0))

        self.active_tab = "overview"
        self.tab_content = ctk.CTkFrame(self.left, fg_color="transparent")
        self.tab_content.pack(fill="both", expand=True)

        # Make 2 equal columns and 2 equal rows
        self.tab_content.grid_columnconfigure(0, weight=1, uniform="col")
        self.tab_content.grid_columnconfigure(1, weight=1, uniform="col")

        self.tab_content.grid_rowconfigure(0, weight=1, uniform="row")
        self.tab_content.grid_rowconfigure(1, weight=1, uniform="row")



    def refresh(self):
        df = DataStorage.get()
        if df is None:
            return
        self.switch_tab(self.active_tab)
        print("REFRESH → Data:", DataStorage.get())
        print("DEBUG:", type(DataStorage.get()), DataStorage.get().shape if DataStorage.get() is not None else None)


    def show_data_head(self):
        df = DataStorage.get()

        if df is None:
            self.data_box.insert("end", "⚠ No dataset loaded.\nPlease go back and load a CSV file.")
        else:
            # Convert df head to string and display
            head_text = df.head().to_string()
            self.data_box.insert("end", head_text)

    def switch_tab(self, tab_name):
        self.active_tab = tab_name

        # Clear previous widgets
        for widget in self.tab_content.winfo_children():
            widget.destroy()

        # Render new content
        if tab_name == "overview":
            self.render_overview_tab()
        elif tab_name == "viz":
            self.render_visualizations_tab()
        
        self.highlight_sidebar(tab_name)

    
    def render_overview_tab(self):
        # --- Build overview summary dict ---
        df = DataStorage.get()
        if df is None:
            overview = {
                "shape": "-",
                "column_types": "",
                "missing_summary": "-",
                "duplicate_summary": "-"
            }
        else:
            # Shape
            n_rows, n_cols = df.shape
            shape_text = f"Rows: {n_rows}, Columns: {n_cols}"

            # Column types
            col_types = ", ".join([f"{col}: {dtype}" for col, dtype in df.dtypes.items()])

            # Missing
            total_missing = show_missing_total(df)
            pct_missing = (total_missing / (n_rows * n_cols)) * 100 if n_rows > 0 else 0
            missing_text = f"Total {total_missing} ({pct_missing:.1f}%)"

            # Duplicates
            dup_rows = show_duplicates_total(df)
            dup_text = f"Total {dup_rows} rows"

            overview = {
                "shape": shape_text,
                "column_types": col_types,
                "missing_summary": missing_text,
                "duplicate_summary": dup_text
            }

        # SUMMARY CARD
        # self.summary_card = SummaryCard(self.tab_content)
        # self.create_shadow_wrapper(self.tab_content, self.summary_card)
        # self.summary_card.update_from_overview(overview)


        # FIRST 5 ROWS CARD
        # self.first5_rows_card = First5RowsCard(self.tab_content)
        # self.create_shadow_wrapper(self.tab_content, self.first5_rows_card)
        # self.first5_rows_card.update(df)

        # COLUMN STATS CARD
        # self.col_stats_card = ColumnStatsCard(self.tab_content)
        # self.create_shadow_wrapper(self.tab_content, self.col_stats_card)
        # self.col_stats_card.update(df)

        # DUPLICATE CARD
        # self.dup_card = DuplicateCard(self.tab_content)
        # self.create_shadow_wrapper(self.tab_content, self.dup_card)
        # self.dup_card.update(df)

        # OVERVIEW TAB CONTENT
        # show_overview(df)

        # SUMMARY CARD
        # wrapper, summary = self.create_shadow_wrapper(self.tab_content, SummaryCard)
        # summary.update_from_overview(overview)

        wrapper_summary, summary = self.create_shadow_wrapper(self.tab_content, SummaryCard)
        summary.update_from_overview(overview)
        wrapper_summary.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)


        # FIRST 5 ROWS CARD
        # wrapper, first5 = self.create_shadow_wrapper(self.tab_content, First5RowsCard)
        # first5.update(df)

        wrapper_first5, first5 = self.create_shadow_wrapper(self.tab_content, First5RowsCard)
        first5.update(df)
        wrapper_first5.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        # COLUMN STATS
        # wrapper, colstats = self.create_shadow_wrapper(self.tab_content, ColumnStatsCard)
        # colstats.update(df)

        wrapper_stats, colstats = self.create_shadow_wrapper(self.tab_content, ColumnStatsCard)
        colstats.update(df)
        wrapper_stats.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        # EXTRA SUMMARY PANEL (NEW)
        wrapper_extra, extra = self.create_shadow_wrapper(self.tab_content, ExtraSummaryCard)
        extra.update(df)
        wrapper_extra.grid(row=1, column=1, sticky="nwe", padx=10, pady=10)

        # DUPLICATES
        # from GUI.pages.eda_components.duplicate_card import DuplicateCard
        # wrapper, dup = self.create_shadow_wrapper(self.tab_content, DuplicateCard)
        # dup.update(df)


    def create_vertical_text(self, text, font_size=16, text_color="#53555D", bg_color=(0, 0, 0, 0)):
        # Load a TTF font (change path if needed)
        try:
            font = ImageFont.truetype("GUI/assets/Poppins-SemiBold.ttf", font_size)
        except:
            font = ImageFont.load_default()
        padding = 16
        # Dummy image to calculate text bounding box
        dummy_img = Image.new("RGBA", (1, 1), bg_color)
        draw = ImageDraw.Draw(dummy_img)

        # Use textbbox instead of textsize
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0] + padding
        text_height = bbox[3] - bbox[1] + padding

        # Create actual image
        img = Image.new("RGBA", (text_width, text_height), bg_color)
        draw = ImageDraw.Draw(img)
        draw.text((0,0), text, font=font, fill=text_color)

        # Rotate 90 degrees
        rotated = img.rotate(90, expand=True)

        return ctk.CTkImage(light_image=rotated, size=rotated.size)

    
    def highlight_sidebar(self, active):
        for name, btn in self.sidebar_buttons.items():
            if name == active:
                btn.configure(fg_color="#A6DCA4")
            else:
                btn.configure(fg_color="transparent")


    def create_shadow_wrapper(self, parent, CardClass):
        wrapper = ctk.CTkFrame(
            parent,
            fg_color="#E5E5E5",
            corner_radius=12
        )

        # FIX: prevent wrapper from shrinking
        wrapper.grid_propagate(False)

        # Force equal card size (change if you want)
        wrapper.configure(width=550, height=300)

        # Create card inside wrapper
        card = CardClass(wrapper)
        card.configure(fg_color="white")
        card.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        wrapper.grid_rowconfigure(0, weight=1)
        wrapper.grid_columnconfigure(0, weight=1)

        return wrapper, card


    def _resize_scroll_width(self, event):
        canvas = self.scroll._parent_canvas

        try:
            # ALWAYS WORKS: get the first canvas window (the scrollable frame)
            window_id = canvas.find_all()[0]

            # Resize the inner frame so it matches canvas width
            canvas.itemconfig(window_id, width=canvas.winfo_width())

        except Exception as e:
            print("Resize error:", e)


    def render_visualizations_tab(self):
        # Clear previous widgets
        for w in self.tab_content.winfo_children():
            w.destroy()
        
        # Get the dataframe
        df = DataStorage.get()

        df_clean = df.copy()

        # Convert all possible numeric columns safely
        # for col in df_clean.columns:
        #     df_clean[col] = pd.to_numeric(df_clean[col], errors="ignore")
        df_clean = df.apply(lambda col: pd.to_numeric(col, errors="coerce"))


        # Select only true numeric columns
        numeric_cols = df_clean.select_dtypes(include=["number"]).columns.tolist()

        # Filter out columns with no real numeric data
        numeric_cols = [
            col for col in numeric_cols 
            if df_clean[col].dropna().size > 0
        ]
        
        # Get numeric columns
        # numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            # If no numeric columns, show a message
            no_data_label = ctk.CTkLabel(self.tab_content, text="No numeric columns available for visualization.", font=("Poppins", 14))
            no_data_label.pack(pady=20)
            return
        
        # Function to create a plot frame with title
        # def create_plot_frame(parent, title):
        #     frame = ctk.CTkFrame(parent, fg_color="white", corner_radius=10)
        #     # frame.pack(fill="x", padx=20, pady=30)
        #     title_label = ctk.CTkLabel(frame, text=title, font=("Poppins", 16, "bold"))
        #     title_label.pack(pady=(10, 5))
        #     plot_frame = ctk.CTkFrame(frame, fg_color="white", height=350)
        #     plot_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        #     plot_frame.pack_propagate(False)

        #     return frame, plot_frame

        def create_plot_frame(parent, title):
            frame = ctk.CTkFrame(parent, fg_color="white", corner_radius=10)

            # Allow expansion
            frame.grid_columnconfigure(0, weight=1)
            frame.grid_rowconfigure(1, weight=1)

            title_label = ctk.CTkLabel(frame, text=title, font=("Poppins", 16, "bold"))
            title_label.grid(row=0, column=0, pady=(10,5))

            plot_frame = ctk.CTkFrame(frame, fg_color="white", height=350)
            plot_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0,10))

            return frame, plot_frame


        
        # container for top row
        top_row = ctk.CTkFrame(self.tab_content, fg_color="transparent")
        top_row.pack(fill="both", expand=True, padx=10, pady=10)

        top_row.grid_columnconfigure(0, weight=1)
        top_row.grid_columnconfigure(1, weight=1)
        top_row.grid_rowconfigure(0, weight=1)

        # 1. Histogram with dropdown
        hist_frame, hist_plot_frame = create_plot_frame(top_row, "Histogram")
        hist_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        hist_var = ctk.StringVar(value=numeric_cols[0] if numeric_cols else "")
        hist_menu = ctk.CTkOptionMenu(
            hist_frame, 
            variable=hist_var, 
            values=numeric_cols, 
            command=lambda col: self.update_histogram(col, hist_plot_frame)
        )
        hist_menu.grid(row=2, column=0, pady=(0,10))
        
        # Initial histogram
        self.update_histogram(hist_var.get(), hist_plot_frame)
        
        # 2. Correlation Heatmap
        heatmap_frame, heatmap_plot_frame = create_plot_frame(top_row, "Correlation Heatmap")
        heatmap_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        hist_frame.grid_columnconfigure(0, weight=1)
        hist_frame.grid_rowconfigure(1, weight=1)

        heatmap_frame.grid_columnconfigure(0, weight=1)
        heatmap_frame.grid_rowconfigure(1, weight=1)


        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Heatmap")
        canvas = FigureCanvasTkAgg(fig, master=heatmap_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # 3. Scatter Plot with X and Y dropdowns
        scatter_frame, scatter_plot_frame = create_plot_frame(self.tab_content, "Scatter Plot")
        scatter_frame.pack(fill="x", padx=10, pady=20)

        x_var = ctk.StringVar(value=numeric_cols[0] if numeric_cols else "")
        y_var = ctk.StringVar(value=numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0] if numeric_cols else "")
        scatter_frame.grid_rowconfigure(2, weight=0)
        scatter_frame.grid_columnconfigure(0, weight=1)
        scatter_frame.grid_columnconfigure(1, weight=1)
        x_menu = ctk.CTkOptionMenu(
            scatter_frame,
            variable=x_var,
            values=numeric_cols,
            command=lambda _: self.update_scatter(x_var.get(), y_var.get(), scatter_plot_frame)
        )

        y_menu = ctk.CTkOptionMenu(
            scatter_frame,
            variable=y_var,
            values=numeric_cols,
            command=lambda _: self.update_scatter(x_var.get(), y_var.get(), scatter_plot_frame)
        )

        x_menu.grid(row=2, column=0, padx=(0, 10), pady=(10, 10), sticky="ew")
        y_menu.grid(row=2, column=1, pady=(10, 10), sticky="ew")


        # Initial scatter plot
        self.update_scatter(x_var.get(), y_var.get(), scatter_plot_frame)
    
    def update_histogram(self, column, plot_frame):
        df = DataStorage.get()
        plot_histogram(df, column, plot_frame)

    def update_scatter(self, x_col, y_col, plot_frame):
        df = DataStorage.get()
        plot_scatter(df, x_col, y_col, plot_frame)

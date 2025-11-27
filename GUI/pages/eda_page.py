import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont
from GUI.utils.data_loader import DataStorage
from EDA.eda_summary import create_duplicates_table
from GUI.pages.eda_components.summary_card import SummaryCard
from GUI.pages.eda_components.first5rows_card import First5RowsCard
from GUI.pages.eda_components.column_stats_card import ColumnStatsCard
from GUI.pages.eda_components.duplicate_card import DuplicateCard

class EDAPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

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
        self.scroll.pack(fill="both", expand=True, padx=20, pady=20)

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
            total_missing = df.isna().sum().sum()
            pct_missing = (total_missing / (n_rows * n_cols)) * 100 if n_rows > 0 else 0
            missing_text = f"Total {total_missing} ({pct_missing:.1f}%)"

            # Duplicates
            dup_rows = df.duplicated().sum()
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

        # SUMMARY CARD
        wrapper, summary = self.create_shadow_wrapper(self.tab_content, SummaryCard)
        summary.update_from_overview(overview)

        # FIRST 5 ROWS CARD
        wrapper, first5 = self.create_shadow_wrapper(self.tab_content, First5RowsCard)
        first5.update(df)

        # COLUMN STATS
        wrapper, colstats = self.create_shadow_wrapper(self.tab_content, ColumnStatsCard)
        colstats.update(df)

        # DUPLICATES
        from GUI.pages.eda_components.duplicate_card import DuplicateCard
        wrapper, dup = self.create_shadow_wrapper(self.tab_content, DuplicateCard)
        dup.update(df)




    def render_visualizations_tab(self):
        label = ctk.CTkLabel(self.tab_content, text="Visualization Coming Soon...")
        label.pack(pady=20)

        # Later you add:
        # - correlation heatmap
        # - distribution plots
        # - scatter matrix

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

    def create_shadow_wrapper(self, parent, CardClass, pad=(20, 10)):
        # Outer shadow container
        wrapper = ctk.CTkFrame(
            parent,
            fg_color="#E5E5E5",
            corner_radius=12
        )
        wrapper.pack(fill="x", pady=pad)

        # Create card instance INSIDE wrapper
        card = CardClass(wrapper)
        card.configure(fg_color="white")
        card.pack(fill="both", expand=True, padx=6, pady=6)

        return wrapper, card

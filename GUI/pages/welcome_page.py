import customtkinter as ctk
from tkinter import filedialog
from GUI.utils.data_loader import DataStorage
import pandas as pd
from PIL import Image

class WelcomePage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.configure(fg_color="#F5FAF4") 

        self.controller = controller

        # MAIN GRID (2 rows)
        self.grid_rowconfigure(0, weight=0)     # Title row
        self.grid_rowconfigure(1, weight=1)     # Content row
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # TITLE
        title = ctk.CTkLabel(
            self,
            text="Smart Data Explorer",
            font=("Poppins", 50, "bold")
        )
        title.grid(row=0, column=0, columnspan=2, pady=(20, 5), sticky="n")

        # Left Section
        left = ctk.CTkFrame(self, fg_color="transparent")
        left.grid(row=1, column=0, sticky="n", padx=70, pady=(20, 60))

        # SUBTITLE
        intro = ctk.CTkLabel(left,
                             text="Your data has a story.\nLet Smart Data Explorer help you\nuncover it.",
                             font=("Poppins", 24, "bold"),
                             justify="center")
        intro.pack(anchor="center", pady=(30, 30))

        # ICONS
        clear_icon = ctk.CTkImage(
            light_image=Image.open("GUI/assets/clear.png"),
            size=(22, 22)
        )
        close_icon = ctk.CTkImage(
            light_image=Image.open("GUI/assets/close.png"),
            size=(18, 18)
        )
        load_icon = ctk.CTkImage(
            light_image=Image.open("GUI/assets/document.png"),
            size=(22, 22)
        )
        next_icon = ctk.CTkImage(
            light_image=Image.open("GUI/assets/next.png"),
            size=(22, 22)
        )
        
        # --- BUTTONS ---
        load_btn = ctk.CTkButton(
            left,
            text="Load CSV",
            font=("Poppins", 18),
            image=load_icon,
            compound="left", 
            width=220,
            height=50,
            corner_radius = 25,
            fg_color="#ffffff",   
            text_color="#53555d",
            hover_color="#e8e8e8",
            border_color="#53555d",  
            border_width=1,
            bg_color="#F5FAF4",
            command=self.load_csv
            )
        load_btn.pack(pady=15)

        clear_btn = ctk.CTkButton(
            left,
            text="Clear Data",
            font=("Poppins", 18),
            image=clear_icon,
            compound="left",   # icon on the left
            width=220,
            height=50,
            corner_radius=100,
            command=self.clear_data)
        clear_btn.pack(pady=15)

        exit_btn = ctk.CTkButton(
            left,
            text="Exit",
            font=("Poppins", 18),
            image=close_icon,
            compound="left",   # icon on the left
            width=220,
            height=50,
            corner_radius=100,
            fg_color="#D9534F",
            hover_color="#C9302C",
            command=self.controller.destroy)
        exit_btn.pack(pady=15)

        # STATUS LABEL (shows if file is loaded)
        self.status_label = ctk.CTkLabel(left, text="", font=("Poppins", 14))
        self.status_label.pack(anchor="w", pady=(10, 0))

        # Right Section
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.grid(row=1, column=1, sticky="nsew", padx=40)
        
        # --- NEXT BUTTON ---
        self.next_btn = ctk.CTkButton(
            self,
            text="Next",
            font=("Poppins", 18),
            image=next_icon,
            compound="right",
            width=130,
            height=45,
            corner_radius=30,
            fg_color="#555555",
            hover_color="#444444",
            command=lambda: controller.switch_page("EDAPage"))
        self.next_btn.grid(row=1, column=1, sticky="se", pady=40, padx=40)

        # --- RIGHT ILLUSTRATION IMAGE ---
        try:
            img = ctk.CTkImage(
                light_image=Image.open("GUI/assets/welcome-illustration.png"),
                size=(500, 500)
            )

            img_label = ctk.CTkLabel(self, text="", image=img)
            img_label.grid(row=1, column=1, sticky="ne")

        except Exception as e:
            print("Image Error:", e)

    # FUNCTIONS
    def refresh(self):
        pass  # no data needed

    def load_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select a CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if file_path:
            success = DataStorage.load_csv(file_path)

            if success:
                filename = file_path.split("/")[-1]

                self.status_label.configure(
                    text=f"Loaded: {filename}",
                    text_color="#3c8d40"
                )

                self.next_btn.configure(state="normal")
                print("CSV Loaded Successfully!")
                print(DataStorage.get().head())

            else:
                self.status_label.configure(
                    text="Error loading file.",
                    text_color="red"
                )

    def clear_data(self):
        self.csv_loaded = False
        self.loaded_df = None
        DataStorage.clear()

        # Reset label
        self.status_label.configure(text="", text_color="#000000")

        # Disable next button
        self.next_btn.configure(state="disabled")

        print("Data cleared.")

    def get_dataset(self):
        return self.loaded_df if self.csv_loaded else None

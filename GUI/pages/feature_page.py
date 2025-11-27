import customtkinter as ctk
from GUI.utils.data_loader import DataStorage
import pandas as pd

class FeatureEngineeringPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.configure(fg_color="#F5FAF4") 

        self.controller = controller

        # Example content
        label = ctk.CTkLabel(
            self,
            text="Feature Engineering Page",
            font=("Poppins", 30, "bold")
        )
        label.pack(pady=20)

    def refresh(self):
        # This method can be used to refresh the page content when navigated to
        pass

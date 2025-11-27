import customtkinter as ctk

class BaseCard(ctk.CTkFrame):
    """
    Reusable card with rounded corners and white background.
    Other cards (summary, first rows, etc.) will inherit from this.
    """

    def __init__(self, parent, title: str = "", *args, **kwargs):
        super().__init__(parent,
                         fg_color="#FFFFFF",
                         corner_radius=18,
                         *args, **kwargs)

        # self.pack_propagate(False)  

        # Make the card expand with content
        # self.grid_columnconfigure(0, weight=1)
        
        # Optional title
        if title:
            title_label = ctk.CTkLabel(
                self,
                text=title,
                font=("Poppins", 18, "bold"),
                text_color="#53555D"
            )
            title_label.pack(anchor="w", padx=20, pady=(15, 5))

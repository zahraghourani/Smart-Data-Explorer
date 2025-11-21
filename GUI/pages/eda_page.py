import customtkinter as ctk

class EDAPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.controller = controller

        title = ctk.CTkLabel(self, text="EDA Page",
                             font=("Segoe UI", 28, "bold"))
        title.pack(pady=40)

        back_btn = ctk.CTkButton(self, text="← Back to Welcome",
                                 command=lambda: controller.show_page("WelcomePage"))
        back_btn.pack(pady=20)

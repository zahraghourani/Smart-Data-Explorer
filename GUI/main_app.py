import customtkinter as ctk

from pages.welcome_page import WelcomePage
from pages.eda_page import EDAPage


class SmartDataExplorerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window setup ---
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("green")  # or "blue", etc.

        self.title("Smart Data Explorer")
        self.geometry("1100x650")   # adjust if needed
        self.resizable(True, True)

        # --- Container for pages ---
        container = ctk.CTkFrame(self)
        container.pack(fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # --- Create & store pages ---
        self.pages = {}

        for PageClass in (WelcomePage, EDAPage):
            page = PageClass(container, controller=self)
            name = PageClass.__name__
            self.pages[name] = page
            page.grid(row=0, column=0, sticky="nsew")

        # show first page
        self.show_page("WelcomePage")

    def show_page(self, page_name: str):
        """Raise the selected page to the top (like routing in React)."""
        page = self.pages[page_name]
        page.tkraise()


if __name__ == "__main__":
    app = SmartDataExplorerApp()
    app.mainloop()

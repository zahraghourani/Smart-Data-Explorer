import customtkinter as ctk
from PIL import ImageFont
from GUI.pages.welcome_page import WelcomePage
from GUI.pages.eda_page import EDAPage


class SmartDataExplorerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window setup ---
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("green")  # or "blue", etc.

        self.title("Smart Data Explorer")
        self.geometry("1200x800")   # adjust if needed
        self.resizable(True, True)

        # ---------- NAVBAR ----------
        self.navbar = ctk.CTkFrame(self, height=60, fg_color="#E8F4E5")
        # self.navbar.pack(fill="x")

        # Navbar Buttons
        # self.pages = {}
        # self.current_page = None

        # btn_home = ctk.CTkButton(navbar, text="Welcome", command=lambda: self.show_page("WelcomePage"))
        # btn_eda = ctk.CTkButton(navbar, text="EDA", command=lambda: self.show_page("EDAPage"))
        # btn_feat = ctk.CTkButton(navbar, text="Feature Engineering", command=lambda: self.show_page("FeatureEngineeringPage"))
        # btn_pred = ctk.CTkButton(navbar, text="Model Prediction", command=lambda: self.show_page("ModelPredictionPage"))

        # btn_home.pack(side="left", padx=10)
        # btn_eda.pack(side="left", padx=10)
        # btn_feat.pack(side="left", padx=10)
        # btn_pred.pack(side="left", padx=10)

        # Store buttons in a dictionary
        self.nav_buttons = {}
        
        font = ImageFont.truetype("GUI/assets/Poppins-SemiBold.ttf")
        # Create buttons
        self.nav_buttons["WelcomePage"] = ctk.CTkButton(self.navbar, text="Welcome",
                                                 command=lambda: self.switch_page("WelcomePage"),
                                                 fg_color="transparent", text_color="#53555D")
        self.nav_buttons["EDAPage"] = ctk.CTkButton(self.navbar, text="EDA",
                                                command=lambda: self.switch_page("EDAPage"),
                                                fg_color="transparent", text_color="#53555D")
        self.nav_buttons["FeatureEngineeringPage"] = ctk.CTkButton(self.navbar, text="Feature Engineering",
                                                 command=lambda: self.switch_page("FeatureEngineeringPage"),
                                                 fg_color="transparent", text_color="#53555D")
        self.nav_buttons["ModelPredictionPage"] = ctk.CTkButton(self.navbar, text="Model Prediction",
                                                 command=lambda: self.switch_page("ModelPredictionPage"),
                                                 fg_color="transparent", text_color="#53555D")
        
        # Pack buttons horizontally
        for btn in self.nav_buttons.values():
            btn.pack(side="left", padx=10, pady=10)

        # ---------- CONTENT AREA ----------
        container = ctk.CTkFrame(self)
        container.pack(fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # --- Create & store pages ---
        self.pages = {}
        self.current_page = None
        for PageClass in (WelcomePage, EDAPage):
            page = PageClass(container, controller=self)
            name = PageClass.__name__
            self.pages[name] = page
            page.grid(row=0, column=0, sticky="nsew")
        self.switch_page("WelcomePage")  # default page
        
    def show_page(self, page_name):
        # page = self.pages[page_name]
        # if hasattr(page, "refresh"):
        #     page.refresh()
        # page.tkraise()
        if self.current_page:
            self.current_page.pack_forget()

        self.current_page = self.pages[page_name]
        self.current_page.tkraise()

    def switch_page(self, name):
        page = self.pages[name]

        # Refresh page if needed
        if hasattr(page, "refresh"):
            page.refresh()

        # Show the page
        page.tkraise()
        self.current_page = page

        # Show navbar only on non-welcome pages
        if name == "WelcomePage":
            self.navbar.pack_forget()
        else:
            # Pack navbar IF it's not already visible
            if not self.navbar.winfo_ismapped():
                self.navbar.pack(fill="x", side="top")

            # Highlight correct tab
            self.highlight_tab(name)



    def highlight_tab(self, active_name):
        for name, btn in self.nav_buttons.items():
            if name == active_name:
                # ACTIVE BUTTON (highlight it)
                btn.configure(fg_color="#A6DCA4", text_color="black")
            else:
                # OTHER BUTTONS (make them transparent)
                btn.configure(fg_color="transparent", text_color="black")


if __name__ == "__main__":
    app = SmartDataExplorerApp()
    app.mainloop()

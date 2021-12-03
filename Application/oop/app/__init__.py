import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

from .pages.menu_page import MenuPage

def create_app():
    class App(tk.Tk):
        def __init__(self, *args, **kwargs):
            tk.Tk.__init__(self, *args, **kwargs)
            
            self.title("Face Mask Detection System")
            self.geometry("1000x750")
            self.resizable(False, False)
            
            self.container = tk.Frame(self)
            self.container.pack(side="top", fill="both", expand=True)
            self.container.grid_rowconfigure(0, weight=1)
            self.container.grid_columnconfigure(0, weight=1)

            self.show_frame(MenuPage)

        def show_frame(self, page, *args):
            frame = page(self.container, self, *args)
            frame.grid(row=0, column=0, sticky="nsew")
            frame.tkraise()
    
    return App()
import tkinter as tk

from . image_page import ImagePage
from . video_page import VideoPage
from . camera_page import CameraPage

class Warning:
    def __init__(self, root, title, message):
        self.root = root
        self.root.title(title)
        self.root.geometry("450x325")
        self.root.resizable(width=False, height=False)
        self.root.configure(background="#000C18")

        frame_main = tk.Frame(self.root, background="#000C18")
        frame_main.pack(pady=20)

        label_warning_icon = tk.Label(frame_main, text="⚠", font=("Tw Cen MT", 50, "bold"), bg="#000C18", fg="#E62A32")
        label_warning_icon.pack(padx=20, pady=10)

        label_message = tk.Label(frame_main, text=message, font=("Tw Cen MT", 20), bg="#000C18", fg="#FFFFFF")
        label_message.pack(padx=20, pady=15)

        frame_button = tk.Frame(self.root, background="#000C18")
        frame_button.pack(pady=20)

        button_quit = tk.Button(frame_button, text="OK", command=self.root.destroy, height=1, width=8, fg="#FFFFFF", bg="#E62A32", bd=0, activebackground="#2c2f33", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 20, "bold"), relief="flat")
        button_quit.pack()

class AskQuit:
    def __init__(self, root, controller, title):
        self.root = root
        self.root.title(title)
        self.root.geometry("450x325")
        self.root.resizable(width=False, height=False)
        self.root.configure(background="#000C18")
        self.controller = controller

        frame_main = tk.Frame(self.root, background="#000C18")
        frame_main.pack(pady=20)

        label_warning_icon = tk.Label(frame_main, text="✋", font=("UD Digi Kyokasho NK-B", 50), bg="#000C18", fg="#E62A32")
        label_warning_icon.pack(padx=20, pady=10)

        label_message = tk.Label(frame_main, text="Are you sure you want to quit?", font=("Tw Cen MT", 20), bg="#000C18", fg="#FFFFFF")
        label_message.pack(padx=20, pady=15)

        frame_button = tk.Frame(self.root, background="#000C18")
        frame_button.pack(pady=20)

        button_yes = tk.Button(frame_button, text="Yes", command=self.yes, height=1, width=8, fg="#FFFFFF", bg="#E62A32", bd=0, activebackground="#2c2f33", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 20, "bold"), relief="flat")
        button_yes.grid(row=0, column=0, padx=20)

        button_no = tk.Button(frame_button, text="No", command=self.no, height=1, width=8, fg="#FFFFFF", bg="#00AAEB", bd=0, activebackground="#2c2f33", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 20, "bold"), relief="flat")
        button_no.grid(row=0, column=1, padx=20)

    def yes(self):
        self.controller.destroy()
    
    def no(self):
        self.root.destroy()

class MenuPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.configure(background="#000C18")
        self.controller = controller

        label_title = tk.Label(self, text="FACE MASK DETECTION SYSTEM", font=("Tw Cen MT", 45, "bold"), bg="#000C18", fg="#FFFFFF")
        label_title.pack(pady=30)

        label_header = tk.Label(self, text="Choose Input Type", font=("Tw Cen MT", 30), bg="#000C18", fg="#FFFFFF")
        label_header.pack(pady=15)

        buttons = tk.Frame(self, background="#000C18")
        buttons.pack(pady=20)

        button_image = tk.Button(buttons, text="Image", command=lambda: controller.show_frame(ImagePage), width=12, fg="#FFFFFF", bg="#00AAEB", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 20, "bold"), relief="raised")
        button_image.pack(pady=20)
        
        button_video = tk.Button(buttons, text="Video", command=lambda: controller.show_frame(VideoPage), width=12, fg="#FFFFFF", bg="#00AAEB", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 20, "bold"), relief="raised")
        button_video.pack(pady=20)
        
        button_camera = tk.Button(buttons, text="Camera", command=self.browse_camera_page, width=12, fg="#FFFFFF", bg="#00AAEB", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 20, "bold"), relief="raised")
        button_camera.pack(pady=20)

        button_quit = tk.Button(buttons, text="Quit", command=lambda: AskQuit(tk.Toplevel(self), self.controller, "Quit"), width=12, fg="#FFFFFF", bg="#E62A32", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 20, "bold"), relief="raised")
        button_quit.pack(pady=20)
    
    def browse_camera_page(self):
        try:
            self.controller.show_frame(CameraPage)
        except ValueError:
            Warning(tk.Toplevel(self), "Warning", "No webcam available")
        except:
            Warning(tk.Toplevel(self), "Warning", "An error occured")
# registration.py

import cv2
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

class Registration:
    def __init__(self, button_frame):
        self.button_frame = button_frame
        self.video_capture = None
        self.is_camera_running = False

    def register_user(self):
        # Clear existing buttons
        for widget in self.button_frame.winfo_children():
            widget.destroy()

        # Registration Frame
        registration_frame = tk.Frame(self.button_frame, bg='#f0f0f0')
        registration_frame.pack(expand=True, fill='both')

        # Name Entry
        tk.Label(registration_frame, text="Name:", bg='#f0f0f0').pack(pady=5)
        self.name_entry = tk.Entry(registration_frame, width=30)
        self.name_entry.pack(pady=5)

        # User ID Entry
        tk.Label(registration_frame, text="User ID:", bg='#f0f0f0').pack(pady=5)
        self.id_entry = tk.Entry(registration_frame, width=30)
        self.id_entry.pack(pady=5)

        # Camera Display Area
        self.camera_label = tk.Label(registration_frame, width=400, height=300)
        self.camera_label.pack(pady=10)

        # Capture Button
        capture_button = tk.Button(
            registration_frame,
            text="Capture Face",
            command=self.capture_face,
            width=20,
            height=2,
            bg='#4CAF50',
            fg='white'
        )
        capture_button.pack(pady=5)

        # Back Button
        back_button = tk.Button(
            registration_frame,
            text="Back to Main Menu",
            command=self.reset_main_menu,
            width=20,
            height=2,
            bg='#FF6347',
            fg='white'
        )
        back_button.pack(pady=10)

        # Start camera when registration page opens
        self.start_camera()

    def start_camera(self):
        # Open video capture
        self.video_capture = cv2.VideoCapture(0)
        self.is_camera_running = True
        self.update_camera_frame()

    def update_camera_frame(self):
        if self.is_camera_running:
            ret, frame = self.video_capture.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (400, 300))
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)

            self.camera_label.after(10, self.update_camera_frame)

    def capture_face(self):
        name = self.name_entry.get()
        user_id = self.id_entry.get()
        
        if not name or not user_id:
            messagebox.showerror("Input Error", "Please enter both Name and User ID.")
            return

        user_image_dir = f"TrainingImage/{user_id}_{name}"
        os.makedirs(user_image_dir, exist_ok=True)

        ret, frame = self.video_capture.read()
        
        if ret:
            image_path = f"{user_image_dir}/captured_face.jpg"
            cv2.imwrite(image_path, frame)
            messagebox.showinfo("Success", "Face captured successfully!")

    def reset_main_menu(self):
        # Stop camera and release resources
        if self.is_camera_running:
            self.is_camera_running = False
            if self.video_capture:
                self.video_capture.release()

        for widget in self.button_frame.winfo_children():
            widget.destroy()

# Usage example in main.py will be shown next.

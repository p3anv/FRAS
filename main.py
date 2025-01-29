import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import os
import face_recognition

# Correct imports
from face_recognition.mark_attendance import AttendanceMark
from face_recognition.train_faces import train_images

class AttendanceSystemApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Recognition Attendance System")
        self.root.geometry("900x600")
        self.root.configure(bg='#f0f0f0')

        # Title Label
        self.title_label = tk.Label(
            root, 
            text="Facial Recognition Attendance System", 
            font=("Arial", 20, "bold"),
            bg='#f0f0f0',
            fg='#333333'
        )
        self.title_label.pack(pady=20)

        # Buttons Frame
        self.button_frame = tk.Frame(root, bg='#f0f0f0')
        self.button_frame.pack(expand=True)

        # Camera capture setup
        self.video_capture = None
        self.is_camera_running = False
        self.camera_label = None

        # Initial Buttons
        self.create_main_buttons()

    def create_main_buttons(self):
        # Clear any existing buttons
        for widget in self.button_frame.winfo_children():
            widget.destroy()

        buttons = [
            ("Register New User", self.open_registration),
            ("Mark Attendance", self.mark_attendance),
            ("View Attendance", self.view_attendance),
            ("Exit", self.exit_application)
        ]

        for text, command in buttons:
            btn = tk.Button(
                self.button_frame, 
                text=text, 
                command=command,
                width=20,
                height=2,
                font=("Arial", 12),
                bg='#4CAF50',
                fg='white',
                activebackground='#45a049'
            )
            btn.pack(pady=10)

    def open_registration(self):
        # Clear existing buttons
        for widget in self.button_frame.winfo_children():
            widget.destroy()

        # Registration Frame
        registration_frame = tk.Frame(self.button_frame, bg='#f0f0f0')
        registration_frame.pack(expand=True, fill='both')

        # Name Entry
        tk.Label(registration_frame, text="Name:", bg='#f0f0f0').pack(pady=5)
        name_entry = tk.Entry(registration_frame, width=30)
        name_entry.pack(pady=5)

        # User ID Entry
        tk.Label(registration_frame, text="User ID:", bg='#f0f0f0').pack(pady=5)
        id_entry = tk.Entry(registration_frame, width=30)
        id_entry.pack(pady=5)

        # Camera Display Area
        self.camera_label = tk.Label(registration_frame, width=400, height=300)
        self.camera_label.pack(pady=10)

        # Capture Button
        capture_button = tk.Button(
            registration_frame, 
            text="Capture Face", 
            command=lambda: self.capture_face(name_entry.get(), id_entry.get()),
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
        if self.is_camera_running and self.camera_label is not None:
            # Read frame from camera
            ret, frame = self.video_capture.read()
            
            if ret:
                # Convert frame to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame
                frame = cv2.resize(frame, (400, 300))
                
                # Convert to PhotoImage
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Update label
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
            
            # Schedule next update
            self.camera_label.after(10, self.update_camera_frame)

    def capture_face(self, name, user_id):
        if not name or not user_id:
            messagebox.showerror("Input Error", "Please enter both Name and User ID.")
            return

        user_image_dir = f"TrainingImage/{user_id}_{name}"
        os.makedirs(user_image_dir, exist_ok=True)

        # Capture frame
        ret, frame = self.video_capture.read()
        
        if ret:
            # Save the image
            image_path = f"{user_image_dir}/captured_face.jpg"
            cv2.imwrite(image_path, frame)
            messagebox.showinfo("Success", "Face captured successfully!")

    def reset_main_menu(self):
        # Stop camera
        self.is_camera_running = False
        if self.video_capture:
            self.video_capture.release()

        # Clear registration frame
        for widget in self.button_frame.winfo_children():
            widget.destroy()

        # Recreate original buttons
        self.create_main_buttons()

    def mark_attendance(self):
        try:
            # Train images and get encodings
            known_face_encodings, known_face_names = train_images()
            
            # Check if any faces are trained
            if not known_face_encodings:
                messagebox.showwarning("Warning", "No faces have been trained yet!")
                return

            # Create attendance marker instance and start attendance marking
            attendance_marker = AttendanceMark(known_face_encodings, known_face_names)
            attendance_marker.start_attendance()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def view_attendance(self):
        messagebox.showinfo("View Attendance", "Attendance View Module (Under Development)")

    def exit_application(self):
        # Release camera before quitting
        if hasattr(self, 'video_capture') and self.video_capture:
            self.video_capture.release()
        
        if messagebox.askokcancel("Exit", "Do you want to exit the application?"):
            self.root.quit()

def main():
    root = tk.Tk()
    app = AttendanceSystemApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

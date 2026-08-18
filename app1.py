import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import cv2
from datetime import datetime
from model import load_model, recognize_face, register_user
from database import initialize_db, load_all_embeddings

# GUI config
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Smart Face Attendance System")
root.attributes("-fullscreen", True)

frame = ctk.CTkFrame(root, corner_radius=15)
frame.pack(pady=20, padx=20, fill="both", expand=True)

# Global directories
user_data_dir = r"C:\Users\Latitude\Desktop\FYP\user_folder"
training_data_dir = r"C:\Users\Latitude\Desktop\FYP\data_folder"

# ------------------ Auto-register known users from filename ------------------
def auto_register_known_users():
    global user_data_dir
    name_map = {
        "imgs.jpeg": "Alishba",
        # Add more mappings if needed: "bob_img.png": "Bob"
    }

    if not os.path.exists(user_data_dir):
        print("User folder does not exist.")
        return

    app = load_model()
    initialize_db()

    registered = load_all_embeddings()
    already_registered = {v['name'] for v in registered.values()}
    print("Already in DB:", already_registered)

    for file in os.listdir(user_data_dir):
        if file in name_map:
            name = name_map[file]
            if name in already_registered:
                print(f"{name} already registered, skipping.")
                continue
            path = os.path.join(user_data_dir, file)
            print(f"Registering {name} from {file}...")
            register_user(name, path, app)

# ------------------ Folder Selection ------------------
def select_user_data_folder():
    global user_data_dir
    user_data_dir = filedialog.askdirectory(title="Select Folder for Registered Users + Attendance")
    if user_data_dir:
        messagebox.showinfo("Folder Selected", f"User data folder set:\n{user_data_dir}")
        auto_register_known_users()

def select_training_data_folder():
    global training_data_dir
    training_data_dir = filedialog.askdirectory(title="Select Training Dataset Folder")
    messagebox.showinfo("Training Folder", f"Training dataset folder set:\n{training_data_dir}")

# ------------------ View Registered ------------------
def view_registered_images():
    if not user_data_dir:
        messagebox.showwarning("Select Folder", "Please select the user folder first.")
        return
    if not os.path.exists(user_data_dir):
        messagebox.showerror("Folder Not Found", "The selected user folder does not exist.")
        return
    try:
        os.startfile(user_data_dir)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open folder:\n{str(e)}")

# ------------------ Register New ------------------
def register_new_user():
    global user_data_dir
    if not user_data_dir:
        messagebox.showerror("No Folder", "Select a folder to save user data first.")
        return

    app = load_model()
    initialize_db()

    name = ctk.CTkInputDialog(title="Register New User", text="Enter new user name:").get_input()
    if not name:
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Camera not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.putText(frame, "Press S = Save | Q = Quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Capture New User", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            img_path = os.path.join(user_data_dir, f"{name}.jpg")
            cv2.imwrite(img_path, frame)
            register_user(name, img_path, app)
            messagebox.showinfo("Saved", f"User '{name}' registered successfully.")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------ Start Recognition ------------------
def start_recognition():
    if not user_data_dir:
        messagebox.showerror("Missing Folder", "Please select the user image folder.")
        return

    initialize_db()
    app = load_model()
    db_embeddings = load_all_embeddings()

    if not db_embeddings:
        messagebox.showwarning("No Users", "No users found in database.")
        print("Database is empty: No embeddings loaded.")
        return

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            emb = face.embedding
            match, name, sim, marked = recognize_face(emb, db_embeddings)
            label = f"{name} ({sim:.2f})" if match else "Unknown"
            color = (0, 255, 0) if match else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if marked:
                print(f"Attendance marked for {name} (Similarity: {sim:.2f})")
            elif match:
                print(f"{name} already marked today (Similarity: {sim:.2f})")
            else:
                print("Unknown face detected.")

        cv2.imshow("Face Recognition Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------ Exit App ------------------
def exit_app():
    root.destroy()

# ------------------ Build GUI ------------------
ctk.CTkLabel(frame, text="Smart Face Recognition Attendance System", font=("Helvetica", 28, "bold")).pack(pady=25)

buttons = [
    ("Select Registered + Attendance Folder", select_user_data_folder),
    ("Select Training Dataset Folder", select_training_data_folder),
    ("View Registered User Images", view_registered_images),
    ("Register New User (via Camera)", register_new_user),
    ("Start Face Recognition", start_recognition),
    ("Exit", exit_app)
]

for text, command in buttons:
    ctk.CTkButton(frame, text=text, font=("Helvetica", 18), command=command, height=50).pack(pady=10, padx=40, fill="x")

root.mainloop()

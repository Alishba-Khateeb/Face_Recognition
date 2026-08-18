# Face Recognition Attendance System

A Python-based face recognition project for automatic attendance tracking using webcam input, facial embeddings, and SQLite storage.

This project detects faces from a live camera feed, compares them to registered users using InsightFace embeddings, and marks attendance only once per person per day.

## Features

- Real-time face detection with OpenCV
- Face recognition using InsightFace (`buffalo_l` model)
- Embedding-based comparison using cosine similarity
- SQLite database for storing user embeddings and attendance logs
- GUI-based attendance workflow with `customtkinter`
- Daily attendance enforcement to prevent duplicate check-ins
- Support for registering new users from a camera feed

## Project Structure

```text
Face_Recognition/
├── app1.py                  # GUI-based attendance application
├── main.py                  # CLI-style registration + recognition demo
├── model.py                 # Face embedding and recognition logic
├── database.py              # SQLite database operations
├── attendance.db            # Local SQLite database file
├── data_folder/             # Training / dataset folder
├── user_folder/             # User images and registration directory
├── notebook.ipynb           # Notebook experiments
├── single-embedding.ipynb   # Embedding-related experiments
├── updated_code.ipynb       # Additional notebook work
├── .env                     # Environment configuration
└── README.md                # Project documentation
```

## Tech Stack

- Python 3
- OpenCV (`opencv-python`)
- InsightFace (`insightface`)
- scikit-learn (`scikit-learn`)
- SQLite
- customtkinter (GUI)

## Prerequisites

Before running this project, make sure you have:

- Python 3.9+
- A working webcam or camera device
- pip installed
- Visual C++ build tools if required by some dependencies on Windows

## Installation

1. Clone or open the project folder.
2. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or
.venv\Scripts\activate      # Windows
```

3. Install the dependencies:

```bash
pip install opencv-python insightface scikit-learn customtkinter
```

If you are using special platform-specific requirements, install any additional packages as needed for your system.

## Database

The project stores users and attendance records in SQLite using the file:

```text
attendance.db
```

The database includes two tables:

- `users`: stores user names and face embeddings
- `attendance`: stores attendance timestamps per user per day

Attendance is only marked once per user per day.

## How It Works

1. A user is registered by extracting a face embedding from an image.
2. The embedding is saved in the SQLite database.
3. The webcam captures live frames.
4. InsightFace detects one or more faces in the frame.
5. The system compares each detected face embedding with stored embeddings.
6. If similarity exceeds the threshold, the user is recognized.
7. Attendance is recorded if it has not already been marked that day.

## Running the Project

### Option 1: GUI Version

Run:

```bash
python app1.py
```

This starts a GUI application with buttons to:

- select the registered user folder
- select the training dataset folder
- view registered images
- register a new user via camera
- start live recognition
- exit the app

### Option 2: Command-Line Version

Run:

```bash
python main.py
```

This script initializes the database, loads the face model, registers sample users, and starts webcam-based recognition.

## Important Notes

- The current code includes hardcoded local Windows paths in some places, for example in `main.py` and `app1.py`.
- You should update those paths to match your own environment before running the project.
- For example, `user_folder` and `data_folder` should point to valid local directories on your machine.

## Example Usage Flow

1. Update the file paths in `app1.py` or `main.py` to match your machine.
2. Run the GUI or CLI version.
3. Register at least one user using a face image or webcam capture.
4. Start recognition.
5. When a registered face is detected, it will be checked against the database.
6. Attendance is logged to `attendance.db`.

## Notes on Accuracy and Model Loading

- The project uses the `buffalo_l` InsightFace model.
- Recognition quality depends on:
  - lighting conditions
  - camera quality
  - face angle / orientation
  - image quality of registered user photos
- A stable front-facing image typically yields the best recognition results.

## Common Issues

### Camera not opening

- Ensure no other app is already using your webcam.
- Check that your system has permission to access the camera.

### No face detected

- Use a clearer image for registration.
- Ensure the face is clearly visible and well-lit.

### File paths not found

- Update the hardcoded paths in the project files to valid directories on your system.



## Future Improvements

- add user authentication or admin panel
- export attendance reports as CSV/Excel
- support multiple cameras
- improve recognition threshold tuning
- add web-based dashboard
- include a training pipeline for larger datasets

## Acknowledgements

This project uses:

- OpenCV for image processing
- InsightFace for face detection and embeddings
- SQLite for local attendance storage


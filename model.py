# model.py

import cv2
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from database import insert_user, mark_attendance

# Load the face analysis model from InsightFace
def load_model():
    app = FaceAnalysis(name='buffalo_l')  # You can change model name if needed
    app.prepare(ctx_id=0)  # Use 0 for GPU, -1 for CPU
    return app

# Extract embedding from a given image using the model
def extract_embedding(image_path, app):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image from path: {image_path}")
        return None
    faces = app.get(img)
    if faces:
        return faces[0].embedding
    return None

# Register a new user by extracting their face embedding and saving to the database
def register_user(name, image_path, app):
    embedding = extract_embedding(image_path, app)
    if embedding is not None:
        insert_user(name, embedding)
        print(f"User '{name}' registered successfully.")
    else:
        print(f"No face detected for '{name}' in the image.")

# Compare input embedding to stored embeddings and return match result
def recognize_face(embedding, db_embeddings, threshold=0.5):
    for user_id, data in db_embeddings.items():
        sim = cosine_similarity([embedding], [data["embedding"]])[0][0]
        if sim > threshold:
            marked, _ = mark_attendance(user_id)
            return True, data["name"], sim, marked
    return False, "Unknown", 0, False

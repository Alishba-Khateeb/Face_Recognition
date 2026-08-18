import cv2
from database import initialize_db, load_all_embeddings
from model import load_model, register_user, recognize_face

if __name__ == "__main__":
    initialize_db()
    app = load_model()

    # Register users (uncomment these lines to add users)
    register_user("Alice", r"C:\Users\Latitude\PyCharmMiscProject\2ndimg.PNG", app)
    # register_user("Bob", r"C:\Users\Latitude\PyCharmMiscProject\new.PNG", app)
    register_user("Alishba", r"C:\Users\Latitude\Desktop\FYP\user_folder\imgs.jpeg", app)

    db_embeddings = load_all_embeddings()

    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            emb = face.embedding
            match, name, sim, marked = recognize_face(emb, db_embeddings)

            if marked:
                print(f"Attendance marked for {name} (Similarity: {sim:.2f})")
            elif match:
                print(f"{name}'s attendance already marked today (Similarity: {sim:.2f})")
            else:
                print("Unknown face detected.")

            # Draw bounding box and label
            color = (0, 255, 0) if match else (0, 0, 255)
            label = f"{name} ({sim:.2f})" if match else "Unknown"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Face Recognition Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


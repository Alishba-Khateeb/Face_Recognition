import sqlite3
import pickle
from datetime import datetime

DB_PATH = "attendance.db"


# Initialize the database and tables

def initialize_db(db_path=DB_PATH):
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()

        # Table for users with embeddings
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        ''')

        # Table for attendance logs with name
        c.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')

        conn.commit()


# Insert a new user and their embedding

def insert_user(name, embedding, db_path=DB_PATH):
    embedding_blob = pickle.dumps(embedding)  # Serialize to BLOB
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute('INSERT INTO users (name, embedding) VALUES (?, ?)', (name, embedding_blob))
        conn.commit()

# Check if user already exists by name

def user_exists(name, db_path=DB_PATH):
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM users WHERE name = ?', (name,))
        return c.fetchone()[0] > 0


# Load all user embeddings from the database

def load_all_embeddings(db_path=DB_PATH):
    embeddings_dict = {}
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute('SELECT id, name, embedding FROM users')
        rows = c.fetchall()
        for user_id, name, blob in rows:
            emb = pickle.loads(blob)
            embeddings_dict[user_id] = {"name": name, "embedding": emb}
    return embeddings_dict

# Mark attendance (only once per user per day)

def mark_attendance(user_id, db_path=DB_PATH):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today_date = datetime.now().strftime("%Y-%m-%d")

    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()

        c.execute('SELECT name FROM users WHERE id = ?', (user_id,))
        result = c.fetchone()
        if not result:
            return False, None
        name = result[0]

        c.execute('''
            SELECT COUNT(*) FROM attendance
            WHERE user_id = ? AND DATE(timestamp) = ?
        ''', (user_id, today_date))
        already_marked = c.fetchone()[0]

        if not already_marked:
            c.execute(
                'INSERT INTO attendance (user_id, name, timestamp) VALUES (?, ?, ?)',
                (user_id, name, timestamp)
            )
            conn.commit()
            return True, name  # Attendance was marked

        return False, name  # Already marked

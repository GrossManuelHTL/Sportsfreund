import sqlite3

DB_PATH = "data/box.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audio_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

def add_audio_file(filename):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("INSERT INTO audio_files (filename) VALUES (?)", (filename,))

    conn.commit()
    conn.close()

def get_audio_files():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM audio_files")
    files = cursor.fetchall()

    conn.close()
    return files

if __name__ == "__main__":
    init_db()

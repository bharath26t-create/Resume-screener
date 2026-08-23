import sqlite3
import json

DB_PATH = "screener.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            resume_data TEXT,
            job_description TEXT,
            score INTEGER,
            justification TEXT,
            predicted_category TEXT,
            category_confidence REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_candidate(filename, resume_data, job_description, score, justification,
                    predicted_category=None, category_confidence=None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO candidates
           (filename, resume_data, job_description, score, justification,
            predicted_category, category_confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (filename, json.dumps(resume_data), job_description, score, justification,
         predicted_category, category_confidence),
    )
    conn.commit()
    conn.close()


def get_all_candidates(min_score=0):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM candidates WHERE score >= ? ORDER BY score DESC", (min_score,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]

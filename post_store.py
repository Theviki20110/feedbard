import sqlite3
from contextlib import closing

DB_PATH = "post_store.sqlite3"


def init_db(db_path: str = DB_PATH) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seen_posts (
                feed_url TEXT NOT NULL,
                post_url TEXT NOT NULL,
                fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (feed_url, post_url)
            )
            """
        )
        conn.commit()


def is_post_seen(feed_url: str, post_url: str, db_path: str = DB_PATH) -> bool:
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT 1 FROM seen_posts WHERE feed_url = ? AND post_url = ?",
            (feed_url, post_url),
        ).fetchone()
        return row is not None


def mark_post_seen(feed_url: str, post_url: str, db_path: str = DB_PATH) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO seen_posts (feed_url, post_url) VALUES (?, ?)",
            (feed_url, post_url),
        )
        conn.commit()


def filter_new_posts(feed_url: str, post_urls: list[str], db_path: str = DB_PATH) -> list[str]:
    with closing(sqlite3.connect(db_path)) as conn:
        seen = {
            row[0]
            for row in conn.execute(
                "SELECT post_url FROM seen_posts WHERE feed_url = ?", (feed_url,)
            )
        }
    return [url for url in post_urls if url not in seen]

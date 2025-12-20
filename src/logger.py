from datetime import datetime

def log_event(message):
    """
    Write audit logs with timestamp
    """
    with open("logs/test_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] {message}\n")

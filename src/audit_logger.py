from datetime import datetime

def log_action(user, action, doc_id):
    with open("audit.log", "a") as f:
        f.write(
            f"{datetime.now()} | USER={user} | ACTION={action} | DOC_ID={doc_id}\n"
        )

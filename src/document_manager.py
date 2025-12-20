import os
import uuid
import json
from datetime import datetime

BASE_STORAGE = "storage"

def save_document(file_path, company_id):
    filename = os.path.basename(file_path)

    company_folder = os.path.join(BASE_STORAGE, company_id, "documents")
    os.makedirs(company_folder, exist_ok=True)

    doc_id = str(uuid.uuid4())
    new_file_path = os.path.join(company_folder, f"{doc_id}_{filename}")

    with open(file_path, "rb") as src, open(new_file_path, "wb") as dst:
        dst.write(src.read())

    metadata = {
        "doc_id": doc_id,
        "filename": filename,
        "company_id": company_id,
        "stored_path": new_file_path,
        "uploaded_at": str(datetime.now())
    }

    metadata_path = os.path.join(company_folder, f"{doc_id}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return metadata

def extract_structured_data(doc_type, text):
    """
    Extract structured fields based on document type
    """

    if doc_type == "INVOICE":
        return {
            "Invoice Number": "Detected from text",
            "Total Amount": "Detected from text",
            "Invoice Date": "Detected from text",
            "Vendor": "Detected from text"
        }

    elif doc_type == "RESUME":
        return {
            "Name": "Detected from text",
            "Skills": "Detected from text",
            "Experience": "Detected from text"
        }

    elif doc_type == "POLICY":
        return {
            "Policy Title": "Detected from text",
            "Effective Date": "Detected from text"
        }

    else:
        return {}

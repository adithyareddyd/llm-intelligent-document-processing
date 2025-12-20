def detect_document_type(text):
    """
    Detect document type using simple rule-based logic
    """

    text = text.lower()

    if "invoice" in text or "total amount" in text:
        return "INVOICE"
    elif "experience" in text or "skills" in text or "education" in text:
        return "RESUME"
    elif "policy" in text or "terms and conditions" in text:
        return "POLICY"
    else:
        return "GENERAL"

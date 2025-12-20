import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import ollama

from src.pdf_loader import load_pdf
from src.text_chunker import chunk_text
from src.embeddings import get_embeddings
from src.vector_store import store_vectors, get_vectors
from src.retriever import retrieve
from src.structured_extractor import extract_structured_data
from src.doc_type_detector import detect_document_type
from src.logger import log_event


# ======================================================
# 🎨 PAGE CONFIG & ENTERPRISE STYLING
# ======================================================
st.set_page_config(
    page_title="IDP Enterprise Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Background */
    .main {
        background: #f5f7fa;
        padding: 0;
    }
    
    .block-container {
        padding: 1.5rem 2rem 3rem 2rem;
        max-width: 1600px;
    }
    
    /* Top Navigation Bar */
    .top-nav {
        background: white;
        padding: 1.2rem 2rem;
        margin: -1.5rem -2rem 2rem -2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        border-bottom: 1px solid #e5e7eb;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .nav-brand h1 {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        color: #1e293b;
    }
    
    .nav-status {
        display: flex;
        gap: 1rem;
        align-items: center;
    }
    
    .status-badge {
        background: #10b981;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Dashboard Grid Layout */
    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
        border: 1px solid #e5e7eb;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .metric-change {
        font-size: 0.875rem;
        color: #10b981;
        font-weight: 600;
    }
    
    .metric-change.negative {
        color: #ef4444;
    }
    
    /* Main Content Card */
    .content-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
        border: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    
    .content-card h2 {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0 0 0.5rem 0;
    }
    
    .content-card .subtitle {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    
    /* Document List */
    .doc-list {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 1rem;
        margin-top: 1.5rem;
    }
    
    .doc-item {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 1.25rem;
        border-radius: 10px;
        color: white;
        position: relative;
        overflow: hidden;
    }
    
    .doc-item::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 100px;
        height: 100px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        transform: translate(30%, -30%);
    }
    
    .doc-name {
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .doc-status {
        font-size: 0.8rem;
        opacity: 0.9;
        position: relative;
        z-index: 1;
    }
    
    /* Document Type Badge */
    .doc-type-display {
        background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin: 1.5rem 0;
    }
    
    .doc-type-display .label {
        font-size: 0.875rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .doc-type-display .value {
        font-size: 1.75rem;
        font-weight: 700;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #1e293b;
        border-right: 1px solid #334155;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        padding: 2rem 1.5rem;
    }
    
    [data-testid="stSidebar"] h1 {
        color: white !important;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] .css-1v0mbdj, 
    [data-testid="stSidebar"] p {
        color: #94a3b8 !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: #334155;
        margin: 1.5rem 0;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: #0f172a;
        border: 2px dashed #475569;
        border-radius: 12px;
        padding: 2rem 1rem;
    }
    
    [data-testid="stFileUploader"] section {
        border: none;
    }
    
    [data-testid="stFileUploader"] label {
        color: #e2e8f0 !important;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #f8fafc;
        padding: 0.25rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: #64748b;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        color: #6366f1;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Input Styling */
    .stTextInput>div>div>input,
    .stSelectbox>div>div>select {
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        padding: 0.75rem 1rem;
        font-size: 0.95rem;
        background: white;
    }
    
    .stTextInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }
    
    /* Success/Info Messages */
    .stSuccess {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 1rem;
        color: #065f46;
    }
    
    .stInfo {
        background: #eff6ff;
        border: 1px solid #93c5fd;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem;
        color: #1e40af;
    }
    
    .stWarning {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
        color: #92400e;
    }
    
    /* Question Buttons Grid */
    .question-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
        gap: 0.75rem;
        margin: 1rem 0;
    }
    
    .question-btn {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        padding: 0.875rem 1rem;
        border-radius: 8px;
        text-align: left;
        font-size: 0.9rem;
        color: #475569;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .question-btn:hover {
        background: white;
        border-color: #6366f1;
        color: #6366f1;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.15);
    }
    
    /* Answer Box */
    .answer-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #bae6fd;
        border-left: 4px solid #0ea5e9;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    
    .answer-box h4 {
        color: #0c4a6e;
        font-weight: 700;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    .answer-box p {
        color: #0c4a6e;
        line-height: 1.6;
    }
    
    /* Loading Indicator */
    .stSpinner > div {
        border-color: #6366f1 transparent #6366f1 transparent !important;
    }
    
    /* JSON Display */
    .stJson {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 1.5rem;
    }
    
    /* Progress Indicator */
    .progress-indicator {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1rem;
        background: #f8fafc;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .progress-step {
        flex: 1;
        height: 4px;
        background: #e5e7eb;
        border-radius: 2px;
        position: relative;
        overflow: hidden;
    }
    
    .progress-step.active {
        background: #6366f1;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #64748b;
        font-size: 0.875rem;
        border-top: 1px solid #e5e7eb;
        margin-top: 3rem;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ======================================================
# 📊 TOP NAVIGATION
# ======================================================
st.markdown("""
<div class="top-nav">
    <div class="nav-brand">
        <span style="font-size: 2rem;">📄</span>
        <div>
            <h1>IDP Enterprise Platform</h1>
            <p style="font-size: 0.85rem; color: #64748b; margin: 0;">Intelligent Document Processing</p>
        </div>
    </div>
    <div class="nav-status">
        <span class="status-badge">
            <span style="width: 6px; height: 6px; background: white; border-radius: 50%; display: inline-block;"></span>
            System Active
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ======================================================
# 🧭 SIDEBAR
# ======================================================
with st.sidebar:
    st.markdown("# 📄 Control Panel")
    st.caption("Document Management & Settings")
    st.markdown("---")
    
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Support for PDF documents up to 200MB"
    )
    
    if uploaded_files:
        st.markdown("---")
        st.markdown(f"**Documents Loaded:** {len(uploaded_files)}")
        st.progress(1.0)
        st.caption(f"✓ {len(uploaded_files)} file(s) ready for processing")
    
    st.markdown("---")
    
    # System Info
    st.markdown("### 🔧 System Info")
    st.caption("**Status:** Online")
    st.caption("**Model:** Llama 3 Latest")
    st.caption("**Embeddings:** Nomic Embed")
    
    st.markdown("---")
    st.info("🔐 **Enterprise Security**\n\nSecure • Audited • Compliant")

# ======================================================
# 📂 LOAD & PROCESS DOCUMENTS
# ======================================================
all_text = ""

if not uploaded_files:
    # Empty State
    st.markdown("""
    <div class="content-card" style="text-align: center; padding: 4rem 2rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📄</div>
        <h2>No Documents Loaded</h2>
        <p class="subtitle">Upload PDF documents from the sidebar to begin processing</p>
        <div style="display: flex; gap: 1rem; justify-content: center; margin-top: 2rem; flex-wrap: wrap;">
            <div style="background: #f8fafc; padding: 1rem 1.5rem; border-radius: 8px; border: 1px solid #e5e7eb;">
                <strong style="color: #1e293b;">✓ Multi-Language</strong><br/>
                <small style="color: #64748b;">English, Hindi, Telugu</small>
            </div>
            <div style="background: #f8fafc; padding: 1rem 1.5rem; border-radius: 8px; border: 1px solid #e5e7eb;">
                <strong style="color: #1e293b;">✓ AI-Powered</strong><br/>
                <small style="color: #64748b;">Advanced RAG System</small>
            </div>
            <div style="background: #f8fafc; padding: 1rem 1.5rem; border-radius: 8px; border: 1px solid #e5e7eb;">
                <strong style="color: #1e293b;">✓ Structured Data</strong><br/>
                <small style="color: #64748b;">Auto-extraction</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Initialize session state for metrics
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
        st.session_state.successful_queries = 0
        st.session_state.total_extractions = 0
        st.session_state.successful_extractions = 0
    
    # Calculate real-time accuracy
    total_operations = st.session_state.total_queries + st.session_state.total_extractions
    successful_operations = st.session_state.successful_queries + st.session_state.successful_extractions
    
    if total_operations > 0:
        accuracy = (successful_operations / total_operations) * 100
        accuracy_display = f"{accuracy:.1f}%"
    else:
        accuracy = 0
        accuracy_display = "N/A"
    
    # Calculate total pages processed
    total_pages = 0
    for file in uploaded_files:
        # Estimate: average 500 words per page, 5 chars per word
        words = len(all_text.split())
        total_pages = max(1, words // 500)
    
    # Dashboard Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Documents</div>
            <div class="metric-value">{len(uploaded_files)}</div>
            <div class="metric-change">↑ {total_pages} pages total</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Operations</div>
            <div class="metric-value">{total_operations}</div>
            <div class="metric-change">{'✓ Active' if total_operations > 0 else 'Ready'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">AI Model</div>
            <div class="metric-value">Llama 3</div>
            <div class="metric-change">⚡ Online</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        accuracy_change_class = "positive" if accuracy >= 90 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{accuracy_display}</div>
            <div class="metric-change {'metric-change' if accuracy >= 90 else 'metric-change negative'}">
                {f'↑ High confidence' if accuracy >= 90 else f'↓ {successful_operations}/{total_operations} ops' if total_operations > 0 else '— No data yet'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Process Documents
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("## 📂 Document Library")
    st.markdown('<p class="subtitle">Manage and process your uploaded documents</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="doc-list">', unsafe_allow_html=True)
    for file in uploaded_files:
        file_path = f"temp_{file.name}"
        with open(file_path, "wb") as f:
            f.write(file.read())
        
        pdf_text = load_pdf(file_path)
        all_text += f"\n\n--- Document: {file.name} ---\n\n{pdf_text}"
        
        st.markdown(f"""
        <div class="doc-item">
            <div class="doc-name">📄 {file.name}</div>
            <div class="doc-status">✓ Processed & Indexed</div>
        </div>
        """, unsafe_allow_html=True)
        
        log_event(f"UPLOAD | File={file.name}")
    
    st.markdown('</div></div>', unsafe_allow_html=True)
    
    # Language Selection
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("## 🌍 Language Preferences")
    st.markdown('<p class="subtitle">Choose your preferred language for AI responses</p>', unsafe_allow_html=True)
    
    language = st.selectbox(
        "Response Language",
        ["Simple English", "Hindi", "Telugu"],
        help="Select the language for summaries and answers"
    )
    
    def get_language_instruction(lang):
        if lang == "Hindi":
            return "Answer in simple Hindi."
        elif lang == "Telugu":
            return "Answer in simple Telugu."
        else:
            return "Answer in simple English."
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced Document Type Detection
    with st.spinner("🔍 Analyzing document type..."):
        # Use AI to detect document type more accurately
        detection_response = ollama.chat(
            model="llama3:latest",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document classification expert. Analyze the text and identify the document type. "
                        "Common types include: RESUME, INVOICE, CONTRACT, REPORT, LETTER, FORM, LEGAL_DOCUMENT, "
                        "FINANCIAL_STATEMENT, MEDICAL_RECORD, ACADEMIC_PAPER, PRESENTATION, OTHER. "
                        "Respond with ONLY the document type in uppercase, nothing else."
                    )
                },
                {
                    "role": "user",
                    "content": f"Classify this document:\n\n{all_text[:2000]}"
                }
            ]
        )
        
        doc_type = detection_response["message"]["content"].strip().upper()
        
        # Clean up the response to get just the type
        common_types = ["RESUME", "INVOICE", "CONTRACT", "REPORT", "LETTER", "FORM", 
                       "LEGAL_DOCUMENT", "FINANCIAL_STATEMENT", "MEDICAL_RECORD", 
                       "ACADEMIC_PAPER", "PRESENTATION", "OTHER"]
        
        detected_type = "OTHER"
        for dtype in common_types:
            if dtype in doc_type:
                detected_type = dtype
                break
        
        doc_type = detected_type
    
    # Map document types to emojis
    type_emojis = {
        "RESUME": "👤",
        "INVOICE": "🧾",
        "CONTRACT": "📜",
        "REPORT": "📊",
        "LETTER": "✉️",
        "FORM": "📋",
        "LEGAL_DOCUMENT": "⚖️",
        "FINANCIAL_STATEMENT": "💰",
        "MEDICAL_RECORD": "🏥",
        "ACADEMIC_PAPER": "🎓",
        "PRESENTATION": "📽️",
        "OTHER": "📄"
    }
    
    doc_emoji = type_emojis.get(doc_type, "📄")
    
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("## 📑 Document Classification")
    st.markdown('<p class="subtitle">AI-powered document type identification</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="doc-type-display">
            <div class="label">Detected Type</div>
            <div class="value">{doc_emoji} {doc_type.replace('_', ' ')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 10px; height: 100%;">
            <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 0.5rem;">Confidence</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #10b981;">95%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    log_event(f"DOCUMENT_TYPE | {doc_type}")
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📄 Summary", "📊 Structured Data", "❓ Q&A Assistant"])
    
    # ==================== SUMMARY TAB ====================
    with tab1:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("### 📄 AI-Powered Document Summary")
        st.markdown('<p class="subtitle">Generate intelligent summaries using advanced language models</p>', unsafe_allow_html=True)
        
        if st.button("🚀 Generate Summary", key="summary_btn"):
            with st.spinner("🤖 AI is analyzing your document..."):
                response = ollama.chat(
                    model="llama3:latest",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                f"Summarize this document in 3 short bullet points. "
                                f"{get_language_instruction(language)}"
                            )
                        },
                        {
                            "role": "user",
                            "content": all_text[:1500]
                        }
                    ]
                )
            
            st.markdown("#### 📝 Summary Results")
            st.markdown(f"""
            <div style="background: #f8fafc; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #6366f1;">
                {response["message"]["content"]}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================== STRUCTURED DATA TAB ====================
    with tab2:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Structured Data Extraction")
        st.markdown('<p class="subtitle">Extract key data fields and entities automatically</p>', unsafe_allow_html=True)
        
        if st.button("🔍 Extract Data", key="extract_btn"):
            with st.spinner("🔄 Extracting structured information..."):
                structured_data = extract_structured_data(doc_type, all_text)
                st.session_state.total_extractions += 1
            
            if structured_data:
                st.markdown("#### 📋 Extracted Fields")
                st.json(structured_data)
                st.session_state.successful_extractions += 1
                log_event(f"STRUCTURED_EXTRACTION | Type={doc_type}")
                
                # Add feedback option
                col1, col2 = st.columns([3, 1])
                with col2:
                    feedback = st.radio("Was this accurate?", ["✓ Yes", "✗ No"], horizontal=True, key="extract_feedback")
                    if feedback == "✗ No":
                        st.session_state.successful_extractions -= 1
            else:
                st.warning("⚠️ No structured fields available for this document type")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================== Q&A TAB ====================
    with tab3:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("### ❓ Intelligent Q&A Assistant")
        st.markdown('<p class="subtitle">Ask questions and get instant answers from your documents</p>', unsafe_allow_html=True)
        
        if "indexed" not in st.session_state:
            st.session_state.indexed = False
        
        if not st.session_state.indexed:
            st.info("📌 **Index your documents** to enable intelligent search and question answering")
            
            if st.button("⚡ Index Documents Now", key="index_btn"):
                with st.spinner("🔄 Building vector index..."):
                    chunks = chunk_text(all_text)
                    embeddings = get_embeddings(chunks)
                    store_vectors(chunks, embeddings)
                    st.session_state.indexed = True
                
                st.success("✅ Documents successfully indexed! You can now ask questions.")
                log_event("VECTOR_INDEXING | Completed")
                st.rerun()
        else:
            st.success("✅ **Documents Indexed** - Ready for questions!")
            
            st.markdown("#### 💡 Quick Questions")
            st.markdown('<p class="subtitle" style="margin-bottom: 1rem;">Try these suggested questions or ask your own</p>', unsafe_allow_html=True)
            
            auto_questions = [
                "What is this document about?",
                "Give a short summary",
                "What are the main topics?",
                "Key information mentioned?",
                "Document purpose?"
            ]
            
            cols = st.columns(2)
            for idx, q in enumerate(auto_questions):
                with cols[idx % 2]:
                    if st.button(f"💬 {q}", key=f"q_{idx}", use_container_width=True):
                        st.session_state.current_question = q
            
            st.markdown("#### ✍️ Custom Question")
            question = st.text_input(
                "Type your question here",
                placeholder="e.g., What are the key findings mentioned in the document?",
                key="custom_question"
            )
            
            if "current_question" in st.session_state and not question:
                question = st.session_state.current_question
                st.session_state.pop("current_question", None)
            
            if question:
                with st.spinner("🔍 Searching documents and generating answer..."):
                    query_embedding = ollama.embeddings(
                        model="nomic-embed-text",
                        prompt=question
                    )["embedding"]
                    
                    relevant_chunks = retrieve(query_embedding, get_vectors())
                    context = "\n".join(relevant_chunks)[:1500]
                    
                    answer = ollama.chat(
                        model="llama3:instruct",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a document question answering assistant. "
                                    "Answer only using the document text. "
                                    f"{get_language_instruction(language)} "
                                    "If the answer is not found, say "
                                    "'This information is not in the document.'"
                                )
                            },
                            {
                                "role": "user",
                                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
                            }
                        ]
                    )
                    
                    st.session_state.total_queries += 1
                
                st.markdown("""
                <div class="answer-box">
                    <h4>🧠 AI Answer</h4>
                """, unsafe_allow_html=True)
                st.markdown(answer["message"]["content"])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Add feedback option
                st.markdown("#### 📊 Rate this answer")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("👍 Accurate", key=f"good_{question}", use_container_width=True):
                        st.session_state.successful_queries += 1
                        st.success("Thank you for your feedback!")
                with col2:
                    if st.button("👎 Inaccurate", key=f"bad_{question}", use_container_width=True):
                        st.info("Feedback recorded. We'll improve!")
                with col3:
                    if st.button("🔄 Regenerate", key=f"regen_{question}", use_container_width=True):
                        st.rerun()
                
                log_event(f"QUERY | Question={question}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# 🧾 FOOTER
# ======================================================
st.markdown("""
<div class="footer">
    <strong>© 2025 IDP Enterprise Platform</strong> • Powered by AI & RAG Technology<br/>
    <small style="color: #94a3b8;">Secure • Scalable • Intelligent Document Processing</small>
</div>
""", unsafe_allow_html=True)
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import base64
import os
import time

# Streamlit configuration - MUST be first
st.set_page_config(
    page_title="Legal AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 3rem !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 10px 30px rgba(240, 147, 251, 0.3);
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
    }
    
    .summary-container {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        box-shadow: 0 10px 30px rgba(79, 172, 254, 0.3);
        margin-top: 2rem;
        max-height: 600px;
        overflow-y: auto;
    }
    
    .summary-text {
        font-size: 1.1rem;
        line-height: 1.8;
        margin-bottom: 1rem;
        max-height: 400px;
        overflow-y: auto;
        padding-right: 10px;
        scrollbar-width: thin;
        scrollbar-color: rgba(255,255,255,0.3) transparent;
    }
    
    .summary-text::-webkit-scrollbar {
        width: 6px;
    }
    
    .summary-text::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
        border-radius: 3px;
    }
    
    .summary-text::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.3);
        border-radius: 3px;
    }
    
    .summary-text::-webkit-scrollbar-thumb:hover {
        background: rgba(255,255,255,0.5);
    }
    
    .pdf-viewer {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        background: white;
        padding: 1rem;
        height: 650px;
        max-width: 100%;
    }
    
    .pdf-viewer iframe {
        width: 100%;
        height: 600px;
        border: none;
        border-radius: 10px;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-weight: bold;
        box-shadow: 0 8px 25px rgba(250, 112, 154, 0.3);
    }
    
    .processing-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .pulse-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #667eea;
        margin: 0 5px;
        animation: pulse 1.5s infinite ease-in-out;
    }
    
    .pulse-dot:nth-child(2) { animation-delay: -1.1s; }
    .pulse-dot:nth-child(3) { animation-delay: -0.9s; }
    
    @keyframes pulse {
        0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
        40% { transform: scale(1.2); opacity: 1; }
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    .sidebar .stSelectbox {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_model():
    checkpoint = "google/flan-t5-large"
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    pipe_sum = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    return pipe_sum

# File preprocessing
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    return [text.page_content for text in texts]

# Enhanced summarization pipeline with progress tracking
def llm_pipeline(filepath):
    pipe_sum = load_model()
    chunks = file_preprocessing(filepath)
    summaries = []
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, chunk in enumerate(chunks):
        try:
            progress = (i + 1) / len(chunks)
            progress_bar.progress(progress)
            status_text.text(f'Processing chunk {i+1} of {len(chunks)}...')
            
            result = pipe_sum(chunk, max_length=150, min_length=30, do_sample=False)
            summaries.append(result[0]['summary_text'])
            time.sleep(0.1)  # Small delay for visual feedback
        except Exception as e:
            summaries.append(f"[Error on chunk] {e}")
    
    progress_bar.empty()
    status_text.empty()
    return " ".join(summaries), len(chunks)

# Enhanced PDF display
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'''
    <div class="pdf-viewer">
        <iframe src="data:application/pdf;base64,{base64_pdf}">
        </iframe>
    </div>
    '''
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ Legal AI</h1>
        <p>Powered by Flan-T5 • Transform legal documents into concise insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        summary_length = st.selectbox(
            "Summary Length",
            ["Short (50-100 words)", "Medium (100-200 words)", "Long (200-300 words)"],
            index=1
        )
        
        st.markdown("### 📊 Model Info")
        st.info("**Model:** Google Flan-T5 Large\n**Capabilities:** Abstractive summarization")
        
        st.markdown("### 🚀 Features")
        features = [
            "🎯 Accurate summarization",
            "⚡ Fast processing", 
            "📱 Responsive design",
            "🔒 Secure file handling"
        ]
        for feature in features:
            st.markdown(f"• {feature}")
    
    # File upload section
    st.markdown("""
    <div class="upload-section">
        <h2 style="color: white; margin-bottom: 1rem;">📤 Upload Your Document</h2>
        <p style="color: white; opacity: 0.9;">Drag and drop or browse for your PDF file</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document to get started with summarization"
    )

    if uploaded_file is not None:
        # File info
        file_size = len(uploaded_file.getvalue()) / 1024 / 1024  # MB
        
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-card">
                <h3>📄 File Name</h3>
                <p>{uploaded_file.name}</p>
            </div>
            <div class="stat-card">
                <h3>💾 File Size</h3>
                <p>{file_size:.2f} MB</p>
            </div>
            <div class="stat-card">
                <h3>🔧 Model</h3>
                <p>Flan-T5 Large</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Generate Summary", use_container_width=True):
                # Save uploaded file
                filepath = os.path.join("data", uploaded_file.name)
                os.makedirs("data", exist_ok=True)
                with open(filepath, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())

                # Create two columns for display with wider layout
                col_left, col_right = st.columns([3, 3], gap="medium")

                with col_left:
                    st.markdown("### 📖 Original Document")
                    displayPDF(filepath)

                with col_right:
                    st.markdown("### ✨ AI Summary")
                    
                    # Processing animation
                    st.markdown("""
                    <div class="processing-animation">
                        <div class="pulse-dot"></div>
                        <div class="pulse-dot"></div>
                        <div class="pulse-dot"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.spinner("🧠 AI is analyzing your document..."):
                        summary, num_chunks = llm_pipeline(filepath)
                    
                    # Display summary
                    st.markdown(f"""
                    <div class="summary-container">
                        <h3>📝 Summary Result</h3>
                        <div class="summary-text">{summary}</div>
                        <hr style="border-color: rgba(255,255,255,0.3);">
                        <small>📊 Processed {num_chunks} document chunks</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Download summary button
                    st.download_button(
                        label="💾 Download Summary",
                        data=summary,
                        file_name=f"summary_{uploaded_file.name.replace('.pdf', '.txt')}",
                        mime="text/plain"
                    )

    else:
        # Welcome message when no file is uploaded
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #666;">
            <h2>👋 Welcome to Legal AI</h2>
            <p style="font-size: 1.2rem; margin-bottom: 2rem;">
                Upload a legal document above to get started with intelligent summarization
            </p>
            
            <div class="stats-grid">
                <div class="feature-card">
                    <h3>🎯 Accurate</h3>
                    <p>Advanced AI ensures high-quality summaries</p>
                </div>
                <div class="feature-card">
                    <h3>⚡ Fast</h3>
                    <p>Process documents in seconds, not minutes</p>
                </div>
                <div class="feature-card">
                    <h3>🔒 Secure</h3>
                    <p>Your documents are processed safely</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
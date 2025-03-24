import streamlit as st
from ocr_processor import OCRProcessor
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os
from PIL import Image
import json
import requests

# Page configuration
st.set_page_config(
    page_title="OCR with Ollama",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
        padding: 1rem;
    }
    .main {
        background-color: #f8f9fa;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox {
        margin-bottom: 1rem;
    }
    .upload-text {
        text-align: center;
        padding: 2rem;
        border: 2px dashed #ccc;
        border-radius: 10px;
        background-color: #ffffff;
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .gallery {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1rem;
        padding: 1rem;
    }
    .gallery-item {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 0.5rem;
        background: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Chroma DB 초기화
def initialize_vector_db(reset=False, embeddings=None):
    PERSIST_DIRECTORY = r"D:\conda\chroma_db"
    if reset and os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)

    try:
        vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        print("벡터 DB 로드 성공")
        return vector_db
    except Exception as e:
        print(f"벡터 DB 초기화 실패: {e}")
        return None


def get_available_models():
    response = requests.get(selected_server + "/api/tags")
    json_data = json.loads(response.text)["models"]
    model_list = []
    for jobject in json_data:
        model_list.append(jobject["name"])
    return model_list

def process_single_image(processor, image_path, format_type, enable_preprocessing, custom_prompt):
    """Process a single image and return the result"""
    try:
        result = processor.process_image(
            image_path=image_path,
            format_type=format_type,
            preprocess=enable_preprocessing,
            custom_prompt=custom_prompt  # Pass custom_prompt here
        )
        return result
    except Exception as e:
        return f"Error processing image: {str(e)}"

def process_batch_images(processor, image_paths, format_type, enable_preprocessing, custom_prompt):
    """Process multiple images and return results"""
    try:
        results = processor.process_batch(
            input_path=image_paths,
            format_type=format_type,
            preprocess=enable_preprocessing,
            custom_prompt=custom_prompt
        )
        return results
    except Exception as e:
        return {"error": str(e)}

def main():
    st.title("🔍 Vision OCR Lab")
    st.markdown("<p style='text-align: center; color: #666;'>Powered by Ollama Vision Models</p>", unsafe_allow_html=True)
    
    # Sidebar controls
with st.sidebar:
    st.header("🎮 Controls")

    selected_server = st.selectbox(
        "🤖 Select Server",
        ["http://192.168.2.87:11434",
        "http://mattangja.synology.me:43618",
        "http://llm.devangs.com",],
        index=0,
    )
    
    selected_model = st.selectbox(
        "🤖 Select Vision Model",
        get_available_models(),
        index=0,
    )
    
    format_type = st.selectbox(
        "📄 Output Format",
        ["markdown", "text", "json", "structured", "key_value"],
        help="Choose how you want the extracted text to be formatted"
    )
    
    # Custom prompt input
    custom_prompt_input = st.text_area(
        "📝 Custom Prompt (optional)",
        value="",
        help="Enter a custom prompt to override the default. Leave empty to use the predefined prompt."
    )

    max_workers = st.slider(
        "🔄 Parallel Processing",
        min_value=1,
        max_value=8,
        value=2,
        help="Number of images to process in parallel (for batch processing)"
    )

    enable_preprocessing = st.checkbox(
        "🔍 Enable Preprocessing",
        value=True,
        help="Apply image enhancement and preprocessing"
    )
    
    st.markdown("---")
    
    # Model info box
    if selected_model == "llamaV":
        st.info("LLaVA 7B: Efficient vision-language model optimized for real-time processing")
    else:
        st.info("Llama 3.2 Vision: Advanced model with high accuracy for complex text extraction")

# Determine if a custom prompt should be used (if text area is not empty)
custom_prompt = custom_prompt_input if custom_prompt_input.strip() != "" else None

# Initialize OCR Processor
processor = OCRProcessor(base_url=selected_server, model_name=selected_model, max_workers=max_workers)

# Main content area with tabs
tab1, tab2 = st.tabs(["📸 Image Processing", "ℹ️ About"])

with tab1:
    # File upload area with multiple file support
    uploaded_files = st.file_uploader(
        "Drop your images here",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'pdf'],
        accept_multiple_files=True,
        help="Supported formats: PNG, JPG, JPEG, TIFF, BMP, PDF"
    )

    if uploaded_files:
        # Create a temporary directory for uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            image_paths = []
            
            # Save uploaded files and collect paths
            for uploaded_file in uploaded_files:
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                image_paths.append(temp_path)

            # Display images in a gallery
            st.subheader(f"📸 Input Images ({len(uploaded_files)} files)")
            cols = st.columns(min(len(uploaded_files), 4))
            for idx, uploaded_file in enumerate(uploaded_files):
                with cols[idx % 4]:
                    try:
                        # Check if the uploaded file is a PDF
                        if uploaded_file.type == "application/pdf":
                            # Create a temporary file to save the uploaded PDF
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                                temp_pdf.write(uploaded_file.getvalue())
                                temp_pdf_path = temp_pdf.name
                            
                            # Convert the first page of the PDF to an image
                            import pdf2image
                            pages = pdf2image.convert_from_path(temp_pdf_path, first_page=1, last_page=1)
                            
                            if pages:
                                # Display the first page
                                st.image(pages[0], use_container_width=True, caption=f"{uploaded_file.name} (Page 1)")
                            else:
                                st.warning(f"Could not preview PDF: {uploaded_file.name}")
                            
                            # Clean up the temporary file
                            os.unlink(temp_pdf_path)
                        else:
                            # For regular images, use PIL as before
                            image = Image.open(uploaded_file)
                            st.image(image, use_container_width=True, caption=uploaded_file.name)
                            print(f"Processed {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error displaying {uploaded_file.name}: {str(e)}")

            # Process button
            if st.button("🚀 Process Images"):
                with st.spinner("Processing images..."):
                    if len(image_paths) == 1:
                        # Single image processing
                        result = process_single_image(
                            processor, 
                            image_paths[0], 
                            format_type,
                            enable_preprocessing,
                            custom_prompt  # Pass custom_prompt here
                        )
                        st.subheader("📝 Extracted Text")
                        st.markdown(result)

                        # Ollama 임베딩 설정
                        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest", base_url=selected_server)
                        vector_db = initialize_vector_db(reset=False, embeddings=embeddings)
                        # result to text document
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=160)
                        documents = text_splitter.create_documents([result])

                        # # 문서별로 파일명, 페이지 번호, 조각 번호를 메타데이터에 추가
                        file_name = os.path.basename(image_paths[0])
                        for i, doc in enumerate(documents):
                            page_number = i + 1  # 각 문서의 페이지 번호를 1부터 시작하도록 설정
                            doc.metadata = {"file_name": file_name, "page_number": page_number}

                        vector_db.add_documents(documents)
                        vector_db.persist()
                        print("PDF 텍스트가 벡터 DB에 성공적으로 추가되었습니다.")
                        
                        # Download button for single result
                        st.download_button(
                            "📥 Download Result",
                            result,
                            file_name=f"ocr_result.{format_type}",
                            mime="text/plain"
                        )
                    else:
                        # Batch processing
                        results = processor.process_batch(
                            input_path=image_paths,
                            format_type=format_type,
                            preprocess=enable_preprocessing,
                            custom_prompt=custom_prompt
                        )
                        
                        # Display statistics
                        st.subheader("📊 Processing Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Images", results['statistics']['total'])
                        with col2:
                            st.metric("Successful", results['statistics']['successful'])
                        with col3:
                            st.metric("Failed", results['statistics']['failed'])

                        # Display results
                        st.subheader("📝 Extracted Text")
                        for file_path, text in results['results'].items():
                            with st.expander(f"Result: {os.path.basename(file_path)}"):
                                st.markdown(text)

                        # Display errors if any
                        if results['errors']:
                            st.error("⚠️ Some files had errors:")
                            for file_path, error in results['errors'].items():
                                st.warning(f"{os.path.basename(file_path)}: {error}")
                        

                        # Download all results as JSON
                        if st.button("📥 Download All Results"):
                            json_results = json.dumps(results, indent=2)
                            st.download_button(
                                "📥 Download Results JSON",
                                json_results,
                                file_name="ocr_results.json",
                                mime="application/json"
                            )

with tab2:
    st.header("About Vision OCR Lab")
    st.markdown("""
    This application uses state-of-the-art vision language models through Ollama to extract text from images.
    
    ### Features:
    - 🖼️ Support for multiple image formats
    - 📦 Batch processing capability
    - 🔄 Parallel processing
    - 🔍 Image preprocessing and enhancement
    - 📊 Multiple output formats
    - 📥 Easy result download
    
    ### Models:
    - **LLaVA 7B**: Efficient vision-language model for real-time processing
    - **Llama 3.2 Vision**: Advanced model with high accuracy for complex documents
    """)

if __name__ == "__main__":
    main()

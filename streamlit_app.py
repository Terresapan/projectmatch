import streamlit as st
import os
import re

# Langchain and AI libraries
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# File processing libraries
import PyPDF2

# --- UI Enhancements ---
st.set_page_config(page_title="Project-Consultant Matcher", page_icon="🤝")

# Attempt to import python-docx, but provide fallback
try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    st.warning("⚠️ python-docx library not installed. Word document support will be limited.")

# Set API keys from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["general"]["GROQ_API_KEY"]

# File processing functions
def extract_text_from_pdf(file):
    """Extract text from a PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
        return ""

def extract_text_from_docx(file):
    """Extract text from a Word document"""
    if not HAS_DOCX:
        st.warning("⚠️ Cannot process Word documents. Please install python-docx.")
        return ""
    
    try:
        doc = docx.Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"❌ Error reading DOCX: {e}")
        return ""

def extract_text_from_txt(file):
    """Extract text from a text file"""
    try:
        return file.read().decode('utf-8')
    except Exception as e:
        st.error(f"❌ Error reading text file: {e}")
        return ""

# Initialize embeddings
@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")

# Load data from Google Sheets
@st.cache_data(ttl=600)
def load_consultant_data():
    try:
        from streamlit_gsheets import GSheetsConnection
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet="Database", ttl="10m", usecols=[0, 1, 2, 3, 4, 5], nrows=15)
        return df
    except Exception as e:
        st.error(f"❌ Error loading consultant data: {e}")
        return None

# Project summary function using AI
def generate_project_summary(text):
    """Generate structured project summary using AI"""
    # Remove excessive whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Truncate text if it's too long to prevent context overflow
    max_length = 10000
    if len(text) > max_length:
        text = text[:max_length]
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0.2,
        api_key=os.environ["GROQ_API_KEY"]
    )
    
    # Prompt for extracting project details
    prompt = f"""Extract and structure the following information from the project document:
    1. Project Name: Create one according to the context if not given
    2. Project Scope
    3. Client Expectations
    4. Skills Needed

    Project Document:
    {text}

    Provide the output in a clear, concise format. If any information is not clearly mentioned, use 'Not Specified' or make a reasonable inference based on the context."""

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        st.error(f"❌ Error generating project summary: {e}")
        return "Unable to generate summary."

# Create vector store for consultants
@st.cache_resource
def create_consultant_vector_store(_embeddings, df):
    if df is None or df.empty:
        st.error("❌ Consultant DataFrame is None or empty")
        return None
    
    try:
        # Ensure column names are correct
        df.columns = ['Name', 'Age', 'Education', 'Domain', 'Bio', 'Availability']
        
        # Create text representations of consultants
        text_data = [
            f"Name: {name}; Age: {age}; Education: {education}; Domain: {domain}; Bio: {bio}; Availability: {availability}"
            for name, age, education, domain, bio, availability in zip(
                df['Name'], df['Age'], df['Education'], df['Domain'], df['Bio'], df['Availability']
            )
        ]
        
        # Create metadata list
        metadatas = df.to_dict('records')
        
        # Create vector store
        vector_store = FAISS.from_texts(
            texts=text_data, 
            embedding=_embeddings,
            metadatas=metadatas
        )
        return vector_store
    except Exception as e:
        st.error(f"❌ Error creating consultant vector store: {e}")
        return None

# Analyze consultant match with AI
def analyze_consultant_match(project_summary, consultant_details):
    """Generate detailed analysis of consultant match"""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0.2,
        api_key=os.environ["GROQ_API_KEY"]
    )
    
    # Detailed prompt to assess consultant fit
    prompt = f"""Analyze the match between this project and the consultant:

Project Summary:
{project_summary}

Consultant Details:
{consultant_details}

Provide a detailed assessment that includes:
1. Strengths of this consultant for the project within 100 words
2. Potential limitations or challenges within 100 words
3. Overall suitability rating (out of 10)

Your analysis should be constructive, highlighting both positive aspects and areas of potential concern."""

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        st.error(f"❌ Error analyzing consultant match: {e}")
        return "Unable to generate detailed match analysis."

# Find best consultant matches
def find_best_consultant_matches(vector_store, project_summary, top_k=3):
    """Find the best consultant matches based on project summary"""
    if not vector_store:
        return []
    
    try:
        # Search in consultant vector store
        results = vector_store.similarity_search(project_summary, k=top_k)
        
        # Format results with match analysis
        matches = []
        for result in results:
            consultant_details = "\n".join([
                f"{key}: {value}" for key, value in result.metadata.items()
            ])
            
            # Generate match analysis
            match_analysis = analyze_consultant_match(project_summary, consultant_details)
            
            matches.append({
                "Name": result.metadata.get('Name', 'N/A'),
                "Age": result.metadata.get('Age', 'N/A'),
                "Education": result.metadata.get('Education', 'N/A'),
                "Domain": result.metadata.get('Domain', 'N/A'),
                "Bio": result.metadata.get('Bio', 'N/A'),
                "Availability": result.metadata.get('Availability', 'N/A'),
                "Match Analysis": match_analysis
            })
        
        return matches
    except Exception as e:
        st.error(f"❌ Error finding consultant matches: {e}")
        return []

# Main Streamlit app
def main():
    st.title("🤝 Project-Consultant Matcher")
    
    # Input method selection using tags
    input_method = st.radio("Choose Input Method", ["📂 File Upload", "✍️ Text Query"])
    
    # Initialize session state variables
    if 'project_summary' not in st.session_state:
        st.session_state.project_summary = None
    if 'matches' not in st.session_state:
        st.session_state.matches = None
    
    # File upload section
    if input_method == "📂 File Upload":
        uploaded_file = st.file_uploader("Upload Project Document", type=["pdf", "docx", "txt"])
        
        if uploaded_file is not None:
            # Process the uploaded file
            if uploaded_file.type == "application/pdf":
                file_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                if HAS_DOCX:
                    file_text = extract_text_from_docx(uploaded_file)
                else:
                    st.error("❌ Cannot process Word documents. Please install python-docx.")
                    return
            elif uploaded_file.type == "text/plain":
                file_text = extract_text_from_txt(uploaded_file)
            else:
                st.error("❌ Unsupported file type")
                return
            
            # Store file text for matching
            st.session_state.file_text = file_text
    
    # Text query section
    else:
        # Text input for project description
        st.session_state.file_text = st.text_area(
            "Enter Project Description", 
            height=200, 
            placeholder="Describe your project, required skills, and expectations..."
        )
    
    # Match button with improved styling
    match_button = st.button("✨ Find Best Consultants")
    
    # Matching process
    if match_button and st.session_state.file_text:
        # Display file processing progress
        with st.spinner('⚙️ Processing project document...'):
            # Generate project summary
            st.session_state.project_summary = generate_project_summary(st.session_state.file_text)
            st.subheader("📝 Project Summary")
            st.write(st.session_state.project_summary)
        
        # Get embeddings and consultant data
        embeddings = get_embeddings()
        consultant_df = load_consultant_data()
        
        if consultant_df is not None:
            # Create consultant vector store
            vector_store = create_consultant_vector_store(embeddings, consultant_df)
            
            # Find best matches
            if vector_store:
                with st.spinner('🔍 Finding best consultant matches...'):
                    st.session_state.matches = find_best_consultant_matches(vector_store, st.session_state.project_summary)
                
                st.subheader("🎯 Best Matching Consultants")
                if st.session_state.matches:
                    for i, consultant in enumerate(st.session_state.matches, 1):
                        st.markdown(f"### 👨‍💼 Consultant {i}")
                        for key, value in consultant.items():
                            # Special handling for match analysis to make it more readable
                            if key == "Match Analysis":
                                st.write(f"**{key}:**")
                                st.markdown(value)
                            else:
                                st.write(f"**{key}:** {value}")
                        st.write("---")
                else:
                    st.write("😔 No matching consultants found.")
            else:
                st.error("❌ Could not create consultant vector store")
        else:
            st.error("❌ Could not load consultant data")

if __name__ == "__main__":
    main()
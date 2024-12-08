import streamlit as st
from streamlit_gsheets import GSheetsConnection
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
import pandas as pd

# Set API keys
os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["general"]["GROQ_API_KEY"]

# Initialize embeddings
@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")

# Load data from Google Sheets
@st.cache_data(ttl=600)
def load_data():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet="Ideas", ttl="10m", usecols=[0, 1, 2], nrows=210)
        
        # Debug: Print DataFrame information
        # st.write("DataFrame Debug:")
        # st.write("DataFrame shape:", df.shape)
        # st.write("Columns:", df.columns)
        # st.write("First few rows:")
        # st.write(df.head())
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Create vector store
@st.cache_resource
def create_vector_store(_embeddings, df):
    if df is None or df.empty:
        st.error("DataFrame is None or empty")
        return None
    
    try:
        # Ensure column names are correct
        df.columns = ['Idea', 'Category', 'How']
        
        # Debug: Check data before creating embeddings
        # st.write("Data before embedding:")
        # st.write(df.head())
        
        text_data = [
            f"Idea: {idea}; Category: {category}; How: {how}"
            for idea, category, how in zip(df['Idea'], df['Category'], df['How'])
        ]
        
        # Debug: Check text data
        # st.write("Text Data:")
        # st.write(text_data[:5])  # Print first 5 entries
        
        # Create metadata list
        metadatas = df.to_dict('records')
        
        # Debug: Check metadata
        # st.write("Metadata (first few):")
        # st.write(metadatas[:5])
        
        vector_store = FAISS.from_texts(
            texts=text_data, 
            embedding=_embeddings,
            metadatas=metadatas
        )
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# Main Streamlit app
def main():
    st.title("Idea Explorer")
    
    # Get embeddings
    embeddings = get_embeddings()
    
    # Load data
    df = load_data()
    if df is None or df.empty:
        st.error("Could not load data or data is empty")
        return
    
    # Create vector store
    vector_store = create_vector_store(embeddings, df)
    if vector_store is None:
        st.error("Could not create vector store")
        return
    
    # User query input
    user_query = st.text_input("Ask a question about ideas:")
    
    if user_query:
        try:
            # Search in vector store
            results = vector_store.similarity_search(user_query, k=5)
                
            # Initialize LLM
            llm = ChatGroq(
                model="llama-3.3-70b-versatile", 
                temperature=0,
                api_key=st.secrets["general"]["GROQ_API_KEY"]
            )
            
            if results:
                result_texts = "\n".join([
                    f"Idea: {result.metadata.get('Idea', 'N/A')}, "
                    f"How: {result.metadata.get('How', 'N/A')}, "
                    f"Category: {result.metadata.get('Category', 'N/A')}" 
                    for result in results
                ])
                
                prompt = f"I found the following ideas based on your query '{user_query}':\n{result_texts}\nCan you summarize or provide insight on these ideas?"
                
                response = llm.invoke(prompt)
                st.write(response.content)
            else:
                st.write("No relevant results found.")

            # Debug: Print results
            # st.write("Search Results Debug:")
            for result in results:
                # st.write("Result Text:", result.page_content)
                st.write("Result Metadata:", result.metadata)
        
        except Exception as e:
            st.error(f"Error processing query: {e}")

if __name__ == "__main__":
    main()
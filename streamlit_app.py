import streamlit as st
from streamlit_gsheets import GSheetsConnection
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
import os

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["general"]["GROQ_API_KEY"]

# Step 1: Initialize the OpenAI embeddings and Faiss vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
index = faiss.IndexFlatL2()
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)


# Step 2: Connect to Google Sheets and read all relevant columns
@st.cache_data(ttl=600)  # Cache for 10 minutes (600 seconds)
def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    return conn.read(worksheet="ideas", ttl="10m", usecols=[0, 1, 2], nrows=50)

df = load_data()


# Step 3: Process the data
# Concatenate multiple columns for creating text data for embeddings
text_data = [
    f"Idea: {idea}; Category: {category}; How: {how}"
    for idea, category, how in zip(df['idea'], df['category'], df['how'])
]

# Generate embeddings for the text data
embeddings_list = [embeddings.embed(text) for text in text_data]


# Step 4: Store embeddings in the vector store
for index, vector in enumerate(embeddings_list):
    metadata = df.iloc[index].to_dict()  # Convert the entire row to a dictionary
    vector_store.add(vector, metadata)  # Storing the vector with all columns' metadata


# Step 5: User input for queries
user_query = st.text_input("Ask a question about ideas:")

if user_query:
    # Step 6: Vectorize user query
    query_vector = embeddings.embed(user_query)

    # Step 7: Search in vector store
    results = vector_store.similarity_search(query_vector)  # Assuming similarity_search is a method of the vector store

    # Step 8: Display results and generate LLM answer
    if results:
        # Create a formatted string from results to pass to LLM
        result_texts = "\n".join(
            [f"Idea: {result['idea']}, How: {result['how']}, Category: {result['category']}" for result in results]
        )
        prompt = f"I found the following ideas based on your query '{user_query}':\n{result_texts}\nCan you summarize or provide insight on these ideas?"

        # Call the OpenAI LLM
        llm_response = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)

        # Get the LLM's answer
        st.write("Answer:")
        st.write(llm_response.invoke(prompt))

    else:
        st.write("No relevant results found.")
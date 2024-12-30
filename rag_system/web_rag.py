import streamlit as st
import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_groq import ChatGroq
from langchain.docstore.document import Document 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Load the GROQ and OpenAI API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("RAG Webpage Assistant Using Mixtral")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="mixtral-8x7b-32768"
)

prompt = ChatPromptTemplate.from_template(
    """
    You are Webpage Assistant that helps users to find information in a context.
    Please provide the most accurate response based on the context and inputs
    only give information that is in the context not in general
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

def extract_links(homepage_url):
    try:
        # Fetch the webpage content
        response = requests.get(homepage_url)
        response.raise_for_status()  # Check if the request was successful

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all anchor tags (links) on the page
        links = soup.find_all('a', href=True)

        # Extract href attribute from each link and join with the homepage URL
        all_links = set()
        homepage_domain = urlparse(homepage_url).netloc  # Extract domain from homepage URL

        for link in links:
            href = link['href']

            # Filter out non-HTTP/HTTPS and non-relative links
            if href.startswith(('http://', 'https://')):
                # Check if the link belongs to the same domain
                link_domain = urlparse(href).netloc
                if link_domain == homepage_domain:
                    full_link = urljoin(homepage_url, href)  # Join with the base URL if needed
                    all_links.add(full_link)

            elif href.startswith('/'):  # Handle relative URLs
                full_link = urljoin(homepage_url, href)
                all_links.add(full_link)

        return all_links

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        return set()

def vector_embedding(webpage_link):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        all_links = extract_links(webpage_link)
        if webpage_link not in all_links:
            all_links.add(webpage_link)
        
        # Initialize the list of final_documents if it doesn't exist
        if 'final_documents' not in st.session_state:
            st.session_state.final_documents = []
        
        # Iterate over all links to load and process the documents
        for link in all_links:
            st.session_state.loader = WebBaseLoader(link)  # Load webpage
            st.session_state.docs = st.session_state.loader.load()  # Document loading
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk creation

            # Split the documents into chunks and append each as a Document
            for doc in st.session_state.text_splitter.split_documents(st.session_state.docs):  # Splitting
                # Ensure 'doc' is a string before passing it to Document
                doc_str = str(doc)  # Convert to string if necessary
                document = Document(page_content=doc_str)  # Wrap each chunk into a Document object
                st.session_state.final_documents.append(document)

        # Now that all documents are wrapped with 'page_content', create vectors
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings

# Input for webpage link
webpage_link = st.text_input("Enter a Webpage Link")

# Embedding button
if st.button("Load and Embed Webpage") and webpage_link:
    vector_embedding(webpage_link)
    st.write("Vector Store DB is ready")

# Input for the question
prompt1 = st.text_input("Enter Your Question From Webpage")

if prompt1:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write("Response time:", time.process_time() - start)
        st.write(response['answer'])

        # With a Streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.write("Please embed the webpage first.") 
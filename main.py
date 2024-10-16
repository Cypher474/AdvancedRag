import time
import os
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
pinekey = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone with API key from environment variable
pc = Pinecone(api_key=pinekey)

# Load the data from the Excel file
# Property Excel File
# df = pd.read_excel("ANTHONY AI/DATA/Extracted_property.xlsx")

# Anthony Excel File (uncomment to use this one)
df = pd.read_excel("ANTHONY AI/DATA/Extracted_Anthony.xlsx")

# Set index name
index_name = "langchain-test-contact"

# Check if the index already exists, otherwise create it
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    
    # Wait until the index is ready
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

# Connect to the index
index = pc.Index(index_name)

# Use Ollama embeddings
ollama_embed = OllamaEmbeddings(model="mxbai-embed-large")

# Create a PineconeVectorStore using Ollama embeddings
vector_store = PineconeVectorStore(index=index, embedding=ollama_embed)

# Property Excel file format columns
property_columns = [
    "Owner (Contact).Full name", "Address 1", "City", "State", "Zip code", 
    "Sq Ft", "Units", "Trans. date", "Trans. price", "Owner.Owner Link", 
    "Owner (Contact).First name", "Owner (Contact).Last name", 
    "Owner.Contact Name", "Owner.Company"
]

# Anthony Excel file format columns
anthony_columns = [
    "Full name", "First name", "Last name", "Title", "Company", "Address 1", 
    "City", "State", "Work", "Zip code", "Fax", "Latitude", "Mobile", "Home", 
    "Longitude", "Investor.Market", "Email", "Web page"
]

# Choose the columns to use based on the file type (uncomment as needed)
# For Property file
# required_columns = property_columns

# For Anthony file (current setting)
required_columns = anthony_columns

# Convert each row in the DataFrame into a separate Document
docs = []
for idx, row in df.iterrows():
    # Create a dictionary of the relevant columns
    chunk = {col: row[col] for col in required_columns if col in row}
    
    # Concatenate the key-value pairs into a single string for embeddings
    content = " ".join([f"{key}: {value}" for key, value in chunk.items()])
    
    # Create a langchain Document object
    doc = Document(page_content=content)
    docs.append(doc)

# Print the first loaded document to verify
print(f"Number of documents (rows): {len(docs)}")
print("First document (row) content:")
print(docs[0].page_content)

# Index the documents in Pinecone
ids = [str(i) for i in range(1, len(docs) + 1)]  # Generate IDs for the documents
vector_store.add_documents(documents=docs, ids=ids)

# Perform similarity search on the updated index
results = vector_store.similarity_search(query="Address of row with BRONX?", k=1)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")

# Perform similarity search with a filter (if needed)
results = vector_store.similarity_search(query="Address of row with BRONX?", k=1, filter={"key": "value"})
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")

# Perform similarity search with score
results = vector_store.similarity_search_with_score(query="Address of row with BRONX?", k=10)
for doc, score in results:
    print(f"* [SIM={score:.6f}] {doc.page_content} [{doc.metadata}]")

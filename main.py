from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.runnables import chain
from typing import List, Any, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv  
import os

# Load environment variables
load_dotenv()

# Loading API keys from .env
pinekey = os.getenv("PINECONE_KEY")
openaikey = os.getenv("OPENAI_API_KEY")

# Initializing Pinecone and OpenAI clients
pc = Pinecone(api_key=pinekey)
client = OpenAI(api_key=openaikey)

# Check existing indexes
index_name = "langchain-pdf-metadata"

# Connect to the Pinecone index
index = pc.Index(index_name)

# Set up embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openaikey)

# Connect to the existing Pinecone index with LangChain
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Function to display document with metadata
def display_document(doc, index):
    print(f"Doc {index}:")
    print(f"Content: {doc.page_content}")
    if 'score' in doc.metadata:
        print(f"Score: {doc.metadata['score']}")
    if 'source' in doc.metadata:
        print(f"Source: {doc.metadata['source']}")
    if 'summary' in doc.metadata:
        print(f"Summary: {doc.metadata['summary']}")
    print()  # Add a blank line for readability

# Function to write scores to file
def write_scores_to_file(retriever_name, scores):
    with open("scores.txt", "a") as f:
        f.write(f"{retriever_name} Scores:\n")
        for score in scores:
            f.write(f"{score}\n")
        f.write("\n")

# Updated function to perform basic retrieval with metadata
@chain
def basic_retrieval(query: str) -> Tuple[List[Document], List[float]]:
    docs, scores = zip(*vector_store.similarity_search_with_score(query))
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score
    return list(docs), list(scores)

# Updated SelfQueryRetriever class with metadata handling
class CustomSelfQueryRetriever(SelfQueryRetriever):
    def _get_docs_with_query(
        self, query: str, search_kwargs: Dict[str, Any]
    ) -> List[Document]:
        docs, scores = zip(*self.vectorstore.similarity_search_with_score(query, **search_kwargs))
        for doc, score in zip(docs, scores):
            doc.metadata["score"] = score
        return list(docs)

# Updated function to perform self-query retrieval with metadata
def self_query_retrieval(query):
    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The source file of the document",
            type="string",
        ),
        AttributeInfo(
            name="summary",
            description="A brief summary of the document content",
            type="string",
        ),
        AttributeInfo(
            name="rating", 
            description="A 1-10 rating for the document", 
            type="float"
        ),
    ]
    document_content_description = "Content about Arthas Menethil and related topics"
    llm = OpenAI(api_key=openaikey)
    self_query_retriever = CustomSelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=PineconeVectorStore(index=index, embedding=embeddings),
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info
    )
    retrieved_docs = self_query_retriever.invoke(query)
    scores = [doc.metadata.get("score", 0) for doc in retrieved_docs]
    print("Documents retrieved (Self-Query retriever):")
    for i, doc in enumerate(retrieved_docs, start=1):
        display_document(doc, i)
    return retrieved_docs, scores

# Updated function to perform hybrid search with metadata
def hybrid_search(query, keyword_filter="Arthas"):
    docs_hybrid, scores = zip(*vector_store.similarity_search_with_score(query))
    if keyword_filter:
        filtered_docs = [doc for doc, score in zip(docs_hybrid, scores) if keyword_filter.lower() in doc.metadata.get('summary', '').lower()]
        filtered_scores = [score for doc, score in zip(docs_hybrid, scores) if keyword_filter.lower() in doc.metadata.get('summary', '').lower()]
    else:
        filtered_docs = docs_hybrid
        filtered_scores = scores
    
    print("Documents retrieved (Hybrid Search):")
    for i, (doc, score) in enumerate(zip(filtered_docs, filtered_scores), start=1):
        doc.metadata['score'] = score
        display_document(doc, i)
    return filtered_docs, filtered_scores

# Updated function for MMR retrieval with scores
def mmr_retrieval(query):
    retriever_mmr = vector_store.as_retriever(search_type="mmr")
    docs_mmr = retriever_mmr.invoke(query)
    # Add scores using similarity_search_with_score
    _, scores = zip(*vector_store.similarity_search_with_score(query, k=len(docs_mmr)))
    for doc, score in zip(docs_mmr, scores):
        doc.metadata["score"] = score
    print("Documents retrieved (MMR retriever):")
    for i, doc in enumerate(docs_mmr, start=1):
        display_document(doc, i)
    return docs_mmr, scores

# Updated function for threshold retrieval with scores
def threshold_retrieval(query, threshold=0.5):
    retriever_with_threshold = vector_store.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={"score_threshold": threshold}
    )
    docs_with_threshold = retriever_with_threshold.invoke(query)
    # Add scores using similarity_search_with_score
    _, scores = zip(*vector_store.similarity_search_with_score(query, k=len(docs_with_threshold)))
    for doc, score in zip(docs_with_threshold, scores):
        doc.metadata["score"] = score
    print("Documents retrieved (Threshold-based retriever):")
    for i, doc in enumerate(docs_with_threshold, start=1):
        display_document(doc, i)
    return docs_with_threshold, scores

# Updated function for multi-query retrieval with scores
def multi_query_retrieval(query, num_queries=3):
    output_parser = CommaSeparatedListOutputParser()
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question", "num_queries"],
        template="""You are an AI language model assistant. Your task is to generate {num_queries} 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by commas.
        Original question: {question}"""
    )
    llm_chain = (QUERY_PROMPT | client | output_parser).with_config(
        {"output_parser": {"parse": lambda x: x}}
    )
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(),
        llm=client,
        include_original=True
    )
    retrieved_docs_multi = multi_query_retriever.invoke(query)
    # Add scores using similarity_search_with_score
    _, scores = zip(*vector_store.similarity_search_with_score(query, k=len(retrieved_docs_multi)))
    for doc, score in zip(retrieved_docs_multi, scores):
        doc.metadata["score"] = score
    print("Documents retrieved (Multi-Query retriever):")
    for i, doc in enumerate(retrieved_docs_multi, start=1):
        display_document(doc, i)
    return retrieved_docs_multi, scores

# Function to plot max scores
def plot_max_scores_line(max_scores):
    retrievers = list(max_scores.keys())
    scores = list(max_scores.values())
    
    # Create a line plot
    plt.figure(figsize=(10, 6))
    plt.plot(retrievers, scores, marker='o', linestyle='-', color='b', label="Max Score")
    
    # Highlight and label points
    for i, score in enumerate(scores):
        if score == 0.0000:
            plt.annotate(f"{score:.4f}", (i, score), textcoords="offset points", xytext=(0,10), ha='center', color='red')
            plt.scatter(i, score, color='blue ', zorder=5)  # Use a red marker for 0.0000 scores
        else:
            plt.annotate(f"{score:.4f}", (i, score), textcoords="offset points", xytext=(0,10), ha='center')

    # Add title and labels
    plt.title("Max Scores by Retriever (Line Plot)")
    plt.xlabel("Retriever")
    plt.ylabel("Max Score")
    plt.ylim(0, 1)  # Assuming scores are between 0 and 1
    plt.xticks(rotation=45, ha='right')
    
    # Save the plot as an image file
    plt.tight_layout()
    plt.savefig("max_scores_line_plot.png")
    plt.close()
# Main execution
if __name__ == "__main__":
    query = "What was the blade of Arthas Menethil?"
    max_scores = {}

    # Clear the scores.txt file
    open("scores.txt", "w").close()

    print("\n--- Basic Retrieval ---")
    docs, scores = basic_retrieval.invoke(query)
    write_scores_to_file("Basic Retrieval", scores)
    max_scores["Basic"] = max(scores)

    print("\n--- Self-Query Retrieval ---")
    _, scores = self_query_retrieval(query)
    write_scores_to_file("Self-Query Retrieval", scores)
    max_scores["Self-Query"] = max(scores)

    print("\n--- Hybrid Search (Vector + Keyword Filter) ---")
    _, scores = hybrid_search(query, keyword_filter="Arthas")
    write_scores_to_file("Hybrid Search", scores)
    max_scores["Hybrid"] = max(scores)

    print("\n--- MMR Retrieval ---")
    _, scores = mmr_retrieval(query)
    write_scores_to_file("MMR Retrieval", scores)
    max_scores["MMR"] = max(scores)

    print("\n--- Threshold Retrieval ---")
    _, scores = threshold_retrieval(query, threshold=0.5)
    write_scores_to_file("Threshold Retrieval", scores)
    max_scores["Threshold"] = max(scores)

    print("\n--- Multi-Query Retrieval ---")
    _, scores = multi_query_retrieval(query, num_queries=3)
    write_scores_to_file("Multi-Query Retrieval", scores)
    max_scores["Multi-Query"] = max(scores)

    # Plot max scores as a line graph
    plot_max_scores_line(max_scores)

    print("\nScores have been written to scores.txt")
    print("Max scores line plot has been saved as max_scores_line_plot.png")

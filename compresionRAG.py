import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import AIMessage
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import OpenAI
from dotenv import load_dotenv

class EnvironmentManager:
    @staticmethod
    def load_environment():
        load_dotenv()
        return os.getenv("PINECONE_API_KEY")

class PineconeManager:
    def __init__(self, api_key):
        self.pc = Pinecone(api_key=api_key)

    def get_index(self, index_name):
        return self.pc.Index(index_name)

class VectorStoreManager:
    def __init__(self, index_name, api_key):
        pinecone_manager = PineconeManager(api_key)
        index = pinecone_manager.get_index(index_name)
        # Embedding model setup here
        ollama_embed = OllamaEmbeddings(model="mxbai-embed-large")
        self.vector_store = PineconeVectorStore(index=index, embedding=ollama_embed)

    def get_retriever(self):
        """Returns a retriever object."""
        return self.vector_store.as_retriever()

class CompressionManager:
    def __init__(self, base_retriever):
        # Setting up the OpenAI LLM for compression
        llm = OpenAI(temperature=0)
        compressor = LLMChainExtractor.from_llm(llm)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )

    def compress_and_retrieve(self, query):
        return self.compression_retriever.invoke(query)

class ChatbotManager:
    def __init__(self, model_name="llama3.1"):
        self.chatbot = ChatOllama(model=model_name, temperature=0)
    
    def chat(self, system_message, human_message):
        messages = [
            ("system", system_message),
            ("human", human_message)
        ]
        ai_msg = self.chatbot.invoke(messages)
        return ai_msg.content

def main():
    # Initialize environment
    api_key = EnvironmentManager.load_environment()

    # Set up VectorStore for two indexes
    index_name_contact = "langchain-test-contact"
    index_name_property = "langchain-test-property"
    
    vector_store_manager_contact = VectorStoreManager(index_name_contact, api_key)
    vector_store_manager_property = VectorStoreManager(index_name_property, api_key)

    user_query = "give me properties having units greater than 100"
    
    # Perform similarity search on both indexes
    print(f"\nPerforming similarity search for: '{user_query}' on 'langchain-test-contact'")
    results_contact = vector_store_manager_contact.get_retriever().get_relevant_documents(user_query)

    print(f"\nPerforming similarity search for: '{user_query}' on 'langchain-test-property'")
    results_property = vector_store_manager_property.get_retriever().get_relevant_documents(user_query)

    # Combine results
    combined_results = results_contact + results_property

    # Store the response in a variable and prepare the content for the chatbot
    response = []
    for doc in combined_results:
        response.append(f"* {doc.page_content} [{doc.metadata}]")

    print("\nCombined Search Results:")
    for res in response:
        print(res)

    # Compression stage before passing to ChatOllama
    base_retriever = vector_store_manager_contact.get_retriever()  # Using contact index for compression example
    compression_manager = CompressionManager(base_retriever)
    compressed_docs = compression_manager.compress_and_retrieve(user_query)
    
    print("\nCompressed Documents:")
    for doc in compressed_docs:
        print(doc.page_content)

    # Prepare messages for ChatOllama
    system_message = "You are a helpful assistant providing property and contact details. Provide a helpful response based on the search results. Make Tables if you feel like it"
    human_message = f"User query: '{user_query}'\nCompressed Results:\n" + "\n".join([doc.page_content for doc in compressed_docs])

    # Send the messages to the chatbot and get the response
    chatbot_manager = ChatbotManager()
    chatbot_response = chatbot_manager.chat(system_message, human_message)

    # Print the chatbot's response
    print("\nChatbot Response:")
    print(chatbot_response)

if __name__ == "__main__":
    main()

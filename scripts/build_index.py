"""
Script to build FAISS index from Markdown files in data/processed folder
"""

import os
import json
import pickle
import sys
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import dotenv

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

class IndexBuilder:
    """Class to build FAISS index from Markdown files"""
    def __init__(self, data_folder: str = "data/processed", vector_store_path: str = "indexes"):
        self.data_folder = data_folder
        self.vector_store_path = vector_store_path
        # Use Azure OpenAI embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-3-small",
            api_version=dotenv.get_key(".env", "AZURE_OPENAI_API_VERSION")
        )
        # Use RecursiveCharacterTextSplitter to split text into chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=300, # 10% overlap
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_documents(self) -> tuple[List[str], List[Dict[str, Any]]]:
        """Load and process all Markdown documents
        
        Returns:
            tuple[List[str], List[Dict[str, Any]]]: Tuple containing the list of documents and metadata
        """
        # Initialize lists for documents and metadata
        documents = []
        metadata_list = []
        
        # Check if data folder exists
        if not os.path.exists(self.data_folder):
            print(f"Data folder {self.data_folder} not found.")
            return documents, metadata_list
        
        # Get all Markdown files in data folder
        markdown_files = list(Path(self.data_folder).glob("*.md"))
        
        # Check if there are any Markdown files
        if not markdown_files:
            print(f"No Markdown files found in {self.data_folder}")
            return documents, metadata_list
        
        print(f"Found {len(markdown_files)} Markdown files")
        
        # Process each Markdown file
        for markdown_file in markdown_files:
            print(f"Processing {markdown_file}")
            # Read the Markdown file
            with open(markdown_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Check if text content is not empty
            if text.strip():
                # Split text into chunks using RecursiveCharacterTextSplitter
                chunks = self.text_splitter.split_text(text)
                
                # Add each chunk to the documents list and metadata list
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        documents.append(chunk)
                        metadata_list.append({
                            "chunk_id": i + 1,
                            "total_chunks": len(chunks),
                            "source_file": markdown_file.name,
                            "title": text.split("\n")[0]
                        })
            else:
                print(f"No text extracted from {markdown_file}")
        
        print(f"Total document chunks: {len(documents)}")
        return documents, metadata_list
    
    def build_index(self):
        """Build FAISS index from documents"""
        print("Loading documents...")
        documents, metadata_list = self.load_documents()
        
        if not documents:
            print("No documents to index.")
            return
        
        print("Generating embeddings...")
        try:
            # Generate embeddings for each document
            embeddings_list = self.embeddings.embed_documents(documents)
            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings_list, dtype=np.float32)
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return
        
        print(f"Embeddings shape: {embeddings_array.shape}")
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        # Create FAISS index with Inner Product (cosine similarity)
        index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Add embeddings to index
        index.add(embeddings_array)
        
        print(f"Created FAISS index with {index.ntotal} documents")
        
        # Create vector store directory
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Create paths for index, documents, and metadata
        index_path = os.path.join(self.vector_store_path, "faiss_index.bin")
        docs_path = os.path.join(self.vector_store_path, "documents.pkl")
        metadata_path = os.path.join(self.vector_store_path, "metadata.pkl")
        
        print("Saving index and metadata...")
        
        # Save index
        faiss.write_index(index, index_path)
        
        # Save documents as pickle file
        with open(docs_path, 'wb') as f:
            pickle.dump(documents, f)
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata_list, f)
        
        # Save chunks as json file for debugging
        with open('data/chunked/documents.json', 'w', encoding='utf-8') as f:
            json.dump([{
                "source_file": metadata_list[i]["source_file"],
                "title": metadata_list[i]["title"],
                "content": chunk,
                "chunk_id": metadata_list[i]["chunk_id"],
                "total_chunks": metadata_list[i]["total_chunks"],
                "length_in_characters": len(chunk),
                "length_in_tokens": int(len(chunk.split(' ')) * 0.75)
            } for i, chunk in enumerate(documents)], f, ensure_ascii=False, indent=4)
        
        print(f"Index saved to {self.vector_store_path}")
        
        # Print statistics
        print("\nIndex Statistics:")
        print(f"Total documents: {len(documents)}")
        print(f"Index dimension: {dimension}")
        print(f"Index type: {type(index).__name__}")

def main():
    """Main function"""
    print("Building FAISS index from Markdown files...")
    
    builder = IndexBuilder()
    builder.build_index()
    
    print("Done!")

if __name__ == "__main__":
    main()
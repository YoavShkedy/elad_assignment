import os
import sys
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any

# Add parent directory to path for imports when running directly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain_openai import AzureOpenAIEmbeddings
from models.schemas import RetrievalResult
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VectorService:
    """Service to handle vector store operations"""
    def __init__(self, vector_store_path: str = "indexes"):
        self.vector_store_path = vector_store_path
        # Use Azure OpenAI embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-3-small",
            api_version="2024-12-01-preview"
        )
        # Load FAISS index and associated data
        self.index = None
        self.documents = None
        self.metadata = None
        self.load_index()
    
    def load_index(self):
        """Load FAISS index and associated data"""
        try:
            # Paths for index, documents, and metadata
            index_path = os.path.join(self.vector_store_path, "faiss_index.bin")
            docs_path = os.path.join(self.vector_store_path, "documents.pkl")
            metadata_path = os.path.join(self.vector_store_path, "metadata.pkl")
            
            # Check if all files exist
            if all(os.path.exists(p) for p in [index_path, docs_path, metadata_path]):
                # Load FAISS index
                self.index = faiss.read_index(index_path)

                # Load documents
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                    
                print(f"Loaded FAISS index with {self.index.ntotal} documents")
            else:
                print("Vector store not found. Please run build_index.py first.")
                self.index = None
                self.documents = []
                self.metadata = []
        except Exception as e:
            print(f"Error loading vector store: {e}")
            self.index = None
            self.documents = []
            self.metadata = []
    
    def search(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Perform similarity search on the FAISS index
        
        Args:
            query: The query to search with
            k: The number of results to return
        
        Returns:
            List[RetrievalResult]: List of retrieval results
        """
        # Check if index is loaded
        if self.index is None:
            return []
        
        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            # Convert query embedding to numpy array
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Search in FAISS
            scores, indexes = self.index.search(query_vector, min(k, self.index.ntotal))
            
            results = []
            # Iterate over retrieved scores and indexes
            for score, idx in zip(scores[0], indexes[0]):
                
                # Get document and metadata
                doc = self.documents[idx] 
                metadata = self.metadata[idx] 
                
                # Add retrieval result
                results.append(RetrievalResult(
                    content=doc,
                    metadata=metadata,
                    score=float(score) # Cosine similarity score
                ))
                
                if len(results) >= k:
                    break
            
            return results
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics
        
        Returns:
            Dict[str, Any]: Dictionary containing vector store statistics
        """
        # Check if index is loaded
        if self.index is None:
            return {"status": "not_loaded", "total_documents": 0}
        
        return {
            "status": "loaded",
            "total_documents": self.index.ntotal,
            "dimension": self.index.d,
            "index_type": type(self.index).__name__
        }
    
    def display_vector_store(self, show_documents: bool = True, max_docs_to_show: int = 10, max_content_length: int = 200) -> None:
        """Display comprehensive information about the vector store and metadata
        
        Args:
            show_documents: Whether to display document contents
            max_docs_to_show: Maximum number of documents to display
            max_content_length: Maximum length of document content to show
        """
        print("=" * 80)
        print("VECTOR STORE INFORMATION")
        print("=" * 80)
        
        # Check if index is loaded
        if self.index is None:
            print("❌ Vector store is not loaded.")
            print("Please run build_index.py first to create the vector store.")
            return
        
        # Display basic statistics
        stats = self.get_stats()
        print(f"✅ Status: {stats['status']}")
        print(f"📊 Total Documents: {stats['total_documents']}")
        print(f"📐 Vector Dimension: {stats['dimension']}")
        print(f"🔧 Index Type: {stats['index_type']}")
        print()
        
        if not self.documents or not self.metadata:
            print("❌ No documents or metadata found.")
            return
        
        # Display metadata summary
        print("📋 METADATA SUMMARY")
        print("-" * 40)
        
        # Collect metadata keys and their frequency
        metadata_keys = {}
        sources = set()
        
        for meta in self.metadata:
            if isinstance(meta, dict):
                for key in meta.keys():
                    metadata_keys[key] = metadata_keys.get(key, 0) + 1
                # Track sources if available
                if 'source' in meta:
                    sources.add(meta['source'])
        
        print(f"Metadata fields found:")
        for key, count in metadata_keys.items():
            print(f"  • {key}: appears in {count}/{len(self.metadata)} documents")
        
        if sources:
            print(f"\nData sources ({len(sources)} unique):")
            for source in sorted(sources):
                print(f"  • {source}")
        
        print()
        
        # Display sample documents if requested
        if show_documents:
            print("📄 SAMPLE DOCUMENTS")
            print("-" * 40)
            
            num_to_show = min(max_docs_to_show, len(self.documents))
            
            for i in range(num_to_show):
                print(f"\n[Document {i+1}/{len(self.documents)}]")
                
                # Display metadata
                if i < len(self.metadata) and self.metadata[i]:
                    meta = self.metadata[i]
                    if isinstance(meta, dict):
                        for key, value in meta.items():
                            # Truncate long values
                            if isinstance(value, str) and len(value) > 100:
                                value = value[:97] + "..."
                            print(f"  {key}: {value}")
                
                # Display document content
                if i < len(self.documents):
                    content = self.documents[i]
                    if len(content) > max_content_length:
                        content = content[:max_content_length] + "..."
                    print(f"  Content: {content}")
                
                if i < num_to_show - 1:  # Don't print separator after last document
                    print("-" * 20)
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    vector_service = VectorService()
    vector_service.display_vector_store()
"""
Embeddings and vector search system using FAISS
"""
import os
import re
import pickle
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging

import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Document, DocumentChunk, RetrievedChunk
from .config import config

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Custom exception for embedding-related errors"""
    pass


class DocumentChunker:
    """Handles document chunking for optimal embedding"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap
    
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Split document into chunks for embedding"""
        content = document.content
        chunks = []
        
        # Smart chunking: preserve paragraphs and sentences
        paragraphs = self._split_into_paragraphs(content)
        
        current_chunk = ""
        chunk_start = 0
        chunk_id = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, create a chunk
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                chunk = self._create_chunk(
                    document.id, chunk_id, current_chunk, 
                    chunk_start, chunk_start + len(current_chunk)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + paragraph
                chunk_start = chunk_start + len(current_chunk) - len(overlap_text)
                chunk_id += 1
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunk = self._create_chunk(
                document.id, chunk_id, current_chunk,
                chunk_start, chunk_start + len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_paragraphs(self, content: str) -> List[str]:
        """Split content into meaningful paragraphs"""
        # Split by double newlines first
        paragraphs = re.split(r'\n\s*\n', content)
        
        # Further split long paragraphs by sentences
        final_paragraphs = []
        for paragraph in paragraphs:
            if len(paragraph) > self.chunk_size:
                sentences = self._split_into_sentences(paragraph)
                final_paragraphs.extend(sentences)
            else:
                final_paragraphs.append(paragraph.strip())
        
        return [p for p in final_paragraphs if p.strip()]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved with spaCy or NLTK)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Group sentences to form reasonable chunks
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk"""
        if len(text) <= self.chunk_overlap:
            return text
        
        # Try to find a good breaking point (sentence or paragraph)
        overlap_text = text[-self.chunk_overlap:]
        
        # Find the last sentence boundary
        sentence_match = re.search(r'[.!?]\s+', overlap_text)
        if sentence_match:
            return overlap_text[sentence_match.end():]
        
        return overlap_text
    
    def _create_chunk(self, doc_id: str, chunk_id: int, content: str, 
                     start_idx: int, end_idx: int) -> DocumentChunk:
        """Create a DocumentChunk object"""
        chunk_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        
        return DocumentChunk(
            id=f"{doc_id}_chunk_{chunk_id}_{chunk_hash}",
            document_id=doc_id,
            content=content.strip(),
            start_index=start_idx,
            end_index=end_idx,
            metadata={
                'chunk_index': chunk_id,
                'word_count': len(content.split()),
                'char_count': len(content)
            }
        )


class EmbeddingModel:
    """Wrapper for sentence transformer models"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.embedding_model
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise EmbeddingError(f"Failed to load embedding model: {str(e)}")
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Encode texts into embeddings"""
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            # Normalize texts
            normalized_texts = [self._normalize_text(text) for text in texts]
            
            # Generate embeddings
            embeddings = self.model.encode(
                normalized_texts,
                batch_size=batch_size,
                show_progress_bar=len(normalized_texts) > 50,
                convert_to_numpy=True
            )
            
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise EmbeddingError(f"Failed to encode texts: {str(e)}")
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better embedding quality"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might not add value
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        return text.strip()
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()


class FAISSVectorStore:
    """FAISS-based vector store for similarity search"""
    
    def __init__(self, embedding_model: EmbeddingModel, index_path: str = None):
        self.embedding_model = embedding_model
        self.index_path = index_path or os.path.join(config.index_dir, "faiss_index")
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.chunk_map: Dict[int, DocumentChunk] = {}
        
        # Create index directory
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Initialize FAISS index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        try:
            dimension = self.embedding_model.dimension
            
            if config.faiss_index_type == "IndexFlatIP":
                # Inner product (cosine similarity for normalized vectors)
                self.index = faiss.IndexFlatIP(dimension)
            elif config.faiss_index_type == "IndexFlatL2":
                # L2 distance
                self.index = faiss.IndexFlatL2(dimension)
            elif config.faiss_index_type == "IndexIVFFlat":
                # IVF for faster search with large datasets
                quantizer = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            else:
                # Default to flat inner product
                self.index = faiss.IndexFlatIP(dimension)
            
            logger.info(f"Initialized FAISS index: {config.faiss_index_type}")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise EmbeddingError(f"Failed to initialize FAISS index: {str(e)}")
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the vector store"""
        if not chunks:
            return
        
        try:
            # Extract text content for embedding
            texts = [chunk.content for chunk in chunks]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} chunks")
            embeddings = self.embedding_model.encode(texts)
            
            # Normalize for cosine similarity
            if config.faiss_index_type == "IndexFlatIP":
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i].tolist()
            
            # Add to FAISS index
            self.index.add(embeddings.astype(np.float32))
            
            # Update chunk storage
            start_idx = len(self.chunks)
            self.chunks.extend(chunks)
            
            # Update chunk mapping
            for i, chunk in enumerate(chunks):
                self.chunk_map[start_idx + i] = chunk
            
            logger.info(f"Added {len(chunks)} chunks to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add chunks to vector store: {e}")
            raise EmbeddingError(f"Failed to add chunks to vector store: {str(e)}")
    
    def search(self, query: str, top_k: int = None, 
               threshold: float = None) -> List[RetrievedChunk]:
        """Search for similar chunks"""
        top_k = top_k or config.max_retrieval_docs
        threshold = threshold or config.similarity_threshold
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Normalize for cosine similarity
            if config.faiss_index_type == "IndexFlatIP":
                query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
            
            # Convert results to RetrievedChunk objects
            retrieved_chunks = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                similarity_score = float(score)
                
                # Apply threshold
                if similarity_score < threshold:
                    continue
                
                chunk = self.chunk_map.get(idx)
                if chunk:
                    retrieved_chunk = RetrievedChunk(
                        chunk=chunk,
                        similarity_score=similarity_score,
                        relevance_explanation=self._generate_relevance_explanation(
                            query, chunk.content, similarity_score
                        )
                    )
                    retrieved_chunks.append(retrieved_chunk)
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query")
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Failed to search vector store: {e}")
            raise EmbeddingError(f"Failed to search vector store: {str(e)}")
    
    def _generate_relevance_explanation(self, query: str, content: str, 
                                      score: float) -> str:
        """Generate explanation for why a chunk is relevant"""
        # Simple keyword-based explanation
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        common_words = query_words.intersection(content_words)
        
        if common_words:
            return f"Contains key terms: {', '.join(list(common_words)[:3])} (similarity: {score:.3f})"
        else:
            return f"Semantic similarity: {score:.3f}"
    
    def save_index(self, path: str = None) -> None:
        """Save FAISS index and metadata to disk"""
        save_path = path or self.index_path
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{save_path}.index")
            
            # Save metadata (chunks and mapping)
            metadata = {
                'chunks': self.chunks,
                'chunk_map': self.chunk_map,
                'model_name': self.embedding_model.model_name
            }
            
            with open(f"{save_path}.metadata", 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved vector store to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise EmbeddingError(f"Failed to save vector store: {str(e)}")
    
    def load_index(self, path: str = None) -> None:
        """Load FAISS index and metadata from disk"""
        load_path = path or self.index_path
        
        try:
            # Load FAISS index
            if os.path.exists(f"{load_path}.index"):
                self.index = faiss.read_index(f"{load_path}.index")
                
                # Load metadata
                if os.path.exists(f"{load_path}.metadata"):
                    with open(f"{load_path}.metadata", 'rb') as f:
                        metadata = pickle.load(f)
                    
                    self.chunks = metadata.get('chunks', [])
                    self.chunk_map = metadata.get('chunk_map', {})
                    
                    logger.info(f"Loaded vector store from {load_path}")
                    logger.info(f"Index contains {len(self.chunks)} chunks")
                else:
                    logger.warning("Index file found but metadata missing")
            else:
                logger.info("No existing index found, starting fresh")
                
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            # Don't raise error, just start fresh
            self._initialize_index()
    
    @property
    def size(self) -> int:
        """Get number of chunks in the vector store"""
        return len(self.chunks)
    
    def clear(self) -> None:
        """Clear the vector store"""
        self._initialize_index()
        self.chunks.clear()
        self.chunk_map.clear()
        logger.info("Cleared vector store")


class EmbeddingManager:
    """High-level manager for embeddings and vector search"""
    
    def __init__(self, model_name: str = None, index_path: str = None):
        self.embedding_model = EmbeddingModel(model_name)
        self.vector_store = FAISSVectorStore(self.embedding_model, index_path)
        self.chunker = DocumentChunker()
        
        # Try to load existing index
        self.vector_store.load_index()
    
    def add_document(self, document: Document) -> List[DocumentChunk]:
        """Add a document to the vector store"""
        try:
            # Chunk the document
            chunks = self.chunker.chunk_document(document)
            
            # Add chunks to vector store
            self.vector_store.add_chunks(chunks)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to add document {document.id}: {e}")
            raise EmbeddingError(f"Failed to add document: {str(e)}")
    
    def add_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """Add multiple documents to the vector store"""
        all_chunks = []
        
        for document in documents:
            try:
                chunks = self.add_document(document)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to add document {document.id}: {e}")
                continue
        
        return all_chunks
    
    def search(self, query: str, top_k: int = None, 
               threshold: float = None, filters: Dict[str, Any] = None) -> List[RetrievedChunk]:
        """Search for relevant chunks"""
        retrieved_chunks = self.vector_store.search(query, top_k, threshold)
        
        # Apply filters if provided
        if filters:
            retrieved_chunks = self._apply_filters(retrieved_chunks, filters)
        
        return retrieved_chunks
    
    def _apply_filters(self, chunks: List[RetrievedChunk], 
                      filters: Dict[str, Any]) -> List[RetrievedChunk]:
        """Apply filters to retrieved chunks"""
        filtered_chunks = []
        
        for chunk in chunks:
            # Check domain filter
            if 'domain' in filters:
                # Would need to get document domain from chunk metadata
                pass
            
            # Check document type filter
            if 'doc_type' in filters:
                # Would need to get document type from chunk metadata
                pass
            
            # Add other filters as needed
            filtered_chunks.append(chunk)
        
        return filtered_chunks
    
    def save(self, path: str = None) -> None:
        """Save the vector store"""
        self.vector_store.save_index(path)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            'total_chunks': self.vector_store.size,
            'model_name': self.embedding_model.model_name,
            'embedding_dimension': self.embedding_model.dimension,
            'index_type': config.faiss_index_type
        }
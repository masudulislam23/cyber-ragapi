import os
import uuid
import json
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document as LangchainDocument
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import logging
from model_loader import PerceiverIOEmbedder
from rag.config import (
    DEFAULT_K,
    KEYWORD_BOOST_FACTOR,
    MIN_RELEVANCE_FALLBACK,
    RELEVANCE_DECAY_FACTOR,
    LOG_RETRIEVAL_DETAILS,
    LOG_CHUNK_CONTENT_PREVIEW
)

# Configure logging
logger = logging.getLogger(__name__)

class PerceiverEmbeddingWrapper:
    def __init__(self, model_path='./model'):
        self.embedder = PerceiverIOEmbedder(model_path)
    def embed_query(self, text):
        return self.embedder.get_embeddings([text])[0]
    def embed_documents(self, texts):
        return self.embedder.get_embeddings(texts)

embedding_model = PerceiverEmbeddingWrapper('./model')

# Initialize Chroma DB
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)

# Initialize vector store with PerceiverEmbeddingWrapper
vector_store = Chroma(
    persist_directory=CHROMA_PERSIST_DIRECTORY,
    embedding_function=embedding_model,
)

async def add_chunks_to_vectorstore(chunks: List[LangchainDocument], document_id: str) -> List[str]:
    """
    Add document chunks to the vector store.
    
    Args:
        chunks: List of document chunks
        document_id: ID of the document
        
    Returns:
        List of chunk IDs
    """
    # Add document chunks to vector store
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    
    # Add to vectorstore
    vector_store.add_documents(
        documents=chunks,
        ids=ids
    )
    
    # Persist to disk
    # vector_store.persist()
    
    return ids

async def retrieve_relevant_chunks(
    query: str,
    k: int = DEFAULT_K,
    document_ids: Optional[List[str]] = None,
    repository_ids: Optional[List[str]] = None,
    user_id: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks from the vector store.
    
    Args:
        query: Query string
        k: Number of chunks to retrieve (reduced default)
        document_ids: Optional list of document IDs to filter by
        repository_ids: Optional list of repository IDs to filter by
        user_id: Optional user ID to filter by (only return chunks from documents owned by this user)
        metadata_filter: Optional metadata filter
        
    Returns:
        List of relevant chunks with metadata
    """
    # SAFEGUARD: Explicitly exclude global repository from retrieval
    if repository_ids and "global_repository" in repository_ids:
        repository_ids.remove("global_repository")
        logger.warning("Global repository was passed to retrieve_relevant_chunks - REMOVED for security")
    
    # Log the total number of documents in the vector store
    collection_count = vector_store._collection.count()
    logger.info(f"Total documents in vector store: {collection_count}")
    
    if collection_count == 0:
        logger.warning("Vector store is empty! No documents have been indexed.")
        return []

    # Build filter
    filter_dict = {}
    
    if document_ids:
        filter_dict["document_id"] = {"$in": document_ids}
    
    # Log repository_ids before building filter_conditions
    logger.info(f"repository_ids before filter_conditions: {repository_ids}")
    search_kwargs = {"k": k}
    filter_conditions = []

    # Handle document IDs
    if document_ids:
        filter_conditions.append({"document_id": {"$in": document_ids}})

    # Handle repository IDs
    if repository_ids:
        logger.info(f"Adding repository_id filter: {repository_ids}")
        if len(repository_ids) > 1:
            filter_conditions.append({"repository_id": {"$in": repository_ids}})
        else:
            filter_conditions.append({"repository_id": repository_ids[0]})

    # Handle user ID
    if user_id:
        filter_conditions.append({"user_id": user_id})

    # Handle metadata filters
    if metadata_filter:
        for key, value in metadata_filter.items():
            filter_conditions.append({"metadata." + key: value})

    # Only set filter if there are conditions
    if filter_conditions:
        if len(filter_conditions) > 1:
            search_kwargs["filter"] = {"$and": filter_conditions}
        else:
            search_kwargs["filter"] = filter_conditions[0]
    # Do not set 'filter' if filter_conditions is empty
    
    # Log search parameters in detail
    logger.info("Search parameters:")
    logger.info(f"Query: {query}")
    logger.info(f"Number of results (k): {k}")
    logger.info(f"Document IDs: {document_ids}")
    logger.info(f"Repository IDs: {repository_ids}")
    logger.info(f"User ID: {user_id}")
    logger.info(f"Metadata filter: {metadata_filter}")
    logger.info("Search kwargs:")
    logger.info(f"  k: {search_kwargs.get('k')}")
    logger.info(f"  filter: {json.dumps(search_kwargs.get('filter', {}), indent=2)}")
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    logger.info("Retriever created with configuration:")
    logger.info(f"  Type: {type(retriever).__name__}")
    logger.info(f"  Search kwargs: {json.dumps(search_kwargs, indent=2)}")
    
    # Optional: Use contextual compression for better retrieval
    # llm = ChatOpenAI(temperature=0)
    # compressor = LLMChainExtractor.from_llm(llm)
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor,
    #     base_retriever=retriever
    # )
    # logger.info("Compression retriever created:")
    # logger.info(f"  Base compressor: {type(compressor).__name__}")
    # logger.info(f"  Base retriever: {type(retriever).__name__}")
    
    # Retrieve documents
    try:
        docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} documents")
        
        if len(docs) == 0:
            logger.warning("No documents retrieved. This could be due to:")
            logger.warning("1. No matching documents in the vector store")
            logger.warning("2. Filter conditions too restrictive")
            logger.warning("3. Query not matching any stored content")
        # No fallback search
    except Exception as e:
        logger.error(f"Error during retrieval: {str(e)}")
        return []
    
    # Log details of each retrieved document
    for i, doc in enumerate(docs):
        logger.info(f"\nDocument {i+1}:")
        logger.info(f"  Content length: {len(doc.page_content)} characters")
        logger.info(f"  Content preview: {doc.page_content[:200]}...")
        logger.info(f"  Metadata: {json.dumps(doc.metadata, indent=2)}")
    
    # Format results with improved relevance scoring
    results = []
    for i, doc in enumerate(docs):
        # SAFEGUARD: Skip any chunks from global repository
        repository_id = doc.metadata.get("repository_id", "")
        if repository_id == "global_repository":
            logger.warning(f"Skipping chunk from global repository: {doc.metadata.get('chunk_id', 'unknown')}")
            continue
        
        # Calculate a more accurate relevance score
        # If the document has a score attribute, use it; otherwise calculate based on position
        if hasattr(doc, 'score') and doc.score is not None:
            relevance_score = float(doc.score)
        else:
            # Fallback: calculate score based on position (higher position = lower score)
            # This assumes that documents are returned in order of relevance
            relevance_score = max(MIN_RELEVANCE_FALLBACK, 1.0 - (RELEVANCE_DECAY_FACTOR * i))
        
        # Additional relevance boost for exact keyword matches
        query_lower = query.lower()
        content_lower = doc.page_content.lower()
        
        # Count keyword matches
        keyword_matches = sum(1 for word in query_lower.split() if word in content_lower)
        if keyword_matches > 0:
            relevance_score = min(1.0, relevance_score + (KEYWORD_BOOST_FACTOR * keyword_matches))
        
        result = {
            "chunk_id": doc.metadata.get("chunk_id", f"chunk_{i}"),
            "document_id": doc.metadata.get("document_id", ""),
            "repository_id": repository_id,
            "content": doc.page_content,
            "metadata": doc.metadata,
            "relevance_score": relevance_score
        }
        results.append(result)
        
        # Log each result
        logger.info(f"\nResult {i+1}:")
        logger.info(f"  Chunk ID: {result['chunk_id']}")
        logger.info(f"  Document ID: {result['document_id']}")
        logger.info(f"  Repository ID: {result['repository_id']}")
        logger.info(f"  Content length: {len(result['content'])} characters")
        logger.info(f"  Relevance score: {result['relevance_score']:.3f}")
        logger.info(f"  Content preview: {result['content'][:200]}...")
    
    # Sort results by relevance score (highest first)
    results.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return results

async def delete_document_chunks(document_id: str) -> bool:
    """
    Delete all chunks associated with a document.
    
    Args:
        document_id: ID of the document to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        vector_store.delete(filter={"document_id": document_id})
        vector_store.persist()
        return True
    except Exception as e:
        print(f"Error deleting document chunks: {str(e)}")
        return False

async def delete_repository_chunks(repository_id: str) -> bool:
    """
    Delete all chunks associated with a repository.
    
    Args:
        repository_id: ID of the repository to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        vector_store.delete(filter={"repository_id": repository_id})
        vector_store.persist()
        return True
    except Exception as e:
        print(f"Error deleting repository chunks: {str(e)}")
        return False 
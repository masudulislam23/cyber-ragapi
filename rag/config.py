"""
Configuration settings for the RAG system.
These settings control context length, chunk sizes, and retrieval parameters.
"""

# Context Management Settings
MAX_CONTEXT_LENGTH = 8000  # Maximum characters in context
MIN_RELEVANCE_SCORE = 0.5  # Minimum relevance score to include chunk
MAX_CHUNKS_TO_INCLUDE = 3  # Maximum number of chunks to include in context

# Document Processing Settings
CHUNK_SIZE = 600  # Default chunk size in characters
CHUNK_OVERLAP = 100  # Default chunk overlap in characters
MAX_CHUNKS_PER_BATCH = 100  # Maximum chunks to process in a single batch

# Retrieval Settings
DEFAULT_K = 3  # Default number of chunks to retrieve
MAX_K = 10  # Maximum number of chunks that can be requested

# Global Repository Settings
INCLUDE_GLOBAL_REPOSITORY = False  # Whether to include global repository in RAG queries

# Token Management
MAX_TOKENS_PER_REQUEST = 250000  # Maximum tokens per embedding request

# Relevance Scoring
KEYWORD_BOOST_FACTOR = 0.1  # Boost relevance score for each keyword match
MIN_RELEVANCE_FALLBACK = 0.1  # Minimum relevance score for fallback calculation
RELEVANCE_DECAY_FACTOR = 0.15  # How much relevance decreases per position

# Logging
LOG_RETRIEVAL_DETAILS = True  # Whether to log detailed retrieval information
LOG_CHUNK_CONTENT_PREVIEW = True  # Whether to log chunk content previews 
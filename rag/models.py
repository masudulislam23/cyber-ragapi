from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class FileType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    XLSX = "xlsx"
    PPT = "ppt"
    PPTX = "pptx"
    TXT = "txt"
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    UNKNOWN = "unknown"

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Repository(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    user_id: str
    metadata: Dict[Any, Any] = Field(default_factory=dict)
    is_global: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class Chunk(BaseModel):
    id: str
    document_id: str
    repository_id: str
    content: str
    metadata: Dict[Any, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
class Document(BaseModel):
    id: str
    filename: str
    file_type: FileType
    file_size: int
    repository_id: str
    user_id: str
    metadata: Dict[Any, Any] = Field(default_factory=dict)
    chunks: Optional[List[Chunk]] = None
    status: ProcessingStatus
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    repository_id: str
    status: str
    message: str

class RepositoryResponse(BaseModel):
    repository_id: str
    name: str
    description: Optional[str] = None
    document_count: Optional[int] = None
    created_at: datetime
    updated_at: datetime

class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    repository_id: str
    content: str
    metadata: Dict[Any, Any]
    relevance_score: float

class QueryResponse(BaseModel):
    query: str
    chunks: List[RetrievedChunk]
    context: str
    
# Request models
class CreateRepositoryRequest(BaseModel):
    id: Optional[str] = None
    name: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
class UpdateRepositoryRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class AuthChatRequest(BaseModel):
    messages: List[ChatMessage]

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    query: Optional[str] = None
    k: int = 3
    repository_ids: Optional[List[str]] = None
    document_ids: Optional[List[str]] = None
    metadata_filter: Optional[Dict[str, Any]] = None 
    thread_id: Optional[str] = None

class ChunkedUploadRequest(BaseModel):
    repository_id: str
    filename: str
    content_type: str
    total_chunks: int
    metadata: Optional[Dict[Any, Any]] = None

class ChunkedUploadResponse(BaseModel):
    upload_id: str
    message: str

class ChunkedUploadStatusResponse(BaseModel):
    upload_id: str
    chunks_received: int
    total_chunks: int
    status: str
    document_id: Optional[str] = None
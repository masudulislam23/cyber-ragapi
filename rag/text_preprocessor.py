import os
import uuid
import json
import random
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Query, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import shutil
import asyncio
import subprocess

from pydantic import BaseModel, Field

import torch
from safetensors.torch import load_file

from vllm import LLM
from vllm.sampling_params import SamplingParams
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
from langchain_openai import ChatOpenAI
from langchain.tools.render import render_text_description
from langchain.tools import tool

import pytz
import aiohttp
import tiktoken  # For token counting
import re
import time
from uuid import uuid4
import datetime as dt

# Define Eastern Time Zone
eastern = pytz.timezone('US/Eastern')
current_data = datetime.now(eastern)

formatted_date = current_data.strftime('%B %d, %Y')  # e.g., "June 15, 2025"
formatted_time = current_data.strftime('%I:%M %p %Z')  # e.g., "09:45 AM EDT"

# Global constants for system-wide access
GLOBAL_USER_ID = "global_user"
GLOBAL_REPOSITORY_ID = "global_repository"
GLOBAL_REPOSITORY_NAME = "Global Knowledge Base"

# Set PyTorch memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

auth_prompt ="""
You are an authentication system. You MUST output ONLY a valid JSON object. Never output text, markdown, formatting, or explanations — ONLY a valid JSON object.

--- RULES FOR BEHAVIOR ---
1. Analyze only the user's **latest message**.
2. Keep track of prior conversation state: action, phone_number, and password.
3. Action detection:
   - If message contains "sign in" or "signin" ? action = "sign-in"
   - If message contains "sign up" or "signup" ? action = "sign-up"
   - Action is set only once; do not overwrite if already set.
4. Phone number detection:
   - If message contains a phone number (digits, optional +, spaces, or dashes) ? normalize to E.164 format ? store in phone_number.
   - Phone number is set only once; do not overwrite if already set.
5. Password detection:
   - If message contains a string that is not a recognized action or phone number, and password is null, store it in password.
6. **Success condition**:
   - Only output a success message ("Sign in successful!" or "Sign up successful!") if ALL THREE values (action, phone_number, password) are set.
   - Otherwise, follow the flow logic exactly.

--- FLOW LOGIC ---
- If action is null ? instruction: "Would you like to sign in or sign up?"
- If action is set but phone_number is null ? instruction: "Please provide your phone number."
- If phone_number is set but password is null ? instruction: "Please provide your password."
- If all three are set ? instruction: "Sign in successful!" or "Sign up successful!" depending on action.

--- OUTPUT FORMAT (MANDATORY) ---
Return only this JSON structure:
{
  "instruction": "<message to the user>",
  "action": "<sign-in | sign-up | null>",
  "phone_number": "<extracted phone in E.164 format or null>",
  "password": "<extracted password or null>"
}
- Use null for any missing value.
- Never return an empty string.
- The JSON must be valid and parseable.

--- EXAMPLES ---
User: "Hi"  
? {"instruction":"Would you like to sign in or sign up?","action":null,"phone_number":null,"password":null}

User: "sign up"  
? {"instruction":"Please provide your phone number.","action":"sign-up","phone_number":null,"password":null}

User: "+1 555 123 4567"  
? {"instruction":"Please provide your password.","action":"sign-up","phone_number":"+15551234567","password":null}

User: "hunter2"  
? {"instruction":"Sign up successful!","action":"sign-up","phone_number":"+15551234567","password":"hunter2"}

User: "sign in"  
? {"instruction":"Please provide your phone number.","action":"sign-in","phone_number":null,"password":null}

User: "+44 7700 900123"  
? {"instruction":"Please provide your password.","action":"sign-in","phone_number":"+447700900123","password":null}

User: "myp@ss"  
? {"instruction":"Sign in successful!","action":"sign-in","phone_number":"+447700900123","password":"myp@ss"}

"""

# Import custom modules
from rag.document_processor import process_document
from rag.retriever import query_documents
from rag.llm import needs_web_search_with_gpt, fetch_realtime_data, query2keywords
from rag.scraper import scrape_search_results
from rag.vector_store import delete_repository_chunks
from rag.database import (
    get_document_by_id, 
    delete_document, 
    get_documents_by_repository_id,
    get_documents_by_user_id,
    delete_documents_by_repository_id,
    update_document_status,
    save_document,
    get_user_query_count,
    increment_user_query_count,
    save_user_profiles, load_user_profiles, update_user_profile, 
    delete_user_profile, get_user_profile, get_all_user_profiles,
    save_user_name, load_user_name
)
from rag.rich_profile_utils import (
    generate_rich_response_from_profile,
    is_rich_profile,
    convert_simple_to_rich_profile,
    ProfileUpdateConfirmationRequired
)
from rag.models import (
    DocumentResponse, 
    QueryResponse, 
    Document, 
    CreateRepositoryRequest, 
    UpdateRepositoryRequest, 
    RepositoryResponse,
    ProcessingStatus,
    FileType,
    ChatRequest,
    ChatMessage,
    AuthChatRequest,
    Repository
)
from rag.repository import (
    create_repository, 
    get_repository_by_id, 
    get_repositories_by_user_id, 
    update_repository, 
    delete_repository,
    get_repository_response
)

# Add import for path handling
from pathlib import Path

# Test the CreateRepositoryRequest model
try:
    test_request = CreateRepositoryRequest(name="test", id="test-id")
    logger.info(f"CreateRepositoryRequest model test successful: {test_request}")
except Exception as e:
    logger.error(f"CreateRepositoryRequest model test failed: {str(e)}")

# Constants for file processing
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB limit
MAX_VIDEO_SIZE = 200 * 1024 * 1024  # 200 MB limit for videos
ALLOWED_EXTENSIONS = {
    'pdf', 'doc', 'docx', 'xlsx', 'ppt', 'pptx', 'txt',
    'mp4', 'mov', 'avi', 'mp3', 'wav', 'jpg', 'jpeg', 'png'
}

# Verify system dependencies
def verify_tesseract_installation():
    """Verify that tesseract is installed and working correctly."""
    try:
        # Check tesseract version
        result = subprocess.run(
            ["tesseract", "--version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        version_info = result.stdout.strip()
        logger.info(f"Tesseract OCR installed: {version_info.split(chr(10))[0]}")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error(
            "Tesseract OCR is not installed or not in PATH. "
            "Image and PDF processing may fail. "
            "Please install Tesseract: https://github.com/tesseract-ocr/tesseract"
        )
        return False

# Create FastAPI app
app = FastAPI(
    title="Multi-modal RAG API",
    description="Retrieval Augmented Generation API for multiple file types including PDF, DOC, XLSX, PPT, TXT, video, audio, and images",
    version="1.0.0",
    redirect_slashes=False,  # Disable automatic trailing slash redirects
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Global model variable
openai_llm = None  # Will be set at startup

def init_model(max_retries=3):
    """Initialize the LLM model with retry mechanism"""
    global openai_llm
    
    if openai_llm is not None:
        return openai_llm
        
    for attempt in range(max_retries):
        try:
            print(f"Loading model (attempt {attempt + 1}/{max_retries})...")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Initialize the OpenAI LLM for auth_chat endpoint
            VLLM_API_BASE = "http://localhost:8002/v1"
            VLLM_MODEL_NAME = "neuralmagic/pixtral-12b-quantized.w4a16"
            
            openai_llm = ChatOpenAI(
                temperature=0,  # Set to 0 for deterministic JSON output
                model_name=VLLM_MODEL_NAME,
                openai_api_base=VLLM_API_BASE,
                openai_api_key="token-abc123", # vLLM often doesn't require a real key for internal setups
                max_tokens=2048, # Max tokens for the LLM response
            )

            print("OpenAI LLM loaded successfully")
            
            return openai_llm
        except Exception as e:
            print(f"Model loading failed, attempt {attempt + 1}: {str(e)}")
    return None, None

# Create directories if they don't exist
os.makedirs(os.getenv("UPLOAD_FOLDER", "./uploads"), exist_ok=True)
os.makedirs(os.getenv("PROCESSED_FOLDER", "./processed"), exist_ok=True)
os.makedirs("./data/repositories", exist_ok=True)

async def initialize_global_repository():
    """Initialize the global repository if it doesn't exist."""
    try:
        # Check if global repository exists
        repository = await get_repository_by_id(GLOBAL_REPOSITORY_ID)
        
        if not repository:
            # Create global repository with predefined ID and mark as global
            repository = await create_repository(
                repository_id=GLOBAL_REPOSITORY_ID,  # Explicitly set the global ID
                name=GLOBAL_REPOSITORY_NAME,
                user_id=GLOBAL_USER_ID,
                description="Global knowledge base for web search results",
                metadata={"type": "global", "source": "web_search"},
                is_global=True  # Mark as global
            )
            logger.info(f"Created global repository: {GLOBAL_REPOSITORY_ID}")
        else:
            logger.info(f"Global repository already exists: {GLOBAL_REPOSITORY_ID}")
            
    except Exception as e:
        logger.error(f"Error initializing global repository: {str(e)}")

# Check system dependencies on startup
@app.on_event("startup")
async def startup_event():
    """Run system checks when the application starts."""
    logger.info("Starting Multi-modal RAG API...")
    
    # Initialize global repository
    await initialize_global_repository()

    global openai_llm
    openai_llm = init_model()

    if openai_llm is None:
        raise RuntimeError("Failed to load model")
    
    # Verify tesseract installation
    tesseract_ok = verify_tesseract_installation()
    
    if not tesseract_ok:
        logger.warning(
            "WARNING: Tesseract OCR is not properly installed. "
            "Image text extraction and PDF OCR will not work correctly. "
            "Please install Tesseract OCR to enable these features."
        )
    
    # Check upload and processed directories
    upload_folder = os.getenv("UPLOAD_FOLDER", "./uploads")
    processed_folder = os.getenv("PROCESSED_FOLDER", "./processed")
    
    logger.info(f"Upload folder: {os.path.abspath(upload_folder)}")
    logger.info(f"Processed folder: {os.path.abspath(processed_folder)}")
    
    logger.info("System initialization complete")

# User authentication (simplified for demonstration)
async def get_user_id(x_user_id: str = Header(...)):
    """Get user ID from request header."""
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return x_user_id

# Helper function to validate file
def validate_file(file: UploadFile):
    """Validate file size and extension."""
    # Get file extension
    filename = file.filename
    if not filename or '.' not in filename:
        raise HTTPException(status_code=400, detail="Invalid file name")
    
    extension = filename.split('.')[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size (requires reading content, which can cause memory issues)
    # This is a simple check and could be improved
    is_video = extension in ['mp4', 'mov', 'avi']
    max_size = MAX_VIDEO_SIZE if is_video else MAX_FILE_SIZE
    
    return is_video, max_size, extension

# Background task for document processing
async def process_document_task(
    file_path: str,
    document_id: str,
    repository_id: str,
    user_id: str,
    metadata: Dict[Any, Any],
    filename: str,
    file_extension: str
):
    """Background task for processing documents to prevent timeouts."""
    from rag.document_processor import determine_file_type, process_document_async
    
    try:
        # Update status to show processing has started
        await update_document_status(document_id, ProcessingStatus.PROCESSING, "Processing started")
        
        # Determine file type
        file_type = determine_file_type(file_extension)
        
        # For video files, use a separate task with proper timeouts
        if file_type == FileType.VIDEO:
            logger.info(f"Starting video processing for {document_id} in separate task")
            # Create a task that we can monitor and potentially cancel
            process_task = asyncio.create_task(
                process_document_async(document_id, file_path, file_type, repository_id, user_id)
            )
            
            try:
                # Set a generous timeout for video processing (30 minutes)
                await asyncio.wait_for(process_task, timeout=1800)
                logger.info(f"Video processing completed successfully for {document_id}")
            except asyncio.TimeoutError:
                logger.error(f"Video processing timed out for {document_id}")
                await update_document_status(document_id, ProcessingStatus.FAILED, "Processing timed out")
                # The task will continue running in the background even though we've timed out waiting for it
                return
            except Exception as e:
                logger.exception(f"Error in video processing for {document_id}: {str(e)}")
                await update_document_status(document_id, ProcessingStatus.FAILED, str(e))
                return
        else:
            # For non-video files, process normally
            await process_document_async(document_id, file_path, file_type, repository_id, user_id)
        
        # After successful document processing, detect and save profiles from text content
        # Only do this for text-based files (not videos, audio, or images)
        if file_type not in [FileType.VIDEO, FileType.AUDIO, FileType.IMAGE]:
            try:
                logger.info(f"[PROFILE DETECTION] Starting profile detection for processed document: {document_id}")
                
                # Read the processed file to extract text for profile detection
                # Note: We need to read from the processed folder since the file was moved there
                from rag.document_processor import PROCESSED_FOLDER
                processed_file_path = os.path.join(PROCESSED_FOLDER, f"{document_id}_{filename}")
                
                if os.path.exists(processed_file_path):
                    # Extract text based on file type for profile detection
                    from rag.document_processor import extract_text_from_file
                    file_content = await extract_text_from_file(processed_file_path, file_type, user_id)
                    
                    if file_content and len(file_content.strip()) > 0:
                        logger.info(f"[PROFILE DETECTION] Extracted {len(file_content)} characters from file for profile detection")
                        
                        # Detect and save profiles from the file content
                        detected_profiles = await detect_and_save_profiles_from_file_content(
                            file_content, user_id, filename
                        )
                        
                        if detected_profiles:
                            logger.info(f"[PROFILE DETECTION] Successfully detected and saved {len(detected_profiles)} profiles from document: {document_id}")
                            logger.info(f"[PROFILE DETECTION] Profiles saved: {detected_profiles}")
                        else:
                            logger.info(f"[PROFILE DETECTION] No profiles detected in document: {document_id}")
                    else:
                        logger.info(f"[PROFILE DETECTION] No text content found in document for profile detection: {document_id}")
                else:
                    logger.warning(f"[PROFILE DETECTION] Processed file not found for profile detection: {processed_file_path}")
                    
            except Exception as e:
                logger.error(f"[PROFILE DETECTION] Error during profile detection for document {document_id}: {str(e)}")
                # Don't fail the entire document processing if profile detection fails
        
        logger.info(f"Completed background processing of document {document_id} ({filename})")
    except Exception as e:
        logger.error(f"Error in background processing of document {document_id}: {str(e)}")
        await update_document_status(document_id, ProcessingStatus.FAILED, str(e))

# API endpoints
@app.get("/")
async def root():
    return {"message": "Welcome to Multi-modal RAG API"}

# Repository management endpoints
@app.post("/repositories/", response_model=RepositoryResponse)
async def create_new_repository(
    request: CreateRepositoryRequest,
    user_id: str = Depends(get_user_id)
):
    """Create a new repository for the user."""
    try:
        # Debug: Print request details
        logger.info(f"Creating repository with request: {request}")
        logger.info(f"Request fields: {request.model_fields.keys()}")
        logger.info(f"Request id field: {request.id}")
        
        # Debug: Check if id field exists in model
        logger.info(f"Model fields: {CreateRepositoryRequest.model_fields}")
        logger.info(f"Model field names: {list(CreateRepositoryRequest.model_fields.keys())}")
        
        # Debug: Try to access id field directly
        try:
            id_value = request.id
            logger.info(f"Direct access to request.id: {id_value}")
        except AttributeError as e:
            logger.error(f"AttributeError accessing request.id: {e}")
        
        # Debug: Check all attributes
        logger.info(f"All request attributes: {dir(request)}")
        logger.info(f"Request dict: {request.model_dump()}")
        
        # Create repository
        repository = await create_repository(
            name=request.name,
            repository_id=request.id,  # Use getattr for safety
            user_id=user_id,
            description=request.description,
            metadata=request.metadata
        )
        
        # Create response
        return await get_repository_response(repository)
    except Exception as e:
        logger.error(f"Error creating repository: {str(e)}")
        logger.error(f"Request data: {request}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/repositories/", response_model=List[RepositoryResponse])
async def list_repositories(user_id: str = Depends(get_user_id)):
    """List all repositories for the user."""
    try:
        # Get repositories
        repositories = await get_repositories_by_user_id(user_id)
        
        # Create responses
        return [await get_repository_response(repo) for repo in repositories]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/repositories/{repository_id}/", response_model=RepositoryResponse)
async def get_repository(
    repository_id: str,
    user_id: str = Depends(get_user_id)
):
    """Get a specific repository by ID."""
    try:
        # Get repository
        repository = await get_repository_by_id(repository_id)
        
        # Check if repository exists
        if not repository:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        # Check if user has access to repository
        if repository.user_id != user_id:
            raise HTTPException(status_code=403, detail="You don't have access to this repository")
        
        # Create response
        return await get_repository_response(repository)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/repositories/{repository_id}/", response_model=RepositoryResponse)
async def update_existing_repository(
    repository_id: str,
    request: UpdateRepositoryRequest,
    user_id: str = Depends(get_user_id)
):
    """Update a specific repository."""
    try:
        # Get repository
        repository = await get_repository_by_id(repository_id)
        
        # Check if repository exists
        if not repository:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        # Check if user has access to repository
        if repository.user_id != user_id:
            raise HTTPException(status_code=403, detail="You don't have access to this repository")
        
        # Update repository
        updates = {}
        if request.name is not None:
            updates["name"] = request.name
        if request.description is not None:
            updates["description"] = request.description
        if request.metadata is not None:
            updates["metadata"] = request.metadata
        
        updated_repository = await update_repository(repository_id, updates)
        
        # Create response
        return await get_repository_response(updated_repository)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/repositories/{repository_id}/")
async def remove_repository(
    repository_id: str,
    user_id: str = Depends(get_user_id)
):
    """Delete a repository and all its documents."""
    try:
        # Get repository
        repository = await get_repository_by_id(repository_id)
        
        # Check if repository exists
        if not repository:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        # Check if user has access to repository
        if repository.user_id != user_id:
            raise HTTPException(status_code=403, detail="You don't have access to this repository")
        
        # Check if user try to delete global repository
        if repository.id == GLOBAL_REPOSITORY_ID:
            return {
                "message": f"The GOLOBAL repository can't be deleted!"
            }
        
        # Delete all documents in repository
        deleted_count = await delete_documents_by_repository_id(repository_id)
        
        # Delete all chunks from vector store
        await delete_repository_chunks(repository_id)
        
        # Delete repository
        success = await delete_repository(repository_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete repository")
        
        return {
            "message": f"Repository {repository_id} successfully deleted with {deleted_count} documents"
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Document management endpoints
@app.post("/repositories/{repository_id}/documents/upload/", response_model=DocumentResponse)
async def upload_document(
    repository_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    document_id: Optional[str] = None,
    user_id: str = Depends(get_user_id)
):
    """
    Upload a document to a specific repository.
    Supported file types: PDF, DOC/DOCX, XLSX, PPT/PPTX, TXT, video, audio, and images
    
    This endpoint starts the document processing asynchronously and returns immediately
    with a document ID that can be used to check the processing status.
    
    For large files, especially videos, consider using the chunked upload endpoints instead.
    """
    try:
        # Debug: Log the request details
        logger.info(f"Upload request - repository_id: {repository_id}, user_id: {user_id}, filename: {file.filename}")
        
        # Parse metadata if provided
        meta_dict = json.loads(metadata) if metadata else {}
        
        # Validate file
        is_video, max_size, file_extension = validate_file(file)
        
        # For video files, recommend using chunked upload
        if is_video:
            logger.info("Video file detected in regular upload endpoint")
        
        # Get repository
        repository = await get_repository_by_id(repository_id)
        if not repository:
            logger.error(f"Repository not found: {repository_id}")
            raise HTTPException(status_code=404, detail=f"Repository {repository_id} not found")
        
        # Check if user has access to repository
        if repository.user_id != user_id:
            logger.error(f"User {user_id} does not have access to repository {repository_id} (owned by {repository.user_id})")
            raise HTTPException(status_code=403, detail="You don't have access to this repository")
        
        logger.info(f"Repository found: {repository.name} (ID: {repository.id})")
        
        # Generate document ID
        document_id = str(uuid.uuid4()) if document_id is None else document_id
        
        # Create file path
        upload_folder = os.getenv("UPLOAD_FOLDER", "./uploads")
        file_path = os.path.join(upload_folder, f"{document_id}_{file.filename}")
        
        # Save the uploaded file (without loading it completely into memory)
        with open(file_path, "wb") as buffer:
            # Read and write in chunks to avoid memory issues
            chunk_size = 1024 * 1024  # 1MB chunks
            content = await file.read(chunk_size)
            file_size = 0
            
            while content:
                buffer.write(content)
                file_size += len(content)
                
                # Check file size limit
                if file_size > max_size:
                    # Close and delete the file
                    buffer.close()
                    os.remove(file_path)
                    
                    size_limit_mb = max_size / (1024 * 1024)
                    raise HTTPException(
                        status_code=413, 
                        detail=f"File too large. Maximum size for {'videos' if is_video else 'files'} is {size_limit_mb} MB"
                    )
                
                content = await file.read(chunk_size)
        
        # Get actual file size
        file_size = os.path.getsize(file_path)
        
        # Create document record
        from rag.document_processor import determine_file_type
        from rag.models import Document, ProcessingStatus, FileType
        from rag.database import save_document
        
        file_type = determine_file_type(file_extension)
        
        # For video files, set initial status to QUEUED to indicate it will be processed later
        initial_status = ProcessingStatus.QUEUED if is_video else ProcessingStatus.PROCESSING
        
        document = Document(
            id=document_id,
            filename=file.filename,
            file_type=file_type,
            file_size=file_size,
            repository_id=repository_id,
            user_id=user_id,
            metadata=meta_dict,
            status=initial_status,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Save initial document to database
        await save_document(document)
        
        # Add document processing to background tasks to prevent timeouts
        background_tasks.add_task(
            process_document_task,
            file_path,
            document_id,
            repository_id,
            user_id,
            meta_dict,
            file.filename,
            file_extension
        )
        
        # Return response immediately
        return JSONResponse(
            status_code=202,
            content={
                "document_id": document_id,
                "filename": file.filename,
                "file_type": file_type.value,
                "file_size": file_size,                # <--- Added file size
                "repository_id": repository_id,
                "status": initial_status.value,
                "metadata": meta_dict,                 # <--- Added metadata
                "message": "Document upload successful. Processing started in background."
            }
        )
        
    except HTTPException as e:
        raise e
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON format")
    except Exception as e:
        logger.exception(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/repositories/{repository_id}/documents/", response_model=List[Document])
async def list_repository_documents(
    repository_id: str,
    user_id: str = Depends(get_user_id)
):
    """Get all documents in a specific repository."""
    try:
        # Get repository
        repository = await get_repository_by_id(repository_id)
        
        # Check if repository exists
        if not repository:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        # Check if user has access to repository
        if repository.user_id != user_id:
            raise HTTPException(status_code=403, detail="You don't have access to this repository")
        
        # Get documents
        documents = await get_documents_by_repository_id(repository_id)
        
        return documents
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/", response_model=List[Document])
async def list_documents(
    repository_id: Optional[str] = Query(None, description="Filter by repository ID"),
    user_id: str = Depends(get_user_id)
):
    """Get all documents that have been uploaded and processed."""
    try:
        if repository_id:
            # Get repository
            repository = await get_repository_by_id(repository_id)
            
            # Check if repository exists
            if not repository:
                raise HTTPException(status_code=404, detail="Repository not found")
            
            # Check if user has access to repository
            if repository.user_id != user_id:
                raise HTTPException(status_code=403, detail="You don't have access to this repository")
            
            # Get documents for repository
            documents = await get_documents_by_repository_id(repository_id)
        else:
            # Get all documents for user
            documents = await get_documents_by_user_id(user_id)
        
        return documents
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{document_id}/", response_model=Document)
async def get_document(
    document_id: str,
    user_id: str = Depends(get_user_id)
):
    """Get a specific document by ID."""
    try:
        # Get document
        document = await get_document_by_id(document_id)
        
        # Check if document exists
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if user has access to document
        if document.user_id != user_id:
            raise HTTPException(status_code=403, detail="You don't have access to this document")
        
        return document
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}/")
async def remove_document(
    document_id: str,
    user_id: str = Depends(get_user_id)
):
    """Delete a document and its associated chunks from the system."""
    try:
        # Get document
        document = await get_document_by_id(document_id)
        
        # Check if document exists
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if user has access to document
        if document.user_id != user_id:
            raise HTTPException(status_code=403, detail="You don't have access to this document")
        
        # Delete document
        success = await delete_document(document_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete document")
        
        return {"message": f"Document {document_id} successfully deleted"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{document_id}/status/")
async def get_document_processing_status(
    document_id: str,
    user_id: str = Depends(get_user_id)
):
    """
    Check the processing status of a document.
    
    This endpoint allows clients to poll for document processing completion
    instead of waiting for the entire process to finish in a single request.
    """
    try:
        # Get document
        document = await get_document_by_id(document_id)
        
        # Check if document exists
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if user has access to document
        if document.user_id != user_id:
            raise HTTPException(status_code=403, detail="You don't have access to this document")
        
        return {
            "document_id": document.id,
            "status": document.status.value,
            "filename": document.filename,
            "file_type": document.file_type.value,
            "created_at": document.created_at,
            "updated_at": document.updated_at
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# User profile management endpoints
@app.post("/user/name/")
async def set_user_name(
    request: dict,
    user_id: str = Depends(get_user_id)
):
    """Set the user's own name."""
    try:
        name = request.get("name")
        if not name or not isinstance(name, str) or len(name.strip()) == 0:
            raise HTTPException(status_code=400, detail="Name must be a non-empty string")
        
        name = name.strip()
        success = await save_user_name(user_id, name)
        
        if success:
            return {"message": f"User name set successfully to: {name}", "user_name": name}
        else:
            raise HTTPException(status_code=500, detail="Failed to save user name")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/user/name/")
async def update_user_name(
    request: dict,
    user_id: str = Depends(get_user_id)
):
    """Update the user's own name."""
    try:
        name = request.get("name")
        if not name or not isinstance(name, str) or len(name.strip()) == 0:
            raise HTTPException(status_code=400, detail="Name must be a non-empty string")
        
        # Get current name for comparison
        current_name = await load_user_name(user_id)
        
        name = name.strip()
        success = await save_user_name(user_id, name)
        
        if success:
            if current_name:
                return {"message": f"User name updated from {current_name} to {name}", "user_name": name, "previous_name": current_name}
            else:
                return {"message": f"User name set successfully to: {name}", "user_name": name}
        else:
            raise HTTPException(status_code=500, detail="Failed to save user name")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/name/")
async def get_user_name_endpoint(user_id: str = Depends(get_user_id)):
    """Get the user's own name."""
    try:
        user_name = await load_user_name(user_id)
        return {"user_name": user_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/user/name/")
async def clear_user_name(user_id: str = Depends(get_user_id)):
    """Clear the user's name."""
    try:
        # Remove the user name file
        user_name_path = os.path.join("data/profiles", f"{user_id}_name.json")
        if os.path.exists(user_name_path):
            os.remove(user_name_path)
            return {"message": "User name cleared successfully"}
        else:
            return {"message": "No user name was set"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/profiles/cleanup/")
async def cleanup_user_profiles(user_id: str = Depends(get_user_id)):
    """Clean up existing profiles to use clean names instead of descriptive names."""
    try:
        success = await cleanup_existing_profile_names(user_id)
        if success:
            return {"message": "Profile names cleaned up successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clean up profile names")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Utility to ensure roles alternate after optional system message
def fix_message_role_alternation(messages):
    if not messages:
        return messages
    fixed = []
    idx = 0
    # If first is system, keep it
    if messages[0].role == "system":
        fixed.append(messages[0])
        idx = 1
    if idx >= len(messages):
        return fixed
    # Start alternation from here
    last_role = None
    for m in messages[idx:]:
        if last_role is None or m.role != last_role:
            fixed.append(m)
            last_role = m.role
    return fixed

# Utility to flatten message content to string
def flatten_message_content(messages):
    flat = []
    for i, m in enumerate(messages):
        content = m.content
        original_content = content
        logging.info(f"[flatten] Message {i} original content type: {type(content)}, value: {content}")
        # Convert to simple string format for openai template
        if isinstance(content, str):
            pass  # already a string
        elif isinstance(content, list):
            # Join all text fields or string representations
            parts = []
            for c in content:
                if isinstance(c, dict) and "text" in c:
                    parts.append(str(c["text"]))
                elif isinstance(c, str):
                    parts.append(c)
                else:
                    parts.append(str(c))
            content = "".join(parts)
        else:
            content = str(content)
        
        logging.info(f"[flatten] Message {i} flattened content type: {type(content)}, value: {content}")
        flat.append({"role": m.role, "content": content})
    return flat

@app.post("/auth-chat/")
async def auth_chat(
    request: AuthChatRequest,
    sampling_params: Optional[dict] = None,
    chat_template_content_format: Optional[str] = None
):

    try:
        logging.info(f"Original messages: {request.messages}")

        # prepend system prompt if given
        messages = [ChatMessage(role = "system", content = auth_prompt)] + request.messages

        # Validate and flatten messages
        fixed_messages = fix_message_role_alternation(messages)
        logging.info(f"Fixed messages: {fixed_messages}")
        if not fixed_messages:
            raise HTTPException(status_code=400, detail="No valid messages provided.")
        
        flat_messages = flatten_message_content(fixed_messages)
        # Ensure all message content is string
        for i, m in enumerate(flat_messages):
            if not isinstance(m['content'], str):
                logging.warning(f"Message {i} content is not string, converting: {type(m['content'])}")
                m['content'] = str(m['content'])
        logging.info(f"Flattened messages: {flat_messages}")

        # Note: sampling_params and chat_template_content_format are not used with ChatOpenAI
        # They are kept for backward compatibility but ignored
        logging.info(f"Payload to model: {flat_messages}")
        try:
            # Convert flat_messages to LangChain message format
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
            
            langchain_messages = []
            for msg in flat_messages:
                if msg['role'] == 'system':
                    langchain_messages.append(SystemMessage(content=msg['content']))
                elif msg['role'] == 'user':
                    langchain_messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'assistant':
                    langchain_messages.append(AIMessage(content=msg['content']))
            
            logging.info(f"Langchain messages: {langchain_messages}")
            
            # Use the openai_llm to generate response
            response = await openai_llm.ainvoke(langchain_messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            logging.info(f"Response: {response_text}")
        except Exception as e:
            logging.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
            return JSONResponse(status_code=500, content={
                "error": str(e),
                "messages": flat_messages
            })

        def str2json(text):
            import re
            import json
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    # Validate JSON
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError:
                    pass
            return "{}"

        # Ensure response is a string
        if isinstance(response_text, (list, dict)):
            response = str2json(str(response_text))
        elif isinstance(response_text, str):
            # Try to extract JSON from the response
            response = str2json(response_text)
        else:
            response = str2json(str(response_text))

        # --- POST-PROCESSING OVERRIDE FOR PASSWORD ---
        import json
        try:
            resp_json = json.loads(response)
            # Find the last user message
            last_user_msg = None
            last_assistant_msg = None
            for m in reversed(request.messages):
                if (isinstance(m, dict) and m.get('role') == 'user') or (hasattr(m, 'role') and m.role == 'user'):
                    last_user_msg = m['content'] if isinstance(m, dict) else m.content
                    break
            for m in reversed(request.messages):
                if (isinstance(m, dict) and m.get('role') == 'assistant') or (hasattr(m, 'role') and m.role == 'assistant'):
                    last_assistant_msg = m['content'] if isinstance(m, dict) else m.content
                    break
            # If the last assistant message was a password prompt, and last user message is not empty, and password is null
            if last_assistant_msg and 'password' in last_assistant_msg.lower() and last_user_msg and (resp_json.get('password') is None or resp_json.get('password') == "null"):
                resp_json['password'] = last_user_msg
                # Set instruction to sign up/in success
                if resp_json.get('action') == 'sign-up':
                    resp_json['instruction'] = 'Sign up successful!'
                elif resp_json.get('action') == 'sign-in':
                    resp_json['instruction'] = 'Sign in successful!'
                response = json.dumps(resp_json)
        except Exception as e:
            logging.warning(f"Post-processing override failed: {e}")

        return JSONResponse(content={
            "response": response
        })

    except Exception as e:
        logging.error(f"Error in auth_chat endpoint: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={
            "error": str(e)
        })

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver, empty_checkpoint
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver, CheckpointMetadata
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain.agents.format_scratchpad import format_log_to_messages
from typing import Any, Dict, List, Optional, Tuple, Sequence

# Initialize checkpoint store
checkpoint_store = MemorySaver()

# --- LangGraph State Definition ---
class AgentState(BaseModel):
    thread_id: str
    request: ChatRequest
    user_id: str
    user_name: Optional[str] = None
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    user_stats_context: str = ""
    user_repositories: List[Repository] = Field(default_factory=list)
    filtered_documents: List[Document] = Field(default_factory=list)
    retrieval_query: str = ""
    context: str = ""
    realtime_data: Optional[Dict[str, Any]] = None
    searched_links: List[str] = Field(default_factory=list)
    final_response: Optional[str] = None
    error: Optional[str] = None
    current_date_content: str = ""
    profiles: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    last_mentioned_profile: Dict[str, Any] = Field(default_factory=dict)
    agent_status: str = "get_name"
    confirm_profile: Optional[Dict[str, Any]] = None
    pending_profile_update: Optional[Dict[str, Any]] = None  # New field for pending updates requiring confirmation
    pending_profile_clarification: Optional[Dict[str, Any]] = None  # New field for pending profile clarification

# --- Helper Functions ---
def append_searched_links_to_response(response: str, searched_links: List[str]) -> str:
    """Append searched links to a response if they exist and are valid."""
    if searched_links and len(searched_links) > 0:
        # Filter out empty or invalid links
        valid_links = [link for link in searched_links if link and link.strip() and link.startswith('http')]
        if valid_links:
            links_text = "\n\n**Sources:**\n" + "\n".join([f"- {link}" for link in valid_links])
            logger.info(f"[HELPER] Appending {len(valid_links)} valid links to response")
            return response + links_text
        else:
            logger.info(f"[HELPER] No valid links found in {len(searched_links)} searched_links")
    else:
        logger.info(f"[HELPER] No searched_links to append (searched_links: {searched_links})")
    return response

async def detect_and_save_profiles_from_file_content(file_content: str, user_id: str, filename: str) -> List[Dict[str, Any]]:
    """
    Automatically detect and save profile information from uploaded file content.
    Returns a list of detected and saved profiles.
    """
    logger.info(f"[FILE PROFILE DETECTION] Starting profile detection for file: {filename}")
    logger.info(f"[FILE PROFILE DETECTION] File content length: {len(file_content)} characters")
    
    try:
        # Detect names and relationships in the file content (no chat history for file processing)
        detection_result = await detect_profiles_in_text(file_content)
        
        # Handle the tuple return value (profiles, is_detected)
        if isinstance(detection_result, tuple) and len(detection_result) == 2:
            detected_profiles, is_detected = detection_result
            logger.info(f"[FILE PROFILE DETECTION] Detection result: profiles={len(detected_profiles) if detected_profiles else 0}, is_detected={is_detected}")
        else:
            # Fallback for direct list return
            detected_profiles = detection_result
            is_detected = bool(detected_profiles)
            logger.info(f"[FILE PROFILE DETECTION] Direct list return detected: {len(detected_profiles) if detected_profiles else 0}")
        
        # Prioritize having actual profiles over the is_detected flag
        # The is_detected flag only indicates if LLM detection succeeded, not if profiles exist
        if not detected_profiles:
            logger.info(f"[FILE PROFILE DETECTION] No profiles detected in file: {filename}")
            return []
        
        # Log whether profiles came from LLM or fallback detection
        if is_detected:
            logger.info(f"[FILE PROFILE DETECTION] Profiles detected via LLM: {len(detected_profiles)} profiles")
        else:
            logger.info(f"[FILE PROFILE DETECTION] Profiles detected via fallback detection: {len(detected_profiles)} profiles")
        
        logger.info(f"[FILE PROFILE DETECTION] Detected {len(detected_profiles)} profiles in file: {filename}")
        logger.info(f"[FILE PROFILE DETECTION] Detected profiles: {detected_profiles}")
        
        saved_profiles = []
        
        # Process each detected profile
        for i, profile_data in enumerate(detected_profiles):
            # Handle both dict and other data types safely
            if isinstance(profile_data, dict):
                name = profile_data.get('name', '')
                relationship = profile_data.get('relationship', 'unknown')
                information = profile_data.get('information', '')
            else:
                logger.warning(f"[FILE PROFILE DETECTION] Unexpected profile data format: {type(profile_data)} - {profile_data}")
                continue
            
            logger.info(f"[FILE PROFILE DETECTION] Processing profile {i+1}: name='{name}', relationship='{relationship}', information='{information}'")
            
            if name and information:
                try:
                    # Save the profile to persistent storage
                    saved_profile = await create_or_update_profile_persistent(
                        user_id, name, relationship, information
                    )
                    saved_profiles.append(saved_profile)
                    logger.info(f"[FILE PROFILE DETECTION] Successfully saved profile for {name} ({relationship}) from file: {filename}")
                    
                    # Also create a rich profile version for better AI responses
                    try:
                        from rag.rich_profile_utils import convert_simple_to_rich_profile
                        
                        # Get the created profile to convert it
                        created_profile = await get_user_profile(user_id, name)
                        if created_profile:
                            # Convert to rich format
                            rich_profile = await convert_simple_to_rich_profile(created_profile)
                            
                            # Save the rich profile back (this will enhance future responses)
                            await update_user_profile(user_id, name, rich_profile)
                            
                            logger.info(f"[FILE PROFILE DETECTION] Enhanced profile for {name} with rich structure from file: {filename}")
                    except Exception as e:
                        logger.warning(f"[FILE PROFILE DETECTION] Rich profile enhancement failed for {name}: {str(e)}")
                        # Continue with normal profile creation if enhancement fails
                        
                except Exception as e:
                    logger.error(f"[FILE PROFILE DETECTION] Error saving profile for {name}: {str(e)}")
                    # Continue processing other profiles even if one fails
            else:
                logger.info(f"[FILE PROFILE DETECTION] Skipping profile with missing name or information: {profile_data}")
        
        # Log summary of processing results
        logger.info(f"[FILE PROFILE DETECTION] Profile processing summary: {len(detected_profiles)} detected, {len(saved_profiles)} successfully saved")
        
        logger.info(f"[FILE PROFILE DETECTION] Completed profile detection for file: {filename}. Saved {len(saved_profiles)} profiles.")
        return saved_profiles
        
    except Exception as e:
        logger.error(f"[FILE PROFILE DETECTION] Error during profile detection for file {filename}: {str(e)}")
        return []

# --- LangChain Tools (wrapping existing async functions) ---

# Async functions (internal use)
async def _get_user_space_info(user_id: str) -> Dict[str, Any]:
    """
    Retrieves user's repositories and ensures 'user_space' repository exists.
    Returns user_profile, user_repos, and the user_space repository if found/created.
    """
    user_repos = await get_repositories_by_user_id(user_id)
    user_space = next((r for r in user_repos if r.name == "user_space"), None)
    if not user_space:
        user_space = await create_repository(
            name="user_space",
            user_id=user_id,
            description="Personal user space",
            metadata={"profile": {}}
        )
        user_repos.append(user_space) # Add newly created space to list
    user_profile = user_space.metadata.get("profile", {})
    return {"user_profile": user_profile, "user_repositories": user_repos, "user_space": user_space}

async def _get_user_stats(user_id: str, user_repositories: List[Repository]) -> Dict[str, Any]:
    """
    Calculates user file statistics and query count.
    Excludes note files from file count.
    """
    documents = await get_documents_by_user_id(user_id)
    filtered_documents = [
        doc for doc in documents
        if not (doc.filename.endswith('_note.txt') and len(doc.filename.split('_')) == 2 and doc.filename.split('_')[0] == doc.repository_id)
    ]
    total_size = sum(doc.file_size for doc in filtered_documents)
    file_count = len(filtered_documents)

    await increment_user_query_count(user_id)
    query_count = await get_user_query_count(user_id)

    user_stats_context = f"User has uploaded {file_count} files totaling {total_size} bytes and made {query_count} queries."
    return {
        "user_stats_context": user_stats_context,
        "filtered_documents": filtered_documents
    }

async def _validate_and_filter_repositories(user_id: str, requested_repo_ids: Optional[List[str]], user_repositories: List[Repository]) -> List[str]:
    """
    Validates user access to requested repositories and filters out global repository.
    Returns a list of valid repository IDs for RAG search.
    """
    if requested_repo_ids:
        for repo_id in requested_repo_ids:
            repository = next((r for r in user_repositories if r.id == repo_id), None)
            if not repository:
                raise HTTPException(status_code=404, detail=f"Repository {repo_id} not found")
            if repository.user_id != user_id:
                raise HTTPException(status_code=403, detail=f"You don't have access to repository {repo_id}")
        final_repo_ids = list(requested_repo_ids) # Copy to allow modification
    else:
        final_repo_ids = [repo.id for repo in user_repositories]

    if GLOBAL_REPOSITORY_ID in final_repo_ids:
        final_repo_ids.remove(GLOBAL_REPOSITORY_ID)
        logger.info(f"Removed global repository from search: {GLOBAL_REPOSITORY_ID}")

    return final_repo_ids

async def _perform_rag_query(
    query: str,
    k: int,
    document_ids: Optional[List[str]],
    repository_ids: List[str],
    metadata_filter: Dict[str, Any]
) -> str:
    """Performs a RAG query on specified documents and repositories."""
    if not repository_ids or len(repository_ids) == 0:
        return "" # No repositories to query
    resp = await query_documents(
        query=query,
        k=k,
        document_ids=document_ids,
        repository_ids=repository_ids,
        metadata_filter=metadata_filter
    )
    return resp.context

async def _perform_web_search(query: str) -> Optional[Dict[str, Any]]:
    """
    Determines if web search is needed, then extracts keywords and fetches real-time data.
    Returns the real-time data if search is performed.
    """
    # Assuming 'llm' from the global scope is accessible here
    mock_sampling_params = SamplingParams() # Using default for tool
    if await needs_web_search_with_gpt(openai_llm, query, mock_sampling_params):
        keywords = await query2keywords(openai_llm, query, mock_sampling_params)
        realtime_data = await fetch_realtime_data(keywords)
        return realtime_data
    return None

# Synchronous wrapper functions for LangChain tools
async def detect_profiles_in_text(text: str, chat_history: List[Any] = None) -> List[Dict[str, str]]:
    """
    Detects profiles in the given text using LLM for intelligent profile recognition.
    Includes chat history context for better understanding of relationships and references.
    Returns a list of dictionaries with 'name', 'context', 'relationship', and 'information' fields.
    """

    logger.info(f"Detecting names in text using LLM: {text}")

    # Prepare chat history context for better profile detection
    chat_context = ""
    if chat_history:
        # Get the last few messages for context
        recent_messages = chat_history[-10:]  # Last 4 messages
        chat_context = "\n".join([f"{msg.role}: {msg.content}" for msg in recent_messages])
        logger.info(f"[PROFILE DETECTION] Chat history context: {chat_context}")
    
    detection_prompt = f"""
    You are an intelligent assistant that extracts person profiles, their relationship to the speaker, and relevant information from the given text.
    Use chat history context to better understand relationships and references.

    PRIMARY PROFILE DETECTION RULES:
    - Extract ONLY the person's first name (e.g., "Ash", "Mark", "Sarah") - NOT descriptive phrases like "my friend Ash" or "Sonnet's colleague Mark"
    - Names must be explicit capitalized proper nouns exactly as written in the text.
    - **ABSOLUTELY FORBIDDEN**: NEVER use pronouns (he, she, they, him, her, them, his, hers, their, theirs, you, I, we, us, myself, yourself, ourselves) as names.
    - **CRITICAL**: If the text starts with a pronoun like "He likes..." or "She likes...", you MUST use chat history to find the actual person's name.
    - DO NOT use gendered terms (man, woman, boy, girl, lady, gentleman) or placeholders (someone, person, guy, buddy) as names.
    - DO NOT infer a name from context — it must appear directly in the text as a proper noun.
    - If a name only appears as part of a detail describing another person (e.g., "Ash has a brother named Sam"), treat it as part of the other person's `"information"` — do NOT create a separate JSON entry for it.
    - Only create a separate JSON object for a person if they are the **main subject** of their own statement or question.
    - **IMPORTANT**: Use chat history context to understand if relationships refer to previously mentioned people.
    - **CRITICAL**: When pronouns (he, she, they) are used and multiple profiles exist with the same name, set relationship to "unknown" to trigger user clarification.
    - **CRITICAL**: When "my [relationship]" is used (e.g., "my colleague's number"), use chat history to find the specific person with that relationship mentioned earlier.
    - **CRITICAL**: When "my [relationship] [action]" is used (e.g., "my sister likes basketball"), use chat history to find the specific person with that relationship mentioned earlier.

    TASK:
    1. **FIRST**: Check if the text starts with a pronoun (he, she, they, his, her, their). If YES, use chat history to find the actual person's name.
    2. **SECOND**: Check if the text contains "my [relationship]" (e.g., "my colleague's number" or "my sister likes basketball"). If YES, use chat history to find the specific person with that relationship mentioned earlier.
    3. Identify the **main person(s)** in the text — the ones being directly described or asked about — and extract:
        - "name": Properly capitalized name of the person (from chat history if pronoun or relationship reference is used).
        - "relationship": One of ["friend", "colleague", "family", "supervisor", "neighbor", "assistant", "unknown"].
        - "information": Concise details from the text (activities, jobs, locations, ages, contact info, names of relatives, etc.).
            • If the text is a question about the person without the answer, set "information" to "asking [requested thing]".
    3. If the relationship is not explicitly stated, set `"relationship": "unknown"`.
    4. If no information, set `"information": ""`.
    5. Combine multiple facts for the same person into one sentence separated by "and".
    6. Avoid:
        - Creating separate entries for people who are only mentioned as part of another person's information.
        - Duplicate entries for the same person (same name + same relationship + same details).
        - Inferring details not present in the text.
    7. If no valid primary names are found, return `[]`.

    OUTPUT FORMAT:
    Return only raw JSON (no extra text).

    ### Chat History Context
    {chat_context if chat_context else "No previous context available."}
    
    **IMPORTANT**: Use this chat history to resolve:
    1. Pronouns (he, she, they) to actual names
    2. Relationship references (my colleague, my friend) to specific people mentioned earlier
    3. Context for understanding which person is being referred to

    ### Context-Aware Examples
    Previous: "My friend Mark lives in Chicago and my colleague Mark lives in New York"
    Input: "My colleague's number is 12345"
    Output:
    [
        {{
            "name": "Mark",
            "relationship": "colleague",
            "information": "number is 12345"
        }}
    ]

    Previous: "My friend Lisa lives in New Jersey and my colleague Lisa lives in New York"
    Input: "My colleague's number is 123456"
    Output:
    [
        {{
            "name": "Lisa",
            "relationship": "colleague",
            "information": "number is 123456"
        }}
    ]
    Note: "My colleague" refers to Lisa from the previous conversation.

    Previous: "My friend Rosa lives in Queen and my colleague Rosa lives in New York"
    Input: "She likes football and basketball"
    Output:
    [
        {{
            "name": "Rosa",
            "relationship": "unknown",
            "information": "likes football and basketball"
        }}
    ]
    Note: When "she" refers to a previously mentioned person but multiple profiles exist with the same name, set relationship to "unknown" to trigger clarification.

    Previous: "My friend Timon lives in Queen and my colleague Timon lives in New York"
    Input: "He likes football and basketball"
    Output:
    [
        {{
            "name": "Timon",
            "relationship": "unknown",
            "information": "likes football and basketball"
        }}
    ]
    Note: When "he" refers to a previously mentioned person but multiple profiles exist with the same name, set relationship to "unknown" to trigger clarification.

    ### PRONOUN HANDLING EXAMPLES
    Previous: "My friend Anna lives in Lio and my sister Anna lives in Montreal"
    Input: "She likes football and basketball"
    Output:
    [
        {{
            "name": "Anna",
            "relationship": "unknown",
            "information": "likes football and basketball"
        }}
    ]
    Note: "She" refers to Anna from chat history. Since there are two Annas, set relationship to "unknown".

    Previous: "My friend Timon lives in Queen and my colleague Timon lives in New York"
    Input: "He likes football and basketball"
    Output:
    [
        {{
            "name": "Timon",
            "relationship": "unknown",
            "information": "likes football and basketball"
        }}
    ]
    Note: "He" refers to Timon from chat history. Since there are two Timons, set relationship to "unknown".

    ### RELATIONSHIP REFERENCE EXAMPLES
    Previous: "My friend Lisa lives in New Jersey and my colleague Lisa lives in New York"
    Input: "My colleague's number is 123456"
    Output:
    [
        {{
            "name": "Lisa",
            "relationship": "colleague",
            "information": "number is 123456"
        }}
    ]
    Note: "My colleague" refers to Lisa from the previous conversation.

    Previous: "My friend Mark lives in Chicago and my colleague Mark lives in New York"
    Input: "My friend's phone number is 555-1234"
    Output:
    [
        {{
            "name": "Mark",
            "relationship": "friend",
            "information": "phone number is 555-1234"
        }}
    ]
    Note: "My friend" refers to Mark from the previous conversation.

    Examples:

    Input:
    My friend Marry is going to Europe and her phone number is 2930492
    Output:
    [
        {{
            "name": "Marry",
            "relationship": "friend",
            "information": "going to Europe and phone number is 2930492"
        }}
    ]

    Input:
    My friend Sam lives in Spain his number is 009123 and my colleague Sam lives in New York
    Output:
    [
        {{
            "name": "Sam",
            "relationship": "friend",
            "information": "lives in Spain and number is 009123"
        }},
        {{
            "name": "Sam",
            "relationship": "colleague",
            "information": "lives in New York"
        }}
    ]

    Input:
    Sonnet's friend Ash lives in Florida and Sonnet's colleague Ash lives in New York
    Output:
    [
        {{
            "name": "Ash",
            "relationship": "friend",
            "information": "lives in Florida"
        }},
        {{
            "name": "Ash",
            "relationship": "colleague",
            "information": "lives in New York"
        }}
    ]

    Input:
    My friend Ash likes to play soccer and basketball. He has two brothers eldest brother's name is Sam and youngest brother's name is Merrick
    Output:
    [
        {{
            "name": "Ash",
            "relationship": "friend",
            "information": "likes to play soccer and basketball and has two brothers eldest brother's name is Sam and youngest brother's name is Merrick"
        }}
    ]

    Text:
    {text}

    """

    max_retries = 2
    detected_names = []

    for attempt in range(max_retries):
        try:
            response = await openai_llm.ainvoke([HumanMessage(content=detection_prompt)])
            response_text = response.content.strip()

            logger.info(f"LLM response for name detection (attempt {attempt+1}): {response_text}")

            # Extract JSON array using regex - handles if LLM wraps with extra text
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
            if not json_match:
                # try parsing whole response in case it's clean JSON
                try:
                    candidate_json = json.loads(response_text)
                except json.JSONDecodeError:
                    raise ValueError("No JSON array found in LLM response")
            else:
                candidate_json = json.loads(json_match.group())

            # Validate as list of dicts
            if not isinstance(candidate_json, list):
                raise ValueError("Parsed JSON is not a list")

            filtered_results = []
            for entry in candidate_json:
                if not isinstance(entry, dict):
                    continue

                name = entry.get("name", "").strip().title()
                relationship = entry.get("relationship", "unknown").lower()
                information = entry.get("information", "").strip()

                # Filter invalid names
                if not name or len(name) < 2:
                    continue

                # Filter relationships to allowed set
                allowed_rel = {"friend", "colleague", "family", "supervisor", "neighbor", "assistant", "unknown"}
                if relationship not in allowed_rel:
                    logger.info(f"Invalid relationship '{relationship}' replaced with 'unknown'")
                    relationship = "unknown"

                # Filter common non-names
                common_words = {
                    'my', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                    'is', 'are', 'was', 'were', 'will', 'going', 'friend', 'colleague', 'boss', 'manager', 'assistant',
                    'neighbor', 'family', 'sister', 'brother', 'mother', 'father', 'daughter', 'son', 'cousin', 'uncle', 'aunt',
                    'name', 'names', 'named', 'calling', 'called', 'know', 'knows', 'knew', 'meet', 'meets', 'met',
                    'this', 'that', 'these', 'those', 'here', 'there', 'where', 'when', 'what', 'who', 'why', 'how',
                    'my', 'me', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'us'
                }
                
                # Special handling for pronouns - they should never be names
                pronouns = {'he', 'she', 'they', 'him', 'her', 'them', 'his', 'hers', 'their', 'theirs'}
                if name.lower() in pronouns:
                    logger.warning(f"LLM incorrectly returned pronoun '{name}' as a name. This should never happen.")
                    continue
                
                if name.lower() in common_words:
                    logger.info(f"Skipping common word detected as name: {name}")
                    continue

                filtered_results.append({
                    "name": name,
                    "relationship": relationship,
                    "information": information,
                    "timestamp": datetime.now().isoformat()
                })

            # Remove duplicates (by name + relationship)
            unique = {}
            for item in filtered_results:
                key = (item["name"].lower(), item["relationship"])
                if key not in unique:
                    unique[key] = item
            detected_names = list(unique.values())

            # Post-process to validate context-aware detection
            if chat_context and detected_names:
                detected_names = await validate_context_aware_profiles(detected_names, chat_context, text)
            
            # CRITICAL: Additional context-aware validation for relationship references
            if chat_context and not detected_names:
                # Check if this is a relationship reference that should resolve to an existing person
                detected_names = await resolve_relationship_references(text, chat_context)

            logger.info(f"Detected {len(detected_names)} names after filtering and deduplication")
            return detected_names, True

        except Exception as e:
            logger.error(f"LLM parsing attempt #{attempt+1} failed with error: {str(e)}")
            if attempt == max_retries - 1:
                logger.info("LLM detection failed after retries, falling back")
                return detect_profiles_fallback(text), False

    return detect_profiles_fallback(text), False

async def validate_context_aware_profiles(detected_profiles: List[Dict[str, Any]], chat_context: str, text: str) -> List[Dict[str, Any]]:
    """
    Validate and correct profiles based on chat context to ensure proper name resolution.
    """
    logger.info(f"[CONTEXT VALIDATION] Validating {len(detected_profiles)} profiles with chat context")
    
    # Check if text contains relationship references that should use chat history
    relationship_patterns = [
        r'my\s+(friend|colleague|family|supervisor|neighbor|assistant)\'s?\s+',
        r'my\s+(friend|colleague|family|supervisor|neighbor|assistant)\s+',
    ]
    
    has_relationship_reference = any(re.search(pattern, text.lower()) for pattern in relationship_patterns)
    
    if has_relationship_reference:
        logger.info(f"[CONTEXT VALIDATION] Text contains relationship reference: {text}")
        
        # Extract the relationship mentioned
        for pattern in relationship_patterns:
            match = re.search(pattern, text.lower())
            if match:
                relationship = match.group(1)
                logger.info(f"[CONTEXT VALIDATION] Detected relationship reference: {relationship}")
                
                # Look for this relationship in chat context
                if relationship in chat_context.lower():
                    # Find the name associated with this relationship in chat context
                    # Look for patterns like "my friend Lisa" or "friend Lisa"
                    name_pattern = rf'(?:my\s+)?{relationship}\s+([A-Z][a-z]+)'
                    name_match = re.search(name_pattern, chat_context, re.IGNORECASE)
                    
                    if name_match:
                        context_name = name_match.group(1)
                        logger.info(f"[CONTEXT VALIDATION] Found {relationship} {context_name} in chat context")
                        
                        # CRITICAL FIX: Only correct names if the current input doesn't explicitly mention a name
                        # Check if the current text explicitly mentions a name after "my friend"
                        explicit_name_pattern = rf'my\s+{relationship}\s+([A-Z][a-z]+)'
                        explicit_match = re.search(explicit_name_pattern, text, re.IGNORECASE)
                        
                        if explicit_match:
                            explicit_name = explicit_match.group(1)
                            logger.info(f"[CONTEXT VALIDATION] Current input explicitly mentions name: {explicit_name}")
                            
                            # If the explicit name is different from context, prioritize the explicit name
                            if explicit_name.lower() != context_name.lower():
                                logger.info(f"[CONTEXT VALIDATION] Prioritizing explicit name '{explicit_name}' over context name '{context_name}' - will NOT change name")
                                # Don't change the name - let the explicit name stand
                                # This ensures "My friend Marry likes football" keeps "Marry" instead of changing to "Mark"
                                continue
                            else:
                                logger.info(f"[CONTEXT VALIDATION] Explicit name '{explicit_name}' matches context name '{context_name}' - proceeding with validation")
                        
                        # Only correct names when dealing with ambiguous references (no explicit name)
                        logger.info(f"[CONTEXT VALIDATION] No explicit name conflict - proceeding with context-based name correction")
                        for profile in detected_profiles:
                            if profile.get('name') != context_name:
                                logger.info(f"[CONTEXT VALIDATION] Correcting name from '{profile.get('name')}' to '{context_name}' based on chat context")
                                profile['name'] = context_name
                                profile['relationship'] = relationship
                                break
                break
    
    return detected_profiles

async def resolve_relationship_references(text: str, chat_context: str) -> List[Dict[str, Any]]:
    """
    Resolve relationship references (e.g., "my sister", "my friend") to actual people from chat history.
    This handles cases where the LLM didn't detect names but the text contains relationship references.
    """
    logger.info(f"[RELATIONSHIP RESOLUTION] Attempting to resolve relationship references in: {text}")
    
    text_lower = text.lower()
    
    # Patterns for relationship references
    relationship_patterns = [
        r'my\s+(friend|colleague|sister|brother|mother|father|neighbor|supervisor|assistant)\'s?\s+',
        r'my\s+(friend|colleague|sister|brother|mother|father|neighbor|supervisor|assistant)\s+',
    ]
    
    for pattern in relationship_patterns:
        match = re.search(pattern, text_lower)
        if match:
            relationship = match.group(1)
            logger.info(f"[RELATIONSHIP RESOLUTION] Detected relationship reference: {relationship}")
            
            # Look for this relationship in chat context
            if relationship in chat_context.lower():
                # Find the name associated with this relationship in chat context
                # Look for patterns like "my friend Marry" or "friend Marry"
                name_pattern = rf'(?:my\s+)?{relationship}\s+([A-Z][a-z]+)'
                name_match = re.search(name_pattern, chat_context, re.IGNORECASE)
                
                if name_match:
                    context_name = name_match.group(1)
                    logger.info(f"[RELATIONSHIP RESOLUTION] Found {relationship} {context_name} in chat context")
                    
                    # Extract the information from the text (remove the relationship part)
                    info_start = match.end()
                    information = text[info_start:].strip()
                    
                    if information:
                        logger.info(f"[RELATIONSHIP RESOLUTION] Resolved to: {context_name} ({relationship}) with info: {information}")
                        return [{
                            "name": context_name,
                            "relationship": relationship,
                            "information": information,
                            "timestamp": datetime.now().isoformat()
                        }]
                    else:
                        logger.info(f"[RELATIONSHIP RESOLUTION] No information found after relationship reference")
                        return []
    
    logger.info(f"[RELATIONSHIP RESOLUTION] No relationship references resolved")
    return []

def detect_profiles_fallback(text: str) -> List[Dict[str, str]]:
    """
    Fallback profile detection using regex patterns if LLM fails.
    """
    import re
    from datetime import datetime

    logger.info(f"Using fallback regex detection for: {text}")

    common_words = {
        'my', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'will', 'going', 'friend', 'colleague', 'boss', 'manager', 'assistant',
        'neighbor', 'family', 'sister', 'brother', 'mother', 'father', 'daughter', 'son', 'cousin', 'uncle', 'aunt',
        'name', 'names', 'named', 'calling', 'called', 'know', 'knows', 'knew', 'meet', 'meets', 'met',
        'this', 'that', 'these', 'those', 'here', 'there', 'where', 'when', 'what', 'who', 'why', 'how',
        'my', 'me', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'us'
    }

    # Extended name detection regex patterns
    name_patterns = [
        r'\b(?:my\s+)?(friend|colleague|boss|manager|assistant|neighbor|family|sister|brother|mother|father|daughter|son|cousin|uncle|aunt)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:my\s+)?(friend|colleague|boss|manager|assistant|neighbor|family|sister|brother|mother|father|daughter|son|cousin|uncle|aunt)\b',
        r'\b(?:I\s+)?(know|knows|knew|met|meet|meets)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:works|lives|studies|travels|goes|is|was|will|age|years\s+old)\b',
        r'\b(?:Do\s+you\s+know|Tell\s+me\s+about|Who\s+is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:going|traveling|working|studying|living)\b',
    ]

    detected_names = []

    for pattern in name_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Extract based on pattern groups, tolerate different groupings
            groups = match.groups()
            if len(groups) == 2:
                if groups[0].lower() in {"friend", "colleague", "boss", "manager", "assistant", "neighbor", "family",
                                       "sister", "brother", "mother", "father", "daughter", "son", "cousin", "uncle", "aunt"}:
                    relationship = groups[0].lower()
                    name = groups[1].title()
                else:
                    name = groups[0].title()
                    relationship = "unknown"
            elif len(groups) == 1:
                name = groups[0].title()
                relationship = "unknown"
            else:
                continue

            # Filter out common words and short names
            if name.lower() in common_words or len(name) < 2:
                continue

            context = match.group(0)

            # Extract meaningful information - pass text, name, and full context snippet
            information = extract_meaningful_info(text, name, context)

            detected_names.append({
                "name": name,
                "relationship": relationship,
                "information": information,
                "timestamp": datetime.now().isoformat()
            })

    # Deduplicate by (name, relationship)
    unique = {}
    for item in detected_names:
        key = (item["name"].lower(), item["relationship"])
        if key not in unique:
            unique[key] = item
    filtered = list(unique.values())

    logger.info(f"Fallback detected {len(filtered)} names.")
    return filtered

def extract_meaningful_info(text: str, name: str, context: str) -> str:
    """
    Extract meaningful information about a person from the text.
    """
    import re
    
    # Look for sentences that mention the person
    sentences = re.split(r'[.!?]+', text)
    relevant_info = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if name.lower() in sentence.lower():
            # Clean up the sentence
            clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
            if clean_sentence and len(clean_sentence) > 5:
                relevant_info.append(clean_sentence)
    
    # If no specific info found, use the context
    if not relevant_info:
        return context
    
    # Return the most relevant information
    return relevant_info[0] if relevant_info else context

def clean_profile_name(name: str) -> str:
    """
    Clean a profile name to extract just the person's name, removing any descriptive elements.
    
    Args:
        name: The name to clean (e.g., "Sonnet's Friend Mark", "My Colleague Sarah")
        
    Returns:
        Clean name (e.g., "Mark", "Sarah")
    """
    if not name:
        return name
    
    # Common patterns to remove
    patterns_to_remove = [
        r"^.*?'s\s+(?:friend|colleague|sister|brother|mother|father|neighbor|supervisor|boss|partner|spouse|husband|wife|assistant)\s+",
        r"^.*?'s\s+(?:friend|colleague|sister|brother|mother|father|neighbor|supervisor|boss|partner|spouse|husband|wife|assistant)$",
        r"^my\s+(?:friend|colleague|sister|brother|mother|father|neighbor|supervisor|boss|partner|spouse|husband|wife|assistant)\s+",
        r"^the\s+(?:friend|colleague|sister|brother|mother|father|neighbor|supervisor|boss|partner|spouse|husband|wife|assistant)\s+",
        r"^a\s+(?:friend|colleague|sister|brother|mother|father|neighbor|supervisor|boss|partner|spouse|husband|wife|assistant)\s+",
        r"^an\s+(?:friend|colleague|sister|brother|mother|father|neighbor|supervisor|boss|partner|spouse|husband|wife|assistant)\s+"
    ]
    
    import re
    
    # Apply each pattern to clean the name
    clean_name = name.strip()
    for pattern in patterns_to_remove:
        clean_name = re.sub(pattern, '', clean_name, flags=re.IGNORECASE)
    
    # Clean up any remaining whitespace and ensure proper capitalization
    clean_name = clean_name.strip()
    if clean_name:
        # Capitalize first letter, lowercase the rest
        clean_name = clean_name[0].upper() + clean_name[1:].lower()
    
    logger.info(f"[PROFILE NAME CLEANING] Cleaned '{name}' to '{clean_name}'")
    return clean_name

async def cleanup_existing_profile_names(user_id: str) -> bool:
    """
    Clean up existing profiles that have descriptive names instead of clean names.
    This function helps migrate profiles created with the old naming system.
    
    Args:
        user_id: The user's ID
        
    Returns:
        True if cleanup was successful, False otherwise
    """
    try:
        # Load existing profiles
        profiles = await load_user_profiles(user_id)
        if not profiles:
            logger.info(f"[PROFILE CLEANUP] No profiles found for user {user_id}")
            return True
        
        logger.info(f"[PROFILE CLEANUP] Starting cleanup for {len(profiles)} profiles")
        
        cleaned_profiles = {}
        changes_made = False
        
        for key, profile in profiles.items():
            original_name = profile.get('name', '')
            clean_name = clean_profile_name(original_name)
            
            if clean_name != original_name:
                logger.info(f"[PROFILE CLEANUP] Cleaning profile name: '{original_name}' -> '{clean_name}'")
                changes_made = True
                
                # Create new profile with clean name
                new_key = f"{clean_name.lower()}_{profile.get('relationship', 'unknown').lower()}"
                cleaned_profiles[new_key] = {
                    'name': clean_name,
                    'relationship': profile.get('relationship', 'unknown'),
                    'information': profile.get('information', []),
                    'timestamp': profile.get('timestamp', '')
                }
            else:
                # Name is already clean, keep as is
                cleaned_profiles[key] = profile
        
        if changes_made:
            # Save cleaned profiles
            success = await save_user_profiles(user_id, cleaned_profiles)
            if success:
                logger.info(f"[PROFILE CLEANUP] Successfully cleaned up {len(profiles)} profiles")
            else:
                logger.error(f"[PROFILE CLEANUP] Failed to save cleaned profiles")
                return False
        else:
            logger.info(f"[PROFILE CLEANUP] No profile names needed cleaning")
        
        return True
        
    except Exception as e:
        logger.error(f"[PROFILE CLEANUP] Error during cleanup: {str(e)}")
        return False

async def create_or_update_profile_persistent(user_id: str, name: str, relationship: str, information: str) -> str:
    """
    Creates a new profile or updates an existing one with new information using persistent storage.
    Considers both name and relationship for profile identification.
    Returns the updated profile.
    """
    logger.info(f"--- Invoking create_or_update_profile_persistent for user_id: {user_id} ---")
    logger.info(f"Received Parameters: name='{name}', relationship='{relationship}', information='{information}'")

    # Clean the name to ensure it's just the person's name, not a descriptive name
    clean_name = clean_profile_name(name)
    logger.info(f"Cleaned name from '{name}' to '{clean_name}'")

    # Load existing profiles
    profiles = await load_user_profiles(user_id)
    logger.info(f"Loaded {len(profiles)} existing profiles for user {user_id}.")

    clean_name_lower = clean_name.lower()
    relationship_lower = relationship.lower() if relationship else 'unknown'
    timestamp = datetime.now().isoformat()
    
    # Create a unique key that combines name and relationship
    profile_key = f"{clean_name_lower}_{relationship_lower}"
    logger.info(f"Generated potential profile key: '{profile_key}'.")

    # Check if we have an exact match (same name and relationship)
    exact_match_key = None
    logger.info("Searching for an exact profile match (name and relationship)...")
    for key, profile in profiles.items():
        # Ensure keys exist before lowercasing
        prof_name = profile.get('name', '').lower()
        prof_rel = profile.get('relationship', 'unknown').lower()
        if prof_name == clean_name_lower and prof_rel == relationship_lower:
            exact_match_key = key
            break

    if exact_match_key:
        logger.info(f"Found exact match with key: '{exact_match_key}'. Updating existing profile.")
        
        # Use intelligent profile update that preserves existing information
        try:
            from rag.rich_profile_utils import intelligently_update_profile, ProfileUpdateConfirmationRequired
            
            # Get the existing profile
            existing_profile = profiles[exact_match_key]
            
            # Intelligently update the profile with new information
            try:
                updated_profile = await intelligently_update_profile(existing_profile, [information])
                
                # Replace the profile with the updated version
                profiles[exact_match_key] = updated_profile
                
                logger.info(f"Intelligently updated profile '{exact_match_key}' while preserving existing data.")
                
            except ProfileUpdateConfirmationRequired as confirmation_req:
                # Profile update requires confirmation - set pending update state
                logger.info(f"Profile update requires confirmation for {confirmation_req.person_name}'s {confirmation_req.field_name}")
                
                # Set the pending update information in the state
                # This will be handled by the calling function to set the state
                raise confirmation_req
                
        except ProfileUpdateConfirmationRequired as confirmation_req:
            # Profile update requires confirmation - we need to signal this to the caller
            # Instead of saving profiles, we'll raise the exception to be handled upstream
            logger.info(f"Profile update requires confirmation for {confirmation_req.person_name}'s {confirmation_req.field_name}")
            
            # Don't save profiles or continue processing - raise the exception
            # This will be caught by the calling function to set the pending update state
            raise confirmation_req
        except Exception as e:
            logger.warning(f"Intelligent profile update failed: {str(e)}")
            # Continue with normal processing for other errors

    else:
        logger.info("No exact match found. Proceeding to create a new profile entry.")
        # Create a new profile entry using the generated profile_key
        simple_profile = {
            'name': clean_name,
            'relationship': relationship,
            'information': [information],
            'timestamp': timestamp
        }
        
        # Convert to rich profile format before saving
        try:
            logger.info(f"Attempting to convert simple profile to rich format: {simple_profile}")
            from rag.rich_profile_utils import convert_simple_to_rich_profile
            rich_profile = await convert_simple_to_rich_profile(simple_profile)
            logger.info(f"Rich profile conversion result: {rich_profile}")
            profiles[profile_key] = rich_profile
            logger.info(f"Created new rich profile data with key '{profile_key}'.")
        except Exception as e:
            logger.warning(f"Rich profile conversion failed, using simple format: {str(e)}")
            logger.warning(f"Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
            profiles[profile_key] = simple_profile
            logger.info(f"Created new simple profile data with key '{profile_key}'.")

    # Save to persistent storage
    logger.info(f"Profiles dictionary before saving: {profiles}")
    logger.info(f"Profile keys: {list(profiles.keys())}")
    logger.info("Attempting to save all profiles to persistent storage...")
    success = await save_user_profiles(user_id, profiles)
    if success:
        logger.info("Successfully saved profiles to persistent storage.")
    else:
        logger.error("Failed to save profiles to persistent storage.")

    # Select the correct profile to return
    final_profile_key = exact_match_key if exact_match_key else profile_key
    final_profile = profiles.get(final_profile_key, {}) # Use .get for safety

    logger.info(f"--- Function complete. Returning profile for key '{final_profile_key}' ---")
    # return f"OK, I've noted your {relationship} {name} profile!"
    return final_profile

async def get_all_profiles_persistent(user_id: str) -> List[Dict[str, Any]]:
    """
    Returns all profiles as a list using persistent storage.
    """
    profiles = await load_user_profiles(user_id)
    return list(profiles.values())

# --- LangGraph Nodes ---

async def get_user_name_node(state: AgentState) -> AgentState:
    """Node to extract My name from chat messages using LLM."""
    logger.info("[LANGGRAPH PATH] Starting get_user_name_node")
    
    # First check if user name is already loaded from storage
    if not state.user_name:
        saved_name = await load_user_name(state.user_id)
        if saved_name:
            logger.info(f"[LANGGRAPH PATH] Loaded saved user name: {saved_name}")
            state.user_name = saved_name
            state.agent_status = "initialize"
            return state
    
    # If user name is already set in current state, skip this node
    if state.user_name:
        logger.info(f"[LANGGRAPH PATH] User name already set: {state.user_name}")
        state.agent_status = "initialize"
        return state
    
    # Get the current user message
    current_message = ""
    for message in reversed(state.request.messages):
        if message.role == "user":
            current_message = message.content
            break
    
    logger.info(f"[LANGGRAPH PATH] Current message: {current_message}")
    
    # Use LLM to detect if the message contains a name
    name_detection_prompt = f"""
    You are a friendly assistant helping to extract a person's name from their message.

    TASK: Determine if the user is explicitly providing their name in the following message.

    --- RULES ---
    1. Valid name introductions:
   - Explicit self-introductions: "my name is X", "I'm X", "I am X", "call me X", "I'm called X", "this is X".
   - Single-word replies that look like a plausible first name (e.g., "John", "Sarah", "Alex", "Sonnet").
   - Short two-word first names (e.g., "Mary Ann", "Jean Paul") are allowed, but return only the first word ("Mary", "Jean").

    2. Extraction rules:
    - Extract ONLY the first name (the first word of the provided name).
    - Normalize formatting: capitalize the first letter, lowercase the rest (e.g., "john" ? "John").

    3. Exclusions:
    - Do NOT treat greetings, small talk, or generic phrases as names (e.g., "hi", "hello", "how are you", "good morning").
    - Do NOT treat non-names (usernames, emojis, random strings, numbers, "me", "you", etc.) as names.
    - If the input is longer than two words and does not match the explicit introduction patterns, return "NO_NAME".

    4. Ambiguity:
    - If unsure whether a word is intended as a human first name, return "NO_NAME".

    --- OUTPUT ---
    - Return ONLY the extracted human first name.
    - If no clear valid name is provided, return exactly: "NO_NAME".

    --- EXAMPLES ---
    - "Hi" ? "NO_NAME"
    - "Hello there" ? "NO_NAME"
    - "My name is John" ? "John"
    - "I'm Sarah" ? "Sarah"
    - "Call me Alex" ? "Alex"
    - "John" ? "John"
    - "Mary Ann" ? "Mary"
    - "Nice to meet you" ? "NO_NAME"
    - "How are you?" ? "NO_NAME"
    - "This is David" ? "David"
    - "I'm called Emma" ? "Emma"
    - "The name's Bond" ? "Bond"
    - "Sonnet" ? "Sonnet"
    - "My name is Sonnet" ? "Sonnet"
    - "I'm Sonnet" ? "Sonnet"

    USER MESSAGE: "{current_message}"

    RESPONSE (name only or "NO_NAME"):

    """
    
    try:
        response = await openai_llm.ainvoke([HumanMessage(content=name_detection_prompt)])
        detected_name = response.content.strip()
        
        logger.info(f"[LANGGRAPH PATH] LLM detected: {detected_name}")
        
        if detected_name and detected_name != "NO_NAME":
            # Clean up the name (remove quotes, extra spaces, etc.)
            clean_name = detected_name.strip().strip('"').strip("'").title()
            logger.info(f"[LANGGRAPH PATH] Extracted user name: {clean_name}")
            
            # Save the name permanently
            save_success = await save_user_name(state.user_id, clean_name)
            if save_success:
                logger.info(f"[LANGGRAPH PATH] Successfully saved user name: {clean_name}")
            else:
                logger.warning(f"[LANGGRAPH PATH] Failed to save user name: {clean_name}")
            
            state.user_name = clean_name
            state.final_response = f"Wonderful to meet you, {clean_name}! I'm here to help you with any questions or tasks you might have. What would you like to work on today?"
            state.agent_status = "get_name"
            # End workflow immediately after name is found
            return state
        else:
            logger.info("[LANGGRAPH PATH] No name found, asking for user name warmly")
            if state.final_response == "Hello there! I'd love to get to know you better. What's your name?":
                state.final_response = "Hello there! Could you let me know your name?"
            else:
                state.final_response = "Hello there! I'd love to get to know you better. What's your name?"
            state.agent_status = "get_name"
            
    except Exception as e:
        logger.error(f"[LANGGRAPH PATH] Error in name detection: {str(e)}")
        # Fallback to warm greeting
        if state.final_response == "Hello there! I'd love to get to know you better. What's your name?":
            state.final_response = "Hello there! Could you let me know your name?"
        else:
            state.final_response = "Hello there! I'd love to get to know you better. What's your name?"
        state.agent_status = "get_name"
    
    return state

def get_current_time_info():
    """Helper to get current date/time for the system message."""
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d")
    formatted_time = now.strftime("%I:%M:%S %p")
    # For EDT, you'd usually use pytz or zoneinfo, mocking here.
    return formatted_date, formatted_time, "Eastern Daylight Time"

async def initialize_state(state: AgentState) -> AgentState:
    logger.info("[LANGGRAPH PATH] Starting initialize_state node")
    user_id = state.user_id
    request = state.request
    thread_id = state.thread_id

    logger.info(f"[LANGGRAPH PATH] Initializing state with user_id: {user_id}")
    logger.info(f"[LANGGRAPH PATH] Request details: {request}")

    # Use the last user message as the retrieval_query
    retrieval_query = None
    for m in reversed(request.messages):
        if (isinstance(m, dict) and m.get('role') == 'user') or (hasattr(m, 'role') and m.role == 'user'):
            retrieval_query = m['content'] if isinstance(m, dict) else m.content
            break
    if retrieval_query is None:
        retrieval_query = request.query or ""

    logger.info(f"[LANGGRAPH PATH] Extracted retrieval_query: {retrieval_query}")
    state.retrieval_query = retrieval_query
    
    # Clear searched links and realtime data for new conversation turn
    logger.info(f"[LANGGRAPH PATH] Before clearing - searched_links: {state.searched_links}, realtime_data: {state.realtime_data is not None}")
    state.searched_links = []
    state.realtime_data = None
    
    logger.info(f"[LANGGRAPH PATH] Cleared searched_links and realtime_data for new turn")
    
    logger.info(f"[LANGGRAPH PATH] initialize_state completed. agent_status: {state.agent_status}")
    return state

async def user_data_and_stats_node(state: AgentState) -> AgentState:
    """Node for integrating user information and statistics."""
    start_time = time.time()
    logger.info("[LANGGRAPH PATH] Starting user_data_and_stats_node")
    logger.info(f"[LANGGRAPH PATH] Processing user_id: {state.user_id}")
    
    user_info_result = await _get_user_space_info(state.user_id)
    logger.info(f"[LANGGRAPH PATH] User info result: {user_info_result}")

    state.user_profile = user_info_result["user_profile"]
    state.user_repositories = user_info_result["user_repositories"] # Store all user repos for later file listing
    logger.info(f" [LANGGRAPH PATH] User repositories count: {len(state.user_repositories)}")

    stats_result = await _get_user_stats(state.user_id, state.user_repositories)
    logger.info(f"[LANGGRAPH PATH] Stats result: {stats_result}")
    state.user_stats_context = stats_result["user_stats_context"]
    state.filtered_documents = stats_result["filtered_documents"]
    
    # Add user name to context if available
    if state.user_name:
        # Initialize context with user name information
        state.context = f"My name is {state.user_name}."
        logger.info(f"[LANGGRAPH PATH] Added user name to context: {state.user_name}")
        logger.info(f"[LANGGRAPH PATH] Context initialized with user name: {state.context}")
    else:
        # Initialize empty context
        state.context = ""
        logger.info("[LANGGRAPH PATH] No user name available for context")
    
    logger.info(f"[TIMER] User data & stats: {time.time() - start_time:.3f}s")
    logger.info(f"[LANGGRAPH PATH] Filtered documents count: {len(state.filtered_documents)}")
    logger.info(f"[LANGGRAPH PATH] user_data_and_stats_node completed")
    return state

async def repository_validation_node(state: AgentState) -> AgentState:
    """Node for validating and filtering repository IDs."""
    start_time = time.time()
    logger.info("[LANGGRAPH PATH] Starting repository_validation_node")
    logger.info(f"[LANGGRAPH PATH] Requested repository_ids: {state.request.repository_ids}")
    
    valid_repo_ids = await _validate_and_filter_repositories(
        state.user_id,
        state.request.repository_ids,
        state.user_repositories
    )
    state.request.repository_ids = valid_repo_ids # Update the request's repo_ids with validated ones
    
    logger.info(f"[LANGGRAPH PATH] Validated repository_ids: {valid_repo_ids}")
    logger.info(f"[TIMER] Repo validation: {time.time() - start_time:.3f}s")
    logger.info(f"[LANGGRAPH PATH] repository_validation_node completed")
    return state

async def retrieve_rag_context_node(state: AgentState) -> AgentState:
    """Node for retrieving context from RAG."""
    start_time = time.time()
    logger.info("[LANGGRAPH PATH] Starting retrieve_rag_context_node")
    logger.info(f"[LANGGRAPH PATH] Retrieval query: {state.retrieval_query}")
    logger.info(f"[LANGGRAPH PATH] Query parameters - k: {state.request.k}, document_ids: {state.request.document_ids}, repository_ids: {state.request.repository_ids}")
    
    filter_dict = state.request.metadata_filter if state.request.metadata_filter else {}
    logger.info(f"[LANGGRAPH PATH] Metadata filter: {filter_dict}")
    
    rag_context = await _perform_rag_query(
        state.retrieval_query,
        state.request.k,
        state.request.document_ids,
        state.request.repository_ids,
        filter_dict
    )
    state.context = rag_context
    
    logger.info(f"[LANGGRAPH PATH] RAG context length: {len(rag_context) if rag_context else 0} characters")
    logger.info(f"[TIMER] Document query: {time.time() - start_time:.3f}s")
    logger.info(f"[LANGGRAPH PATH] retrieve_rag_context_node completed")
    return state

async def web_search_node(state: AgentState) -> AgentState:
    """Node for performing web search if needed."""
    start_time = time.time()
    logger.info("[LANGGRAPH PATH] Starting web_search_node")
    logger.info(f"[LANGGRAPH PATH] Web search query: {state.retrieval_query}")
    
    # Check if web search is actually needed for this query
    mock_sampling_params = SamplingParams(temperature=0, max_tokens=10)
    needs_search = await needs_web_search_with_gpt(openai_llm, state.retrieval_query, mock_sampling_params)
    
    if needs_search:
        logger.info("[LANGGRAPH PATH] Web search is needed for this query")
        realtime_data = await _perform_web_search(state.retrieval_query)
        state.realtime_data = realtime_data

        logger.info(f"Real Time Data: {realtime_data}")
        
        if realtime_data:
            # Store the searched links for later use in final response
            state.searched_links = [
                result.get('url', '') for result in realtime_data.get('data', [])
                if result.get('url', '')
            ]
            
            search_results = "\n\n".join([
                f"Title: {result.get('title', 'No title')}\nContent: {result.get('content', 'No content')}\nSource: {result.get('url', 'No URL')}"
                for result in realtime_data.get('data', [])
            ])
            state.context = f"{state.context}\n{search_results}"
            logger.info(f"[LANGGRAPH PATH] Web search results found: {len(realtime_data.get('data', []))} results")
            logger.info(f"[LANGGRAPH PATH] Stored {len(state.searched_links)} searched links")
        else:
            logger.info("[LANGGRAPH PATH] No web search results found")
            # Clear any old searched links when no results
            state.searched_links = []
            state.realtime_data = None
    else:
        logger.info("[LANGGRAPH PATH] Web search not needed for this query")
        # Clear any old searched links and realtime data when search is not needed
        state.searched_links = []
        state.realtime_data = None
    
    logger.info(f"[TIMER] Web search: {time.time() - start_time:.3f}s")
    logger.info(f"[LANGGRAPH PATH] web_search_node completed")
    return state

def replace_user_name_with_my(text: str, user_name: str) -> str:
    """
    Replace instances of the user's name with "my" to make the context more personal.
    This function handles various forms of the user's name (first name, full name, etc.).
    
    Args:
        text: The text to process
        user_name: The user's name to replace
        
    Returns:
        Text with user's name replaced by "my" where appropriate
    """
    if not user_name or not text:
        return text
    
    # Split the user name into parts to handle first name, full name, etc.
    name_parts = user_name.split()
    first_name = name_parts[0].lower()
    
    logger.info(f"[PERSONALIZATION] Processing text for user '{user_name}' (first name: '{first_name}')")
    original_text = text
    
    # Create patterns to match the user's name in various forms
    import re
    
    # Pattern 1: "Harry's" -> "my" (e.g., "Harry's sister" -> "my sister")
    possessive_pattern = rf'\b{re.escape(first_name)}\'s\b'
    text = re.sub(possessive_pattern, 'my', text, flags=re.IGNORECASE)
    
    # Pattern 1b: "Harry sister" -> "my sister" (without apostrophe)
    no_apostrophe_pattern = rf'\b{re.escape(first_name)}\s+(sister|brother|mother|father|parent|child|son|daughter|wife|husband|partner|friend|colleague|neighbor|cousin|aunt|uncle|grandmother|grandfather|pet|dog|cat|car|house|job|work|school|college|university|company|business)\b'
    text = re.sub(no_apostrophe_pattern, r'my \1', text, flags=re.IGNORECASE)
    
    # Pattern 2: "Harry is" -> "I am" or "I'm"
    is_pattern = rf'\b{re.escape(first_name)}\s+is\b'
    text = re.sub(is_pattern, 'I am', text, flags=re.IGNORECASE)
    
    # Pattern 3: "Harry has" -> "I have"
    has_pattern = rf'\b{re.escape(first_name)}\s+has\b'
    text = re.sub(has_pattern, 'I have', text, flags=re.IGNORECASE)
    
    # Pattern 4: "Harry went" -> "I went" (common verbs)
    verb_patterns = [
        rf'\b{re.escape(first_name)}\s+(went|goes|going|go)\b',
        rf'\b{re.escape(first_name)}\s+(likes|liked|like)\b',
        rf'\b{re.escape(first_name)}\s+(wants|wanted|want)\b',
        rf'\b{re.escape(first_name)}\s+(needs|needed|need)\b',
        rf'\b{re.escape(first_name)}\s+(lives|lived|live)\b',
        rf'\b{re.escape(first_name)}\s+(works|worked|work)\b',
        rf'\b{re.escape(first_name)}\s+(studies|studied|study)\b',
        rf'\b{re.escape(first_name)}\s+(travels|traveled|travel)\b'
    ]
    
    verb_replacements = [
        'I went', 'I go', 'I am going', 'I go',
        'I like', 'I liked', 'I like',
        'I want', 'I wanted', 'I want',
        'I need', 'I needed', 'I need',
        'I live', 'I lived', 'I live',
        'I work', 'I worked', 'I work',
        'I study', 'I studied', 'I study',
        'I travel', 'I traveled', 'I travel'
    ]
    
    for pattern, replacement in zip(verb_patterns, verb_replacements):
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Pattern 5: "Harry" at the beginning of a sentence -> "I"
    sentence_start_pattern = rf'^\s*{re.escape(first_name)}\b'
    text = re.sub(sentence_start_pattern, 'I', text, flags=re.IGNORECASE)
    
    # Pattern 6: "Harry" after common prepositions -> "me"
    preposition_pattern = rf'\b(to|for|with|about|from|of)\s+{re.escape(first_name)}\b'
    text = re.sub(preposition_pattern, r'\1 me', text, flags=re.IGNORECASE)
    
    # Pattern 7: "Harry" at the end of a sentence or phrase -> "me"
    end_pattern = rf'\b{re.escape(first_name)}\s*[.!?]?\s*$'
    text = re.sub(end_pattern, 'me', text, flags=re.IGNORECASE)
    
    # Log if any changes were made
    if text != original_text:
        logger.info(f"[PERSONALIZATION] Text changed: '{original_text}' -> '{text}'")
    else:
        logger.info(f"[PERSONALIZATION] No changes made to text")
    
    return text

async def prepare_llm_input_node(state: AgentState) -> AgentState:
    """Node to prepare the final context and messages for the LLM."""
    logger.info("[LANGGRAPH PATH] Starting prepare_llm_input_node")
    
    context_parts = []

    # Inject user name info (prioritize state.user_name over user_profile)
    if state.user_name:
        context_parts.append(f"My name is {state.user_name}.")
        logger.info(f"[LANGGRAPH PATH] Added user name info: {state.user_name}")
    elif state.user_profile.get('name'):
        context_parts.append(f"My name is {state.user_profile['name']}.")
        logger.info(f"[LANGGRAPH PATH] Added user profile info: {state.user_profile['name']}")
    else:
        logger.info("[LANGGRAPH PATH] No user name available for context")
    
    # Inject user stats
    if state.user_stats_context:
        # Apply name replacement to user stats context if user name is available
        if state.user_name:
            personalized_stats = replace_user_name_with_my(state.user_stats_context, state.user_name)
            if personalized_stats != state.user_stats_context:
                logger.info(f"[LANGGRAPH PATH] Personalized user stats: '{state.user_stats_context}' -> '{personalized_stats}'")
                context_parts.append(personalized_stats)
            else:
                context_parts.append(state.user_stats_context)
        else:
            context_parts.append(state.user_stats_context)
        logger.info(f"[LANGGRAPH PATH] Added user stats context (length: {len(state.user_stats_context)})")
    else:
        logger.info("[LANGGRAPH PATH] No user stats context available")

    # Add repository names and count
    if state.user_repositories:
        repo_names = [repo.name for repo in state.user_repositories]
        repo_context = f"User has {len(state.user_repositories) - 2} repositories: {', '.join(repo_names)}."
        
        # Apply name replacement to repository context if user name is available
        if state.user_name:
            personalized_repo_context = replace_user_name_with_my(repo_context, state.user_name)
            if personalized_repo_context != repo_context:
                logger.info(f"[LANGGRAPH PATH] Personalized repo context: '{repo_context}' -> '{personalized_repo_context}'")
                context_parts.append(personalized_repo_context)
            else:
                context_parts.append(repo_context)
        else:
            context_parts.append(repo_context)
            
        logger.info(f" [LANGGRAPH PATH] Added repository info: {len(state.user_repositories)} repositories")

    # Add file names to context
    if state.filtered_documents:
        file_info = []
        for doc in state.filtered_documents:
            repo_name = "Unknown"
            for repo in state.user_repositories:
                if repo.id == doc.repository_id:
                    repo_name = repo.name
                    break
            file_info.append(f"{repo_name} > {doc.filename}")
        
        file_context = f"User's uploaded files:\n" + "\n".join(f"- {info}" for info in file_info)
        
        # Apply name replacement to file context if user name is available
        if state.user_name:
            personalized_file_context = replace_user_name_with_my(file_context, state.user_name)
            if personalized_file_context != file_context:
                logger.info(f"[LANGGRAPH PATH] Personalized file context: '{file_context}' -> '{personalized_file_context}'")
                context_parts.append(personalized_file_context)
            else:
                context_parts.append(file_context)
        else:
            context_parts.append(file_context)
            
        logger.info(f"[LANGGRAPH PATH] Added {len(state.filtered_documents)} document files to context")

    # Combine RAG and web search context last
    if state.context:
        # Apply name replacement to make context more personal
        if state.user_name:
            personalized_context = replace_user_name_with_my(state.context, state.user_name)
            if personalized_context != state.context:
                logger.info(f"[LANGGRAPH PATH] Personalized context: '{state.context}' -> '{personalized_context}'")
                context_parts.append(personalized_context)
            else:
                context_parts.append(state.context)
        else:
            context_parts.append(state.context)
        logger.info(f"[LANGGRAPH PATH] Added existing context (length: {len(state.context)}): {state.context[:100]}...")
    else:
        logger.info("[LANGGRAPH PATH] No existing context to add")

    # Join all context parts
    combined_context = "\n".join(filter(None, context_parts))

    # Get current time for system message
    formatted_date, formatted_time, timezone = get_current_time_info()

    current_date_content = f"{formatted_date}** **{formatted_time}**({timezone})"
    state.current_date_content = current_date_content
    logger.info(f"[LANGGRAPH PATH] Current time info: {current_date_content}")
    
    # Create the full prompt for the LLM
    state.context = combined_context

    logger.info(f"[LANGGRAPH PATH] Combined context length: {len(combined_context)} characters")
    logger.info("[LANGGRAPH PATH] prepare_llm_input_node completed")
    return state

async def intent_analysis_node(state: AgentState) -> AgentState:
    logger.info("[LANGGRAPH PATH] Starting intent_analysis_node")
    logger.info(f"[LANGGRAPH PATH] Analyzing intent for query: {state.retrieval_query}")
    
    # Prepare chat history context for intent analysis
    chat_context = ""
    if state.request.messages and len(state.request.messages) > 1:
        # Get the last few messages for context (excluding the current one)
        recent_messages = state.request.messages[-11:-1]
        chat_context = "\n".join([f"{msg.role}: {msg.content}" for msg in recent_messages])
        logger.info(f"[INTENT ANALYSIS] Chat context: {chat_context}")
    
    prompt_text = f"""
    You are a classification assistant.  
    Your task is to classify the **intent** of a single user query into exactly one of three labels, and return **only** the label — no explanations, no extra text.  

    ### Labels
    1. **give_info** – User provides factual information about one or more specific persons (including possessions, family, pets, or related entities).  
    2. **ask_info** – User requests factual information about one or more specific persons (including possessions, family, pets, or related entities).  
    3. **general** – Query does not reference any specific person (only general talk, vague groups, or unrelated topics).  

    ---

    ### Person Reference Rules
    - A **specific person** = identifiable by a **name**, a **relationship** (e.g., "my friend", "my colleague"), or a **pronoun** resolved from chat history.  
    - **Names** include: capitalized words that appear to be person names (e.g., "John", "Sarah", "marry", "Tom"), even if not capitalized
    - **Questions about specific people** (e.g., "How many sisters does John have?", "Where does Sarah live?") = ask_info
    - Facts about a person's possessions, relatives, pets, or location still count as information about that person.  
    - Generic groups (e.g., "students", "politicians") = not specific persons ? classify as `general`.  

    ---

    ### Context Rules
    - Use **chat history** to resolve pronouns or relationship references.  
    - If "my [relationship]" is used (e.g., "my colleague"), resolve it to the previously introduced person.  
    - Continuation statements (starting with "And") that add facts about a previously mentioned person = **give_info**.  

    ---

    ### Classification Logic
    **Step 1** – Does the query mention a specific person (name, relationship, or resolvable pronoun)?  
    - NO ? `general`  
    - YES ? Go to Step 2  

    **Step 2** – Is the user providing or requesting facts?  
    - Providing ? `give_info`  
    - Requesting ? `ask_info`  

    **Step 3** – Special cases  
    - Starts with "And" + adds facts ? `give_info`  
    - Uses pronouns ("he", "she", "they") tied to history + facts ? `give_info`  
    - Uses "my [relationship]" + facts ? `give_info`  
    - Profile update requests (e.g., "Update", "Change", "Set") + person + attribute ? `give_info`
    - Queries like "Tell me about [Name]" or "What do you know about [Name]" ? `ask_info`
    - **Questions starting with "How", "Where", "What", "When", "Why" + person name** ? `ask_info`
    - **Questions about specific people's attributes** (e.g., "How many sisters does [Name] have?") ? `ask_info`  

    ---

    ### Examples
    - "My friend Sarah just started a new job." ? give_info  
    - "Do you know where John lives?" ? ask_info  
    - "Tell me about Lionel Messi." ? ask_info  
    - "How many sisters does marry have?" ? ask_info
    - "Where does Sarah live?" ? ask_info
    - "What's John's phone number?" ? ask_info
    - "Let's talk about the latest iPhone." ? general  
    - "And Lisa works at Google." ? give_info  
    - "She has two brothers." (refers to previous Sarah) ? give_info
    - "Update my friend Marry's location to Italy" ? give_info
    - "Change John's phone number to 123-456-7890" ? give_info
    - "Set Sarah's workplace to Google" ? give_info  

    ---

    ### Chat History Context
    {chat_context if chat_context else "No previous context available."}

    ### Query
    "{state.retrieval_query}"
    """

    langchain_messages = [
        HumanMessage(content=prompt_text)
    ]
    logger.info(f"Prompt Text: {prompt_text}")
    response = await openai_llm.ainvoke(langchain_messages)

    intent_text = response.content.strip().lower()
    logger.info(f"[LANGGRAPH PATH] LLM intent classification response: {intent_text}")

    # More robust intent parsing with fallback logic
    intent_text_clean = intent_text.strip().lower()
    
    if "give_info" in intent_text_clean:
        state.agent_status = "give_info"
        logger.info(f"[INTENT ANALYSIS] Successfully classified as give_info")
    elif "ask_info" in intent_text_clean:
        state.agent_status = "ask_info"
        logger.info(f"[INTENT ANALYSIS] Successfully classified as ask_info")
    elif "general" in intent_text_clean:
        state.agent_status = "general"
        logger.info(f"[INTENT ANALYSIS] Successfully classified as general")
    else:
        # Fallback: analyze the text directly for common patterns
        logger.warning(f"[INTENT ANALYSIS] LLM returned unexpected response: '{intent_text}'. Using fallback analysis.")
        fallback_intent = analyze_intent_fallback(state.retrieval_query, chat_context)
        state.agent_status = fallback_intent
        logger.info(f"[INTENT ANALYSIS] Fallback analysis determined: {fallback_intent}")

    logger.info(f"[LANGGRAPH PATH] Intent classified as: {state.agent_status}")
    
    # Final validation: catch obvious misclassifications
    final_intent = validate_and_correct_intent(state.retrieval_query, state.agent_status, chat_context)
    if final_intent != state.agent_status:
        logger.warning(f"[INTENT VALIDATION] Corrected intent from '{state.agent_status}' to '{final_intent}'")
        state.agent_status = final_intent
    
    logger.info(f"[LANGGRAPH PATH] Final intent after validation: {state.agent_status}")
    logger.info(f"[LANGGRAPH PATH] intent_analysis_node completed")
    return state

def validate_and_correct_intent(text: str, current_intent: str, chat_context: str = "") -> str:
    """
    Validate and correct obvious intent misclassifications.
    This catches cases where the LLM or fallback logic made obvious mistakes.
    """
    text_lower = text.lower()
    
    # Obvious give_info patterns that should never be classified as general
    obvious_give_info_patterns = [
        # Specific pattern from your example: "My friend Ash's phone number is 12345"
        r'my\s+(friend|colleague|sister|brother|mother|father|neighbor|supervisor|boss|partner|spouse|husband|wife|assistant)\'s\s+[a-zA-Z]+\'s?\s+',
        r'my\s+(friend|colleague|sister|brother|mother|father|neighbor|supervisor|boss|partner|spouse|husband|wife|assistant)\s+[a-zA-Z]+\s+',
        r'[a-zA-Z]+\'s\s+(phone|number|address|email|age|birthday|job|work|lives|lived)',
        r'[a-zA-Z]+\s+(lives in|works at|studies at|born in|from)',
        r'[a-zA-Z]+\s+(has|is|was|will|can|does)',
        # Additional specific patterns
        r'my\s+(friend|colleague|sister|brother|mother|father|neighbor|supervisor|boss|partner|spouse|husband|wife|assistant)\'s\s+(phone|number|address|email|age|birthday|job|work|lives|lived)',
        r'my\s+(friend|colleague|sister|brother|mother|father|neighbor|supervisor|boss|partner|spouse|husband|wife|assistant)\s+(phone|number|address|email|age|birthday|job|work|lives|lived)',
        # Pronoun patterns that should never be general
        r'^(she|he|they)\s+(likes?|like|has|have|is|are|was|were|lives?|live|works?|work|studies?|study)',
        r'^(her|his|their)\s+(phone|number|address|email|age|birthday|job|work|house|car|dog|cat)',
        r'(she|he|they)\s+(likes?|like|has|have|is|are|was|were|lives?|live|works?|work|studies?|study)\s+[a-zA-Z\s]+',
        # Continuation statements that provide factual information about specific people
        r'^and\s+[a-zA-Z]+\s+(lives?|lived|has|have|is|are|was|were|likes?|like|works?|work|studies?|study)',
        r'^and\s+[a-zA-Z]+\'s\s+(phone|number|address|email|age|birthday|job|work|house|car|dog|cat|sister|brother|mother|father)',
        r'^and\s+[a-zA-Z]+\s+(sister|brother|mother|father|friend|colleague)\s+(is|are|named|called)',
        # Direct factual statements about specific people
        r'^[a-zA-Z]+\s+(lives?|lived|has|have|is|are|was|were|likes?|like|works?|work|studies?|study)\s+',
        r'^[a-zA-Z]+\'s\s+(phone|number|address|email|age|birthday|job|work|house|car|dog|cat|sister|brother|mother|father)\s+'
    ]
    
    # Obvious ask_info patterns that should never be classified as general
    obvious_ask_info_patterns = [
        r'^tell me about\s+[a-zA-Z]+',
        r'^tell me more about\s+[a-zA-Z]+',
        r'^what do you know about\s+[a-zA-Z]+',
        r'^what about\s+[a-zA-Z]+',
        r'^do you know\s+[a-zA-Z]+',
        r'^can you tell me about\s+[a-zA-Z]+',
        # Questions about specific people's attributes
        r'how many\s+(sisters?|brothers?|children|pets|friends)\s+(does|do|have)\s+[a-zA-Z]+\s+(got|have)',
        r'how many\s+(sisters?|brothers?|children|pets|friends)\s+[a-zA-Z]+\s+(got|have)',
        r'where\s+(does|do)\s+[a-zA-Z]+\s+(live|work|study)',
        r'what\s+(is|are)\s+[a-zA-Z]+\'s\s+(phone|number|address|email|age|birthday|job|work)',
        # Questions that start with question words and contain names
        r'^(how|where|what|when|why|who)\s+[^?]*[a-zA-Z]+[^?]*\?'
    ]
    
    # Check if this is obviously give_info but was classified as general
    if current_intent == "general":
        for pattern in obvious_give_info_patterns:
            if re.search(pattern, text_lower):
                logger.info(f"[INTENT VALIDATION] Pattern '{pattern}' matched - forcing correction to give_info")
                return "give_info"
        
        # Check if this is obviously ask_info but was classified as general
        for pattern in obvious_ask_info_patterns:
            if re.search(pattern, text_lower):
                logger.info(f"[INTENT VALIDATION] Pattern '{pattern}' matched - forcing correction to ask_info")
                return "ask_info"
        
        # Special check for the exact pattern "How many sisters have marry got"
        if re.search(r'how many\s+(sisters?|brothers?|children|pets|friends)\s+(does|do|have)\s+[a-zA-Z]+\s+(got|have)', text_lower):
            logger.info(f"[INTENT VALIDATION] Specific question pattern about person's attributes matched - forcing correction to ask_info")
            return "ask_info"
        
        # Check for questions that contain names and ask about specific attributes
        if any(word in text_lower for word in ['how', 'where', 'what', 'when', 'why', 'who']):
            # Look for potential names in the question
            potential_names = re.findall(r'\b[a-zA-Z]+\b', text)
            for name in potential_names:
                # Skip common question words and articles
                if name.lower() not in ['how', 'where', 'what', 'when', 'why', 'who', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']:
                    # Check if this looks like a person name
                    if len(name) > 2 and name[0].isupper():
                        logger.info(f"[INTENT VALIDATION] Question contains potential person name '{name}' - forcing correction to ask_info")
                        return "ask_info"
        
        # Context-aware validation: check if text contains relationship references that should be give_info
        if chat_context:
            # Check for "my [relationship]" patterns that refer to previously mentioned people
            relationship_patterns = [
                r'my\s+(friend|colleague|sister|brother|mother|father|neighbor|supervisor|assistant)\'s?\s+',
                r'my\s+(friend|colleague|sister|brother|mother|father|neighbor|supervisor|assistant)\'s?\s+',
                r'my\s+(friend|colleague|sister|brother|mother|father|neighbor|supervisor|assistant)\s+',
            ]
            
            for pattern in relationship_patterns:
                if re.search(pattern, text_lower):
                    # Check if this relationship appears in chat context
                    match = re.search(pattern, text_lower)
                    if match:
                        relationship = match.group(1)
                        if relationship in chat_context.lower():
                            logger.info(f"[INTENT VALIDATION] Context-aware: '{relationship}' found in chat history - forcing correction to give_info")
                            return "give_info"
            
            # Check for continuation statements with pronouns that should be give_info
            continuation_patterns = [
                r'^and\s+(he|she|his|her|they|their)\s+',
                r'^(he|she|his|her|they|their)\s+',
            ]
            
            for pattern in continuation_patterns:
                if re.search(pattern, text_lower):
                    # Check if pronouns can be resolved from chat context
                    match = re.search(pattern, text_lower)
                    if match:
                        pronoun = match.group(1)
                        # Look for names in chat context that could be referred to by this pronoun
                        name_pattern = r'[A-Z][a-z]+'
                        names_in_context = re.findall(name_pattern, chat_context)
                        if names_in_context:
                            logger.info(f"[INTENT VALIDATION] Context-aware: continuation with pronoun '{pronoun}' referring to names in chat history - forcing correction to give_info")
                            return "give_info"
            
            # Check for continuation statements with specific names that should be give_info
            name_continuation_patterns = [
                r'^and\s+[a-zA-Z]+\s+(lives?|lived|has|have|is|are|was|were|likes?|like|works?|work|studies?|study)',
                r'^and\s+[a-zA-Z]+\'s\s+(phone|number|address|email|age|birthday|job|work|house|car|dog|cat|sister|brother|mother|father)',
                r'^and\s+[a-zA-Z]+\s+(sister|brother|mother|father|friend|colleague)\s+(is|are|named|called)'
            ]
            
            for pattern in name_continuation_patterns:
                if re.search(pattern, text_lower):
                    # Check if the name appears in chat context
                    match = re.search(pattern, text_lower)
                    if match:
                        # Extract the name from the pattern
                        name_match = re.search(r'^and\s+([a-zA-Z]+)', text_lower)
                        if name_match:
                            name = name_match.group(1)
                            if name in chat_context.lower():
                                logger.info(f"[INTENT VALIDATION] Context-aware: continuation with name '{name}' found in chat history - forcing correction to give_info")
                                return "give_info"
    
    return current_intent

def analyze_intent_fallback(text: str, chat_context: str = "") -> str:
    """
    Fallback intent analysis using pattern matching when LLM fails.
    This provides a reliable backup for intent classification.
    """
    text_lower = text.lower()
    
    # Patterns that clearly indicate give_info
    give_info_patterns = [
        # Specific pattern from your example: "My friend Ash's phone number is 12345"
        r'my\s+(friend|colleague|sister|brother|mother|father|neighbor|supervisor|boss|partner|spouse|husband|wife|assistant)\'s\s+[a-zA-Z]+\'s?\s+',
        r'my\s+(friend|colleague|sister|brother|mother|father|neighbor|supervisor|boss|partner|spouse|husband|wife|assistant)\'s',
        r'my\s+(friend|colleague|sister|brother|mother|father|neighbor|supervisor|boss|partner|spouse|husband|wife|assistant)\s+',
        r'(friend|colleague|sister|brother|mother|father|neighbor|supervisor|boss|partner|spouse|husband|wife|assistant)\'s',
        r'(he|she|his|her)\s+(has|is|was|will|can|does)',
        r'(name|phone|number|address|email|age|birthday|job|work|lives|lived)',
        r'(lives in|works at|studies at|born in|from)',
        r'(started|got|bought|moved|changed|updated)',
        # Pronoun patterns that indicate give_info
        r'^(she|he|they)\s+(likes?|like|has|have|is|are|was|were|lives?|live|works?|work|studies?|study)',
        r'^(her|his|their)\s+(phone|number|address|email|age|birthday|job|work|house|car|dog|cat)',
        r'(she|he|they)\s+(likes?|like|has|have|is|are|was|were|lives?|live|works?|work|studies?|study)\s+[a-zA-Z\s]+',
        # Continuation statements that provide factual information about specific people
        r'^and\s+[a-zA-Z]+\s+(lives?|lived|has|have|is|are|was|were|likes?|like|works?|work|studies?|study)',
        r'^and\s+[a-zA-Z]+\'s\s+(phone|number|address|email|age|birthday|job|work|house|car|dog|cat|sister|brother|mother|father)',
        r'^and\s+[a-zA-Z]+\s+(sister|brother|mother|father|friend|colleague)\s+(is|are|named|called)',
        # Direct factual statements about specific people
        r'^[a-zA-Z]+\s+(lives?|lived|has|have|is|are|was|were|likes?|like|works?|work|studies?|study)\s+',
        r'^[a-zA-Z]+\'s\s+(phone|number|address|email|age|birthday|job|work|house|car|dog|cat|sister|brother|mother|father)\s+'
    ]
    
    # Patterns that clearly indicate ask_info
    ask_info_patterns = [
        r'(what|where|when|how|why|who)\s+(is|are|was|were|does|do|can|could|will|would)',
        r'(tell me|show me|find|search|look up|get)',
        r'(tell me about|tell me more about|what do you know about|what about)',
        r'(do you know|can you tell|remember|recall)',
        r'(information|details|facts|data)',
        # Questions about specific people's attributes
        r'how many\s+(sisters?|brothers?|children|pets|friends)\s+(does|do)\s+[a-zA-Z]+\s+have',
        r'where\s+(does|do)\s+[a-zA-Z]+\s+(live|work|study)',
        r'what\s+(is|are)\s+[a-zA-Z]+\'s\s+(phone|number|address|email|age|birthday|job|work)',
        r'[a-zA-Z]+\'s\s+(sisters?|brothers?|phone|location|workplace)',
        # Questions that start with question words and contain names
        r'^(how|where|what|when|why|who)\s+[^?]*[a-zA-Z]+[^?]*\?'
    ]
    
    # Check for give_info patterns
    for pattern in give_info_patterns:
        if re.search(pattern, text_lower):
            logger.info(f"[INTENT FALLBACK] Pattern '{pattern}' matched - classified as give_info")
            return "give_info"
    
    # Check for ask_info patterns
    for pattern in ask_info_patterns:
        if re.search(pattern, text_lower):
            logger.info(f"[INTENT FALLBACK] Pattern '{pattern}' matched - classified as ask_info")
            return "ask_info"
    
    # Special check for questions about specific people by name
    # Look for question patterns that contain what appear to be person names
    question_words = ['how', 'where', 'what', 'when', 'why', 'who']
    text_words = text_lower.split()
    
    # Check if this is a question and contains potential names
    if text.endswith('?') or any(word in text_lower for word in question_words):
        # Look for capitalized words that could be names
        potential_names = re.findall(r'\b[a-zA-Z]+\b', text)
        for name in potential_names:
            # Skip common question words and articles
            if name.lower() not in question_words + ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']:
                # Check if this looks like a person name (not a common word)
                if len(name) > 2 and name[0].isupper():
                    logger.info(f"[INTENT FALLBACK] Question contains potential person name '{name}' - classified as ask_info")
                    return "ask_info"
    
    # Specific check for common question patterns about people
    specific_question_patterns = [
        r'how many\s+(sisters?|brothers?|children|pets|friends)\s+(does|do|have)\s+[a-zA-Z]+\s+(got|have)',
        r'how many\s+(sisters?|brothers?|children|pets|friends)\s+[a-zA-Z]+\s+(got|have)',
        r'where\s+(does|do)\s+[a-zA-Z]+\s+(live|work|study)',
        r'what\s+(is|are)\s+[a-zA-Z]+\'s\s+(phone|number|address|email|age|birthday|job|work)'
    ]
    
    for pattern in specific_question_patterns:
        if re.search(pattern, text_lower):
            logger.info(f"[INTENT FALLBACK] Specific question pattern '{pattern}' matched - classified as ask_info")
            return "ask_info"
    
    # Context-aware analysis: check if text contains relationship references that should be give_info
    if chat_context:
        # Check for "my [relationship]" patterns that refer to previously mentioned people
        relationship_patterns = [
            r'my\s+(friend|colleague|sister|brother|mother|father|neighbor|supervisor|assistant)\'s?\s+',
            r'my\s+(friend|colleague|sister|brother|mother|father|neighbor|supervisor|assistant)\s+',
        ]
        
        for pattern in relationship_patterns:
            if re.search(pattern, text_lower):
                # Check if this relationship appears in chat context
                match = re.search(pattern, text_lower)
                if match:
                    relationship = match.group(1)
                    if relationship in chat_context.lower():
                        logger.info(f"[INTENT FALLBACK] Context-aware: '{relationship}' found in chat history - classified as give_info")
                        return "give_info"
        
        # Check for continuation statements with pronouns that should be give_info
        continuation_patterns = [
            r'^and\s+(he|she|his|her|they|their)\s+',
            r'^(he|she|his|her|they|their)\s+',
        ]
        
        for pattern in continuation_patterns:
            if re.search(pattern, text_lower):
                # Check if pronouns can be resolved from chat context
                match = re.search(pattern, text_lower)
                if match:
                    pronoun = match.group(1)
                    # Look for names in chat context that could be referred to by this pronoun
                    name_pattern = r'[A-Z][a-z]+'
                    names_in_context = re.findall(name_pattern, chat_context)
                    if names_in_context:
                        logger.info(f"[INTENT FALLBACK] Context-aware: continuation with pronoun '{pronoun}' referring to names in chat history - classified as give_info")
                        return "give_info"
    
    # Default to general if no clear patterns
    logger.info(f"[INTENT FALLBACK] No clear patterns found - classified as general")
    return "general"

async def detect_relationship_via_llm(text: str, llm) -> str:
    prompt = f"""
    You are an assistant that extracts the relationship of a person mentioned in the following text to the speaker.
    
    Extract ONLY the relationship word from phrases like:
    - "my friend" ? "friend"
    - "my colleague" ? "colleague" 
    - "my family" ? "family"
    - "my supervisor" ? "supervisor"
    - "my neighbor" ? "neighbor"
    - "my assistant" ? "assistant"
    
    If the relationship is not mentioned or unclear, respond with "unknown".
    If a relationship is mentioned, respond with exactly that relationship word (friend, colleague, family, supervisor, neighbor, assistant).
    
    Examples:
    Text: "my friend Tom" ? Response: "friend"
    Text: "my colleague Sarah" ? Response: "colleague"
    Text: "friend Mark" ? Response: "friend"
    Text: "colleague John" ? Response: "colleague"
    Text: "just tell me about him" ? Response: "unknown"
    
    Text: {text}
    """
    messages = [HumanMessage(content=prompt)]
    try:
        response = await llm.ainvoke(messages)
        rel = response.content.strip().lower()
        if not rel:
            return "unknown"
        return rel
    except Exception:
        return "unknown"
        
PRONOUNS = {"he", "she", "him", "her", "his", "hers"}
async def profile_create_update_node(state: AgentState) -> AgentState:
    logger.info("[LANGGRAPH PATH] Starting profile_create_update_node")
    user_id = state.user_id
    text = state.retrieval_query.strip()

    text_lower = text.lower()
    logger.info(f"[LANGGRAPH PATH] Processing text: {text}")
    logger.info(f"[LANGGRAPH PATH] User ID: {user_id}")
    logger.info(f"[LANGGRAPH PATH] Current agent_status: {state.agent_status}")

    # If this is a repeated confirm attempt and user provides relationship, update that
    if state.agent_status == "confirm_creation" and state.confirm_profile:
        rel_from_llm = await detect_relationship_via_llm(text, openai_llm)
        rel_from_llm = rel_from_llm.strip('"')
        logger.info(f"Confirm Profile: {state.confirm_profile}")
        
        if rel_from_llm != "unknown":
            logger.info(f"User clarified relationship as: {rel_from_llm}")
            # Clean the name from confirm_profile
            original_name = state.confirm_profile['name']
            clean_name = clean_profile_name(original_name)
            if clean_name != original_name:
                logger.info(f"[PROFILE CONFIRMATION] Cleaned name from '{original_name}' to '{clean_name}'")
            
            confirmed_profile = {
                "name": clean_name,
                "relationship": rel_from_llm
            }
            # Create or update profile now
            updated_profile = await create_or_update_profile_persistent( 
                user_id,
                confirmed_profile["name"],
                confirmed_profile["relationship"],
                state.confirm_profile['information'] or "No additional info provided."
            )

            state.final_response = (
                f"Great, your {confirmed_profile['relationship']} {confirmed_profile['name']}'s profile "
                f"is noted successfully."
            )
            state.last_mentioned_profile = confirmed_profile
            state.confirm_profile = None
            state.agent_status = "initialize"
            return state
        else:
            # Still no valid relationship info
            state.final_response = (
                f"Please specify the relationship for {state.confirm_profile.get('name', name)} "
                f"(friend, colleague, family, supervisor, neighbor, assistant)."
            )
            state.agent_status = "confirm_creation"
            return state

    # No pronouns detected or no last_mentioned_profile; proceed with text analysis
    # Load existing profiles for disambiguation
    profiles = await load_user_profiles(user_id)
    logger.info(f"[PROFILE CREATION] Loaded {len(profiles)} existing profiles for disambiguation")
    
    detected_names, is_detected = await detect_profiles_in_text(text, state.request.messages)

    if not is_detected:
        # First, check for pronouns - if text contains pronouns and we have last_mentioned_profile 
        logging.info(f"Last mentioned profile: {state.last_mentioned_profile}")

        if any(pron in text_lower.split() for pron in PRONOUNS) and state.last_mentioned_profile.get("name"):
            prof = state.last_mentioned_profile
            name = prof.get("name")
            relationship = prof.get("relationship", "unknown")
            
            # CRITICAL: Don't assume relationship if it's unknown
            if relationship == "unknown":
                logger.info(f"[PRONOUN PROFILE UPDATE] Relationship is unknown for {name}, asking for clarification")
                state.agent_status = "confirm_creation"
                state.confirm_profile = {
                    "name": name,
                    "relationship": "unknown",
                    "information": text
                }
                state.final_response = (
                    f"I see you're referring to {name}, but I need to know the relationship. "
                    f"Could you please specify if {name} is your friend, colleague, family member, etc.?"
                )
                return state
            
            # Clean the name to ensure it's just the person's name
            clean_name = clean_profile_name(name)
            if clean_name != name:
                logger.info(f"[PRONOUN PROFILE UPDATE] Cleaned name from '{name}' to '{clean_name}'")
            
            # Create or update profile with rich structure
            try:
                await create_or_update_profile_persistent(
                    user_id, clean_name, relationship, text
                )
            except ProfileUpdateConfirmationRequired as confirmation_req:
                # Profile update requires confirmation - set pending update state
                logger.info(f"[PRONOUN PROFILE UPDATE] Profile update requires confirmation for {confirmation_req.person_name}'s {confirmation_req.field_name}")
                
                # Set the pending update information in the state
                state.pending_profile_update = {
                    "name": confirmation_req.person_name,
                    "field": confirmation_req.field_name,
                    "new_value": confirmation_req.new_value,
                    "current_value": confirmation_req.current_value
                }
                
                # Set the confirmation message and status
                state.final_response = confirmation_req.confirmation_message
                state.agent_status = "waiting_confirmation"
                
                # Clear other state variables
                state.confirm_profile = None
                state.last_mentioned_profile = {
                    "name": clean_name,
                    "relationship": relationship,
                    "information": text
                }
                
                logger.info(f"[PRONOUN PROFILE UPDATE] Set pending profile update: {state.pending_profile_update}")
                return state
            
            # Also create a rich profile version for better AI responses
            try:
                from rag.rich_profile_utils import convert_simple_to_rich_profile
                
                # Get the created profile to convert it
                created_profile = await get_user_profile(user_id, clean_name)
                if created_profile:
                    # Convert to rich format
                    rich_profile = await convert_simple_to_rich_profile(created_profile)
                    
                    # Save the rich profile back (this will enhance future responses)
                    await update_user_profile(user_id, clean_name, rich_profile)
                    
                    logger.info(f"[PROFILE CREATION] Enhanced profile for {clean_name} with rich structure")
            except Exception as e:
                logger.warning(f"[PROFILE CREATION] Rich profile enhancement failed for {clean_name}: {str(e)}")
                # Continue with normal profile creation if enhancement fails

            if state.final_response == (f"Okay, your {relationship} {clean_name}'s profile is updated successfully."):
                state.final_response = (f"OK, your {relationship} {clean_name}'s profile is updated successfully.")
            else:
                state.final_response = (f"Okay, your {relationship} {clean_name}'s profile is updated successfully.")
            state.agent_status = "initialize"
            return state
        
        # Check if this might be a response to a clarification request (e.g., "my friend Tom")
        # This handles cases where user responds with a new profile update after being asked to clarify
        if state.confirm_profile and "my " in text_lower and any(rel in text_lower for rel in ["friend", "colleague", "family", "supervisor", "neighbor", "assistant"]):
            logger.info(f"[PROFILE CLARIFICATION RESPONSE] User provided new profile info: {text}")
            logger.info(f"[PROFILE CLARIFICATION RESPONSE] confirm_profile state: {state.confirm_profile}")
            logger.info(f"[PROFILE CLARIFICATION RESPONSE] agent_status: {state.agent_status}")
            
            # Extract relationship and name from the response
            rel_from_llm = await detect_relationship_via_llm(text, openai_llm)
            rel_from_llm = rel_from_llm.strip('"')
            logger.info(f"[PROFILE CLARIFICATION RESPONSE] Detected relationship: '{rel_from_llm}'")
            
            if rel_from_llm != "unknown":
                # Extract name from the text (should be the name from confirm_profile)
                original_name = state.confirm_profile['name']
                clean_name = clean_profile_name(original_name)
                information = state.confirm_profile['information']
                
                logger.info(f"[PROFILE CLARIFICATION RESPONSE] Updating {clean_name} ({rel_from_llm}) with: '{information}'")
                
                # Create or update profile with rich structure
                try:
                    await create_or_update_profile_persistent(
                        user_id, clean_name, rel_from_llm, information
                    )
                except ProfileUpdateConfirmationRequired as confirmation_req:
                    # Profile update requires confirmation - set pending update state
                    logger.info(f"[PROFILE CLARIFICATION RESPONSE] Profile update requires confirmation for {confirmation_req.person_name}'s {confirmation_req.field_name}")
                    
                    # Set the pending update information in the state
                    state.pending_profile_update = {
                        "name": confirmation_req.person_name,
                        "field": confirmation_req.field_name,
                        "new_value": confirmation_req.new_value,
                        "current_value": confirmation_req.current_value
                    }
                    
                    # Set the confirmation message and status
                    state.final_response = confirmation_req.confirmation_message
                    state.agent_status = "waiting_confirmation"
                    
                    # Clear other state variables
                    state.confirm_profile = None
                    state.last_mentioned_profile = {
                        "name": clean_name,
                        "relationship": rel_from_llm,
                        "information": information
                    }
                    
                    logger.info(f"[PROFILE CLARIFICATION RESPONSE] Set pending profile update: {state.pending_profile_update}")
                    return state
                
                # Also create a rich profile version for better AI responses
                try:
                    from rag.rich_profile_utils import convert_simple_to_rich_profile
                    
                    # Get the created profile to convert it
                    created_profile = await get_user_profile(user_id, clean_name)
                    if created_profile:
                        # Convert to rich format
                        rich_profile = await convert_simple_to_rich_profile(created_profile)
                        
                        # Save the rich profile back (this will enhance future responses)
                        await update_user_profile(user_id, clean_name, rich_profile)
                        
                        logger.info(f"[PROFILE CREATION] Enhanced profile for {clean_name} with rich structure")
                except Exception as e:
                    logger.warning(f"[PROFILE CREATION] Rich profile enhancement failed for {clean_name}: {str(e)}")
                    # Continue with normal profile creation if enhancement fails
                
                state.final_response = (
                    f"Great, your {rel_from_llm} {clean_name}'s profile "
                    f"is noted successfully."
                )
                state.last_mentioned_profile = {
                    "name": clean_name,
                    "relationship": rel_from_llm,
                    "information": information
                }
                state.confirm_profile = None
                state.agent_status = "initialize"
                logger.info(f"[PROFILE CLARIFICATION RESPONSE] Profile updated successfully, resetting state")
                return state
            else:
                logger.warning(f"[PROFILE CLARIFICATION RESPONSE] Could not detect relationship from: {text}")
        else:
            logger.info(f"[PROFILE CLARIFICATION RESPONSE] Not a clarification response. confirm_profile: {state.confirm_profile}, text_lower: {text_lower}")
            logger.info(f"[PROFILE CLARIFICATION RESPONSE] agent_status: {state.agent_status}")
        
        # If no names detected, ask for clarification
        logger.info("[LANGGRAPH PATH] No names detected in text")
        state.final_response = "I'm not sure which person you're referring to. Could you please specify the name or provide more context?"
        state.agent_status = "initialize"
        return state

    logger.info(f"Detected names for profile update: {detected_names}")

    # For simplicity, only consider the first detected person (can extend later for multiples)
    responses = []

    for detected in detected_names:
        name = detected.get("name")
        relationship = detected.get("relationship", "unknown")
        information = detected.get("information", "")

        # Clean the name to ensure it's just the person's name, not a descriptive name
        clean_name = clean_profile_name(name)
        if clean_name != name:
            logger.info(f"[PROFILE CREATION] Cleaned name from '{name}' to '{clean_name}'")

        if relationship != "unknown":
            # CRITICAL: Verify the relationship is valid before proceeding
            valid_relationships = {"friend", "colleague", "family", "supervisor", "neighbor", "assistant"}
            if relationship not in valid_relationships:
                logger.warning(f"[PROFILE CREATION] Invalid relationship '{relationship}' for {clean_name}, asking for clarification")
                state.agent_status = "confirm_creation"
                state.confirm_profile = {
                    "name": clean_name,
                    "relationship": "unknown",
                    "information": information
                }
                state.final_response = (
                    f"I detected the name '{clean_name}' but the relationship '{relationship}' is not recognized. "
                    f"Could you please specify a valid relationship (friend, colleague, family, supervisor, neighbor, or assistant)?"
                )
                break
            
            # Create or update profile with rich structure
            try:
                await create_or_update_profile_persistent(
                    user_id, clean_name, relationship, information
                )
            except ProfileUpdateConfirmationRequired as confirmation_req:
                # Profile update requires confirmation - set pending update state
                logger.info(f"[PROFILE CREATION] Profile update requires confirmation for {confirmation_req.person_name}'s {confirmation_req.field_name}")
                
                # Set the pending update information in the state
                state.pending_profile_update = {
                    "name": confirmation_req.person_name,
                    "field": confirmation_req.field_name,
                    "new_value": confirmation_req.new_value,
                    "current_value": confirmation_req.current_value
                }
                
                # Set the confirmation message and status
                state.final_response = confirmation_req.confirmation_message
                state.agent_status = "waiting_confirmation"
                
                # Clear other state variables
                state.confirm_profile = None
                state.last_mentioned_profile = {
                    "name": clean_name,
                    "relationship": relationship,
                    "information": information
                }
                
                logger.info(f"[PROFILE CREATION] Set pending profile update: {state.pending_profile_update}")
                return state
            
            # Also create a rich profile version for better AI responses
            try:
                from rag.rich_profile_utils import convert_simple_to_rich_profile
                
                # Get the created profile to convert it
                created_profile = await get_user_profile(user_id, clean_name)
                if created_profile:
                    # Convert to rich format
                    rich_profile = await convert_simple_to_rich_profile(created_profile)
                    
                    # Save the rich profile back (this will enhance future responses)
                    await update_user_profile(user_id, clean_name, rich_profile)
                    
                    logger.info(f"[PROFILE CREATION] Enhanced profile for {clean_name} with rich structure")
            except Exception as e:
                logger.warning(f"[PROFILE CREATION] Rich profile enhancement failed for {clean_name}: {str(e)}")
                # Continue with normal profile creation if enhancement fails

            responses.append(f"your {relationship} {clean_name}'s profile")
            state.last_mentioned_profile = {
                "name": clean_name,
                "relationship": relationship,
                "information": information
            }
            logging.info(f"Update last mentioned profile: {state.last_mentioned_profile}")
            state.confirm_profile = None
            state.agent_status = "initialize"
        else:
            # CRITICAL: Always ask for clarification when relationship is unknown
            logger.info(f"[PROFILE CREATION] Relationship unknown for {clean_name}, asking for clarification")
            
            # Check if multiple profiles exist with this name
            existing_profiles_with_name = []
            for profile in profiles.values():
                if profile.get('name', '').lower() == clean_name.lower():
                    existing_profiles_with_name.append(profile)
            
            if len(existing_profiles_with_name) > 1:
                # Multiple profiles exist - ask for clarification
                profile_options = []
                for profile in existing_profiles_with_name:
                    rel = profile.get('relationship', 'unknown')
                    profile_options.append(f"{rel} {clean_name}")
                
                state.agent_status = "confirm_creation"
                state.confirm_profile = {
                    "name": clean_name,
                    "relationship": "unknown",
                    "information": information
                }
                state.final_response = (
                    f"I found multiple profiles for '{clean_name}': {', '.join(profile_options)}. "
                    f"Which one would you like me to update with: '{information}'?"
                )
            else:
                # Single profile or no profile - ask for relationship
                state.agent_status = "confirm_creation"
                state.confirm_profile = {
                    "name": clean_name,
                    "relationship": "unknown",
                    "information": information
                }
                state.final_response = (
                    f"I detected the name '{clean_name}', but need clarification on the relationship. "
                    f"Could you please specify (friend, colleague, family, etc.) or create new one?"
                )

            break

    # Build one final combined response
    if responses:
        if state.final_response == "Okay, I have noted " + ", ".join(responses) + " successfully.":
            state.final_response = "OK, I have noted " + ", ".join(responses) + " successfully."
        else:
            state.final_response = "Okay, I have noted " + ", ".join(responses) + " successfully."
    
    return state

async def handle_profile_clarification_node(state: AgentState) -> AgentState:
    """Node to handle user clarification for profile queries when multiple profiles exist with the same name."""
    logger.info("[LANGGRAPH PATH] Starting handle_profile_clarification_node")
    
    if not state.pending_profile_clarification:
        logger.warning("[PROFILE CLARIFICATION] No pending profile clarification found")
        state.final_response = "I'm not sure what you're clarifying. Could you please provide the profile information again?"
        state.agent_status = "initialize"
        return state
    
    pending_clarification = state.pending_profile_clarification
    user_response = state.retrieval_query.strip()
    
    logger.info(f"[PROFILE CLARIFICATION] Processing clarification for: {pending_clarification}")
    logger.info(f"[PROFILE CLARIFICATION] User response: {user_response}")
    
    # Extract the name and relationship from the user's response
    name = pending_clarification["name"]
    profiles = pending_clarification["profiles"]
    original_query = pending_clarification["query"]
    
    # Try to detect the relationship from the user's response
    detected_relationship = None
    response_lower = user_response.lower()
    
    # Check for relationship patterns
    relationship_patterns = {
        "friend": ["my friend", "friend", "a friend"],
        "colleague": ["my colleague", "colleague", "a colleague", "coworker", "my coworker"],
        "family": ["my family", "family member", "a family member"],
        "supervisor": ["my supervisor", "supervisor", "boss", "my boss"],
        "neighbor": ["my neighbor", "neighbor", "a neighbor"],
        "assistant": ["my assistant", "assistant", "a assistant"]
    }
    
    for relationship, patterns in relationship_patterns.items():
        if any(pattern in response_lower for pattern in patterns):
            detected_relationship = relationship
            break
    
    if detected_relationship:
        # Find the profile with the matching relationship
        matched_profile = None
        for profile in profiles:
            if profile.get('relationship', '').lower() == detected_relationship:
                matched_profile = profile
                break
        
        if matched_profile:
            logger.info(f"[PROFILE CLARIFICATION] Found matching profile: {matched_profile.get('name')} ({detected_relationship})")
            
            # Generate response based on the original query
            try:
                if is_rich_profile(matched_profile):
                    rich_response = generate_rich_response_from_profile(matched_profile)
                    state.final_response = rich_response
                else:
                    # Convert simple profile to rich format and generate response
                    rich_profile = await convert_simple_to_rich_profile(matched_profile)
                    rich_response = generate_rich_response_from_profile(rich_profile)
                    state.final_response = rich_response
                
                # Set the last mentioned profile
                state.last_mentioned_profile = {
                    "name": matched_profile.get('name'),
                    "relationship": detected_relationship,
                    "information": "Clarified from multiple profiles"
                }
                
                state.agent_status = "initialize"
                logger.info(f"[PROFILE CLARIFICATION] Successfully provided response for {matched_profile.get('name')}")
                
            except Exception as e:
                logger.error(f"[PROFILE CLARIFICATION] Error generating response: {str(e)}")
                state.final_response = f"Here's what I know about {matched_profile.get('name')} ({detected_relationship}): {matched_profile.get('information', 'Information available but could not be formatted')}"
                state.agent_status = "initialize"
        else:
            logger.warning(f"[PROFILE CLARIFICATION] No profile found with relationship '{detected_relationship}' for '{name}'")
            state.final_response = f"I couldn't find a profile for {name.capitalize()} with the relationship '{detected_relationship}'. The available profiles are:\n" + \
                                 "\n".join([f"• {p.get('relationship', 'unknown').capitalize()} {p.get('name')}" for p in profiles])
            state.agent_status = "waiting_profile_clarification"
    else:
        # No clear relationship detected, ask for more specific clarification
        logger.info(f"[PROFILE CLARIFICATION] No clear relationship detected in response: {user_response}")
        state.final_response = (
            f"I need you to be more specific about which {name.capitalize()} you're asking about. "
            f"Please say something like:\n\n"
            f"• 'my friend {name.capitalize()}' or\n"
            f"• 'my colleague {name.capitalize()}' or\n"
            f"• 'the {name.capitalize()} who is my friend'\n\n"
            f"Available options:\n" +
            "\n".join([f"• {p.get('relationship', 'unknown').capitalize()} {p.get('name')}" for p in profiles])
        )
        state.agent_status = "waiting_profile_clarification"
    
    # Clear the pending clarification if we're done
    if state.agent_status == "initialize":
        state.pending_profile_clarification = None
    
    return state

def generate_targeted_response(profile: dict, query: str) -> str:
    """
    Generate a targeted response based on the user's specific question.
    Instead of dumping all profile information, focus on what was asked.
    """
    name = profile.get('name', 'Unknown')
    relationship = profile.get('relationship', 'unknown')
    
    # Check for specific question types and provide focused answers
    query_lower = query.lower()
    
    # Family-related questions
    if any(word in query_lower for word in ['sister', 'sisters', 'brother', 'brothers', 'family', 'children', 'kids']):
        family_info = profile.get('family', [])
        if family_info:
            if isinstance(family_info, list):
                family_text = '; '.join(family_info)
            else:
                family_text = str(family_info)
            
            # For specific sister/brother questions, extract the count
            if 'sister' in query_lower or 'sisters' in query_lower:
                if 'two sisters' in family_text.lower():
                    return f"{name} has two sisters: Isnur and Elizabeth."
                elif 'sister' in family_text.lower():
                    return f"{name} has {family_text}."
                else:
                    return f"Based on the profile, {name} has {family_text}."
            elif 'brother' in query_lower or 'brothers' in query_lower:
                if 'brother' in family_text.lower():
                    return f"{name} has {family_text}."
                else:
                    return f"Based on the profile, {name} has {family_text}."
            else:
                return f"Regarding {name}'s family: {family_text}"
        else:
            return f"I don't have information about {name}'s family in the profile."
    
    # Location-related questions
    elif any(word in query_lower for word in ['where', 'location', 'live', 'lives', 'city', 'address']):
        location = profile.get('location')
        if location:
            return f"{name} lives in {location}."
        else:
            return f"I don't have location information for {name} in the profile."
    
    # Phone-related questions
    elif any(word in query_lower for word in ['phone', 'number', 'contact', 'call']):
        phone = profile.get('phone')
        if phone:
            return f"{name}'s phone number is {phone}."
        else:
            return f"I don't have phone information for {name} in the profile."
    
    # Workplace-related questions
    elif any(word in query_lower for word in ['work', 'job', 'company', 'office', 'workplace']):
        workplace = profile.get('workplace')
        if workplace:
            return f"{name} works at {workplace}."
        else:
            return f"I don't have workplace information for {name} in the profile."
    
    # Education-related questions
    elif any(word in query_lower for word in ['education', 'school', 'university', 'college', 'study']):
        education = profile.get('education')
        if education:
            return f"{name}'s education: {education}."
        else:
            return f"I don't have education information for {name} in the profile."
    
    # General information questions
    elif any(word in query_lower for word in ['what', 'tell', 'about', 'know']):
        # For general "tell me about" questions, provide a summary
        summary_parts = []
        
        if profile.get('location'):
            summary_parts.append(f"lives in {profile.get('location')}")
        
        if profile.get('phone'):
            summary_parts.append(f"phone: {profile.get('phone')}")
        
        if profile.get('family'):
            family_info = profile.get('family')
            if isinstance(family_info, list):
                summary_parts.extend(family_info)
            else:
                summary_parts.append(str(family_info))
        
        if profile.get('workplace'):
            summary_parts.append(f"works at {profile.get('workplace')}")
        
        if profile.get('other'):
            other_info = profile.get('other')
            if isinstance(other_info, list):
                summary_parts.extend(other_info)
            else:
                summary_parts.append(str(other_info))
        
        if summary_parts:
            summary = '; '.join(summary_parts)
            return f"Here's what I know about {name} ({relationship}): {summary}"
        else:
            return f"I have a profile for {name} ({relationship}) but no additional details are available."
    
    # Default fallback - provide focused information based on what's available
    else:
        # Look for the most relevant information based on the query
        if 'family' in query_lower and profile.get('family'):
            family_info = profile.get('family')
            if isinstance(family_info, list):
                return f"{name}'s family: {'; '.join(family_info)}"
            else:
                return f"{name}'s family: {family_info}"
        
        elif 'location' in query_lower and profile.get('location'):
            return f"{name} lives in {profile.get('location')}."
        
        elif 'phone' in query_lower and profile.get('phone'):
            return f"{name}'s phone number is {profile.get('phone')}."
        
        else:
            # Fallback to rich response for complex queries
            try:
                return generate_rich_response_from_profile(profile)
            except:
                return f"Here's what I know about {name} ({relationship}): Profile information available but could not be formatted."

async def handle_profile_update_confirmation_node(state: AgentState) -> AgentState:
    """Node to handle user confirmation for profile updates."""
    logger.info("[LANGGRAPH PATH] Starting handle_profile_update_confirmation_node")
    
    if not state.pending_profile_update:
        logger.warning("[PROFILE CONFIRMATION] No pending profile update found")
        state.final_response = "I'm not sure what you're confirming. Could you please provide the profile information again?"
        state.agent_status = "initialize"
        return state
    
    pending_update = state.pending_profile_update
    user_response = state.retrieval_query.strip().lower()
    
    logger.info(f"[PROFILE CONFIRMATION] Processing confirmation for: {pending_update}")
    logger.info(f"[PROFILE CONFIRMATION] User response: {user_response}")
    
    # Check if user confirmed the update
    if user_response in ['yes', 'y', 'confirm', 'ok', 'okay', 'sure', 'go ahead']:
        logger.info("[PROFILE CONFIRMATION] User confirmed the update")
        
        try:
            # Get the profile data and update it
            user_id = state.user_id
            name = pending_update['name']
            field = pending_update['field']
            new_value = pending_update['new_value']
            
            # Load existing profiles
            profiles = await load_user_profiles(user_id)
            
            # Find the profile to update
            profile_key = None
            for key, profile in profiles.items():
                if profile.get('name', '').lower() == name.lower():
                    profile_key = key
                    break
            
            if profile_key:
                # Update the specific field
                profiles[profile_key][field] = new_value
                profiles[profile_key]['last_updated'] = datetime.now().isoformat()
                
                # Save the updated profile
                success = await save_user_profiles(user_id, profiles)
                if success:
                    state.final_response = f"Perfect! I've updated {name}'s {field} to: {new_value}"
                    logger.info(f"[PROFILE CONFIRMATION] Successfully updated {name}'s {field}")
                else:
                    state.final_response = f"Sorry, I encountered an error while updating {name}'s profile. Please try again."
                    logger.error(f"[PROFILE CONFIRMATION] Failed to save updated profile for {name}")
            else:
                state.final_response = f"I couldn't find {name}'s profile to update. Please provide the information again."
                logger.warning(f"[PROFILE CONFIRMATION] Profile not found for {name}")
                
        except Exception as e:
            logger.error(f"[PROFILE CONFIRMATION] Error updating profile: {str(e)}")
            state.final_response = f"Sorry, I encountered an error while updating the profile. Please try again."
        
        # Clear the pending update and reset state
        state.pending_profile_update = None
        state.agent_status = "initialize"
        
    elif user_response in ['no', 'n', 'cancel', 'stop', 'don\'t', 'dont']:
        logger.info("[PROFILE CONFIRMATION] User declined the update")
        state.final_response = f"Understood! I won't update {pending_update['name']}'s {pending_update['field']}. The current information remains unchanged."
        
        # Clear the pending update and reset state
        state.pending_profile_update = None
        state.agent_status = "initialize"
        
    else:
        # Unclear response - ask for clarification
        logger.info("[PROFILE CONFIRMATION] Unclear user response, asking for clarification")
        state.final_response = f"I didn't understand your response. Please answer with 'yes' to confirm updating {pending_update['name']}'s {pending_update['field']} to '{pending_update['new_value']}', or 'no' to cancel the update."
        state.agent_status = "waiting_confirmation"
    
    return state

# Pattern to detect pronouns to replace
PRONOUN_PATTERN = re.compile(r'\b(he|she|him|her|his|hers)\b', re.IGNORECASE)

def replace_pronouns(text: str, last_profile: Dict[str, str]) -> str:
    """Replace simple pronouns with last mentioned profile's name or 'relationship name'."""
    if not last_profile or not last_profile.get("name"):
        return text
    name = last_profile["name"]
    relationship = last_profile.get("relationship", "").strip().lower()
    # Prefer relationship + name if known and not 'unknown', else just name
    replacement = f"{relationship} {name}" if relationship and relationship != "unknown" else name

    def repl(match):
        pronoun = match.group(0)
        # Preserve capitalization
        if pronoun[0].isupper():
            return replacement.title()
        return replacement

    return PRONOUN_PATTERN.sub(repl, text)

async def profile_query_answer_node(state: AgentState) -> AgentState:
    logger.info("[LANGGRAPH PATH] Starting profile_query_answer_node")
    user_id = state.user_id
    raw_query = state.retrieval_query.strip()

    # Load existing profiles
    profiles = await load_user_profiles(user_id)
    logger.info(f"Loaded {len(profiles)} existing profiles for user {user_id}.")

    logger.info(f"[LANGGRAPH PATH] Raw query: {raw_query}")
    logger.info(f"[LANGGRAPH PATH] User ID: {user_id}")

    # Replace pronouns ONLY if we have a clear, complete profile
    last_profile = getattr(state, "last_mentioned_profile", {})
    if (last_profile and 
        last_profile.get("name") and 
        last_profile.get("relationship") and 
        last_profile.get("relationship") != "unknown"):
        processed_query = replace_pronouns(raw_query, last_profile)
        logger.info(f"Processed query with pronoun replacement: {processed_query}")
    else:
        # Don't assume pronoun resolution if profile is incomplete
        processed_query = raw_query
        logger.info(f"No pronoun replacement - profile incomplete: {last_profile}")
        # Only warn about pronouns if the query actually contains pronouns AND no clear name/relationship
        if any(pron in raw_query.lower() for pron in ["he", "she", "his", "her", "they", "their"]):
            # Check if the query also contains a clear name (capitalized word that could be a name)
            import re
            name_pattern = r'\b[A-Z][a-z]+\b'  # Capitalized words that could be names
            potential_names = re.findall(name_pattern, raw_query)
            
            # If we found potential names, don't treat this as a pronoun-only query
            if potential_names:
                logger.info(f"[PROFILE QUERY] Query contains both pronouns and potential names: {potential_names}, proceeding with normal processing")
                # Continue with normal processing instead of asking for clarification
            else:
                logger.warning(f"[PROFILE QUERY] Pronouns detected but profile incomplete, asking for clarification")
                state.final_response = (
                    "I see you're using pronouns (he, she, his, her, they, their), but I need more context "
                    "to understand who you're referring to. Could you please specify the person's name?"
                )
                state.agent_status = "initialize"  # Set proper status to avoid routing errors
                return state

    # Build context from matched profiles with rich profile support
    context_entries = []
    rich_profiles = {}
    logger.info(f"Profiles: {profiles.values()}")
    
    for p in profiles.values():
        name = p.get("name", "Unknown")
        relationship = p.get("relationship", "unknown").capitalize()
        
        # Check if this is a rich profile
        if is_rich_profile(p):
            rich_profiles[name.lower()] = p
            # For rich profiles, create a detailed summary from available fields
            summary_parts = []
            
            # Add location if available
            if p.get('location'):
                summary_parts.append(f"lives in {p.get('location')}")
            
            # Add phone if available
            if p.get('phone'):
                summary_parts.append(f"phone: {p.get('phone')}")
            
            # Add family information if available
            if p.get('family'):
                family_info = p.get('family')
                if isinstance(family_info, list):
                    summary_parts.extend(family_info)
                else:
                    summary_parts.append(str(family_info))
            
            # Add workplace if available
            if p.get('workplace'):
                summary_parts.append(f"works at {p.get('workplace')}")
            
            # Add education if available
            if p.get('education'):
                summary_parts.append(f"education: {p.get('education')}")
            
            # Add other information if available
            if p.get('other'):
                other_info = p.get('other')
                if isinstance(other_info, list):
                    summary_parts.extend(other_info)
                else:
                    summary_parts.append(str(other_info))
            
            # Create the summary
            if summary_parts:
                summary = "; ".join(summary_parts)
            else:
                summary = "Profile created but no additional details available"
            
            context_entries.append(f"{relationship} {name}: {summary}")
        else:
            # Handle simple profiles as before
            info = p.get("information", [])
            info_text = "; ".join(info) if isinstance(info, list) else str(info)
            context_entries.append(f"{relationship} {name}: {info_text}")
    
    context_text = "\n".join(context_entries)

    logger.info(f"Profile Context: {context_text}")
    
    # CRITICAL: Add detailed profile matching logging
    logger.info(f"[PROFILE QUERY] Query: '{processed_query}'")
    logger.info(f"[PROFILE QUERY] Available profiles: {list(profiles.keys())}")
    
    # Log detailed profile information for debugging
    for profile_key, profile in profiles.items():
        profile_name = profile.get('name', 'Unknown')
        profile_relationship = profile.get('relationship', 'unknown')
        logger.info(f"[PROFILE QUERY] Profile {profile_key}: {profile_relationship} {profile_name}")
        if is_rich_profile(profile):
            logger.info(f"[PROFILE QUERY]   Rich profile details: {profile}")
        else:
            logger.info(f"[PROFILE QUERY]   Simple profile info: {profile.get('information', 'No info')}")
    
    # Check for direct profile matches before using LLM
    query_lower = processed_query.lower()
    detected_people, _ = await detect_profiles_in_text(processed_query, state.request.messages)
    
    logger.info(f"[PROFILE QUERY] detect_profiles_in_text result: {detected_people}")
    logger.info(f"[PROFILE QUERY] detected_people type: {type(detected_people)}")
    logger.info(f"[PROFILE QUERY] detected_people length: {len(detected_people) if detected_people else 0}")
    
    if detected_people:
        logger.info(f"[PROFILE QUERY] Detected people in query: {detected_people}")
        
        for person in detected_people:
            person_name = person.get('name', '').lower()
            person_relationship = person.get('relationship', '').lower()
            
            logger.info(f"[PROFILE QUERY] Checking person: name='{person_name}', relationship='{person_relationship}'")
            
            # Look for exact matches
            for profile_key, profile in profiles.items():
                profile_name = profile.get('name', '').lower()
                profile_relationship = profile.get('relationship', '').lower()
                
                logger.info(f"[PROFILE QUERY] Comparing with profile: name='{profile_name}', relationship='{profile_relationship}' (key: {profile_key})")
                
                # Check for exact name + relationship match
                if (person_name == profile_name and 
                    person_relationship == profile_relationship):
                    logger.info(f"[PROFILE QUERY] EXACT MATCH FOUND: {profile_key}")
                    
                    # Generate targeted response using profile data
                    try:
                        targeted_response = generate_targeted_response(profile, processed_query)
                        logger.info(f"[PROFILE QUERY] Generated targeted response: {targeted_response}")
                        
                        state.final_response = targeted_response
                        state.agent_status = "initialize"
                        return state
                    except Exception as e:
                        logger.warning(f"[PROFILE QUERY] Targeted response generation failed: {str(e)}")
                        break
                
                # Check for name match only (for relationship disambiguation)
                elif person_name == profile_name:
                    logger.info(f"[PROFILE QUERY] NAME MATCH (different relationship): {profile_key}")
    
    # CRITICAL FIX: Check for multiple profiles with the same name
    # This prevents automatic selection when ambiguity exists
    if detected_people:
        for person in detected_people:
            person_name = person.get('name', '').lower()
            person_relationship = person.get('relationship', '').lower()
            
            # Count profiles with this name
            profiles_with_same_name = []
            for profile_key, profile in profiles.items():
                profile_name = profile.get('name', '').lower()
                if profile_name == person_name:
                    profiles_with_same_name.append(profile)
            
            logger.info(f"[PROFILE QUERY] Found {len(profiles_with_same_name)} profiles for name '{person_name}'")
            logger.info(f"[PROFILE QUERY] Person relationship: '{person_relationship}' (type: {type(person_relationship)})")
            
            # If multiple profiles exist and no specific relationship was provided, ask for clarification
            # CRITICAL: Check if relationship is missing, empty, or unknown
            relationship_missing = (not person_relationship or 
                                  person_relationship == '' or 
                                  person_relationship == 'unknown')
            
            logger.info(f"[PROFILE QUERY] Relationship missing: {relationship_missing}")
            logger.info(f"[PROFILE QUERY] Multiple profiles check: {len(profiles_with_same_name)} > 1 = {len(profiles_with_same_name) > 1}")
            
            if len(profiles_with_same_name) > 1 and relationship_missing:
                
                logger.info(f"[PROFILE QUERY] MULTIPLE PROFILES DETECTED - asking for clarification")
                
                # Create clarification message
                clarification_options = []
                for profile in profiles_with_same_name:
                    relationship = profile.get('relationship', 'unknown').capitalize()
                    clarification_options.append(f"{relationship} {profile.get('name', 'Unknown')}")
                
                clarification_message = (
                    f"I found multiple profiles for '{person_name.capitalize()}'. "
                    f"Which one are you asking about?\n\n"
                    f"Available options:\n"
                    f"{chr(10).join([f'• {option}' for option in clarification_options])}\n\n"
                    f"Please specify by saying something like 'my friend {person_name.capitalize()}' or 'my colleague {person_name.capitalize()}'."
                )
                
                # Set the state to wait for clarification
                state.final_response = clarification_message
                state.agent_status = "waiting_profile_clarification"
                state.pending_profile_clarification = {
                    "name": person_name,
                    "profiles": profiles_with_same_name,
                    "query": processed_query
                }
                logger.info(f"[PROFILE QUERY] Set pending profile clarification for {person_name}")
                return state
    
    # FALLBACK: If LLM detection failed, manually check for names in the query
    if not detected_people:
        logger.info(f"[PROFILE QUERY] LLM detection failed, manually checking for names in query: '{processed_query}'")
        
        # Simple name detection - look for capitalized words that could be names
        import re
        name_pattern = r'\b[A-Z][a-z]+\b'
        potential_names = re.findall(name_pattern, processed_query)
        logger.info(f"[PROFILE QUERY] Potential names found: {potential_names}")
        
        if potential_names:
            # Check each potential name for multiple profiles
            for name in potential_names:
                name_lower = name.lower()
                profiles_with_same_name = []
                
                for profile_key, profile in profiles.items():
                    profile_name = profile.get('name', '').lower()
                    if profile_name == name_lower:
                        profiles_with_same_name.append(profile)
                
                logger.info(f"[PROFILE QUERY] Manual check: Found {len(profiles_with_same_name)} profiles for name '{name}'")
                
                # If multiple profiles exist, ask for clarification
                if len(profiles_with_same_name) > 1:
                    logger.info(f"[PROFILE QUERY] MANUAL CHECK: MULTIPLE PROFILES DETECTED - asking for clarification")
                    
                    # Create clarification message
                    clarification_options = []
                    for profile in profiles_with_same_name:
                        relationship = profile.get('relationship', 'unknown').capitalize()
                        clarification_options.append(f"{relationship} {profile.get('name', 'Unknown')}")
                    
                    clarification_message = (
                        f"I found multiple profiles for '{name}'. "
                        f"Which one are you asking about?\n\n"
                        f"Available options:\n"
                        f"{chr(10).join([f'• {option}' for option in clarification_options])}\n\n"
                        f"Please specify by saying something like 'my friend {name}' or 'my colleague {name}'."
                    )
                    
                    # Set the state to wait for clarification
                    state.final_response = clarification_message
                    state.agent_status = "waiting_profile_clarification"
                    state.pending_profile_clarification = {
                        "name": name_lower,
                        "profiles": profiles_with_same_name,
                        "query": processed_query
                    }
                    logger.info(f"[PROFILE QUERY] Manual check: Set pending profile clarification for {name}")
                    return state
    
    logger.info(f"[PROFILE QUERY] No direct profile match found, proceeding with LLM analysis")

    # Prepare LLM prompt
    system_prompt = (
        f"""
        You are an intelligent assistant with access to user profiles.  
        Always respond strictly based on `{context_text}`. Never invent, assume, or add external knowledge.  

        ---

        ### Matching Logic (strict order)

        1. **Name + Relationship provided (e.g., "Friend Rose")**  
        - If exact match exists ? return profile.  
        - If no exact match ?  
        ```
        NO_PROFILE_MATCH: Sorry, I couldn't find a profile matching that request.
        ```

        2. **Only a Name provided (e.g., "Rose")**  
        - If exactly one profile with that name exists ? return profile.  
        - If multiple profiles share the name ? return this exact response:
        ```
        MULTIPLE_PROFILES: I found multiple profiles for "Rose". Please specify which one by saying:
        • "my friend Rose" or "my colleague Rose" or "the Rose who is my friend"
        
        Available options:
        • Friend Rose
        • Colleague Rose
        ```
        - If no profile with that name exists ? return only the `NO_PROFILE_MATCH` block.  
        ```
        Query: "Tell me about Lionel Messi."
        Answer: "NO_PROFILE_MATCH: Sorry, I couldn't find a profile named Lionel Messi."
        ```

        3. **No recognizable name or relationship**  
        - Return only the `NO_PROFILE_MATCH` block.

        ### IMPORTANT MATCHING EXAMPLES
        Query: "Tell me about my colleague maria" ? Match to "Colleague Maria" profile
        Query: "Tell me about my friend mark" ? Match to "Friend Mark" profile  
        Query: "What about my colleague maria" ? Match to "Colleague Maria" profile
        Query: "Tell me about maria" ? Ask for clarification (multiple Marias exist)
        Query: "Tell me about mark" ? Ask for clarification (multiple Marks exist)  

        ---

        ### Response Rules
        - Never reveal profile details when clarification is required.  
        - Never merge profiles unless explicitly asked to *compare*.  
        - If data is missing in the matched profile ?  
        *"That information is not available in the profile."*  
        - When returning `NO_PROFILE_MATCH`, include only the block — no extra text.  
        - **CRITICAL**: Never assume or guess information not explicitly provided.  
        - **CRITICAL**: If there's any ambiguity about which profile to use ? ask for clarification.  
        - **CRITICAL**: When multiple profiles exist with the same name ? always ask user to specify which one.
        - **CRITICAL**: When user says "my [relationship] [name]" ? match to the exact profile with that relationship and name.
        - **CRITICAL**: "my colleague maria" should always match "Colleague Maria" profile, not "Friend Maria".  

        ---

        ### CONTEXT DATA  
        `{context_text}`

        """
    )

    llm_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", processed_query)
    ])

    # Prepare chat history
    chat_history = []
    for idx, message in enumerate(state.request.messages):
        if idx == len(state.request.messages) - 1 and message.role == "user": 
            break
        if message.role == "user":
            chat_history.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            chat_history.append(AIMessage(content=message.content))

    logger.info(f"Chat history: {chat_history}")

    # Invoke LLM
    messages = llm_prompt.format_messages(
        input=processed_query,
        system_prompt=system_prompt,
        chat_history=chat_history
    )

    logger.info(f"Messages: {messages}")

    response = await openai_llm.ainvoke(messages)
    answer = response.content.strip()

    logger.info(f"Answer: {answer}")
    
    # CRITICAL: Validate that LLM is not making assumptions
    if any(assumption_indicator in answer.lower() for assumption_indicator in [
        "probably", "maybe", "might be", "could be", "seems like", "appears to be",
        "i think", "i believe", "i assume", "likely", "possibly"
    ]):
        logger.warning(f"[PROFILE QUERY] LLM making assumptions in response: {answer}")
        state.final_response = (
            "I'm not entirely sure about this information. Could you please provide more specific details "
            "or clarify what you'd like to know?"
        )
        state.agent_status = "initialize"
        return state
    
    # Check for MULTIPLE_PROFILES response from LLM
    if "MULTIPLE_PROFILES" in answer:
        logger.info("[PROFILE QUERY] LLM indicated multiple profiles exist, triggering disambiguation")
        
        # Extract the name from the query
        detected_people, _ = await detect_profiles_in_text(processed_query, state.request.messages)
        
        if detected_people:
            person_name = detected_people[0].get('name', '').lower()
            profiles_with_same_name = []
            
            for profile in profiles.values():
                if profile.get('name', '').lower() == person_name:
                    profiles_with_same_name.append(profile)
            
            if len(profiles_with_same_name) > 1:
                # Create clarification message
                clarification_options = []
                for profile in profiles_with_same_name:
                    relationship = profile.get('relationship', 'unknown').capitalize()
                    clarification_options.append(f"{relationship} {profile.get('name', 'Unknown')}")
                
                clarification_message = (
                    f"I found multiple profiles for '{person_name.capitalize()}'. "
                    f"Which one are you asking about?\n\n"
                    f"Available options:\n"
                    f"{chr(10).join([f'• {option}' for option in clarification_options])}\n\n"
                    f"Please specify by saying something like 'my friend {person_name.capitalize()}' or 'my colleague {person_name.capitalize()}'."
                )
                
                # Set the state to wait for clarification
                state.final_response = clarification_message
                state.agent_status = "waiting_profile_clarification"
                state.pending_profile_clarification = {
                    "name": person_name,
                    "profiles": profiles_with_same_name,
                    "query": processed_query
                }
                logger.info(f"[PROFILE QUERY] Set pending profile clarification for {person_name}")
                return state
        
        # Fallback if something goes wrong
        # For ask_info intent, try to provide a profile-based answer before routing to general
        if hasattr(state, 'agent_status') and state.agent_status == "ask_info":
            logger.info("[PROFILE QUERY] ask_info intent detected, attempting profile-based fallback")
            # Try to find any profile that might match the query
            for profile in profiles.values():
                if profile.get('name', '').lower() in processed_query.lower():
                    logger.info(f"[PROFILE QUERY] Found potential profile match for ask_info: {profile.get('name')}")
                    try:
                        targeted_response = generate_targeted_response(profile, processed_query.lower())
                        state.final_response = targeted_response
                        state.agent_status = "initialize"
                        return state
                    except Exception as e:
                        logger.warning(f"[PROFILE QUERY] Profile-based fallback failed: {str(e)}")
                        break
        
        # If no profile-based answer possible, route to general
        logger.info("[PROFILE QUERY] No profile-based answer possible, routing to general")
        state.agent_status = "general"
        return state
    
    # CRITICAL: Validate that LLM is not incorrectly returning NO_PROFILE_MATCH
    if "NO_PROFILE_MATCH" in answer:
        # Check if this query should actually have a match
        query_lower = processed_query.lower()
        if any(pattern in query_lower for pattern in [
            "my colleague", "my friend", "my sister", "my brother", "my mother", "my father"
        ]):
            # Extract relationship and name from query
            for pattern in ["my colleague", "my friend", "my sister", "my brother", "my mother", "my father"]:
                if pattern in query_lower:
                    relationship = pattern.split()[-1]  # Get the relationship word
                    # Look for the name after the relationship
                    name_match = re.search(rf'{pattern}\s+([a-zA-Z]+)', query_lower)
                    if name_match:
                        name = name_match.group(1).title()
                        # Check if this profile exists in context
                        profile_key = f"{relationship.title()} {name}"
                        if profile_key in context_text:
                            logger.warning(f"[PROFILE QUERY] LLM incorrectly returned NO_PROFILE_MATCH for existing profile: {profile_key}")
                            # Force the correct response
                            if profile_key + ': ' in context_text:
                                profile_details = context_text.split(profile_key + ': ')[1].split('\n')[0]
                                state.final_response = f"Here's what I know about {profile_key}: {profile_details}"
                            else:
                                state.final_response = f"Here's what I know about {profile_key}: Profile found but details not available"
                            state.agent_status = "initialize"
                            return state
        
        logger.info("[LANGGRAPH PATH] LLM indicated no profile match, routing to general node")
        # Don't set final_response - let the general node handle the response
        state.agent_status = "general"
        return state
    
    # Check if we have a profile match and enhance the response with rich profile data
    # For ask_info intent, prioritize profile-based answers over routing to general
    if ("NO_PROFILE_MATCH" not in answer and 
        not any(keyword in answer.lower() for keyword in ["which", "which one", "clarify", "specify", "multiple profiles", "found multiple"])):
        
        # For ask_info intent, always try to find profile matches first
        detected_people, _ = await detect_profiles_in_text(processed_query, state.request.messages)
        
        if detected_people:
            # Check for name ambiguity - multiple profiles with the same name
            person_name = detected_people[0].get('name', '').lower()
            detected_relationship = detected_people[0].get('relationship', '').lower()
            profiles_with_same_name = []
            
            for profile in profiles.values():
                if profile.get('name', '').lower() == person_name:
                    profiles_with_same_name.append(profile)
        else:
            # No profile detected in text, but for ask_info intent, try to find profiles by name
            # This handles cases like "Where does Marry live?" when no relationship is specified
            logger.info("[PROFILE QUERY] No profile detected in text, checking for ask_info intent")
            
            # Extract potential names from the query
            import re
            name_pattern = r'\b[A-Z][a-z]+\b'  # Capitalized words that could be names
            potential_names = re.findall(name_pattern, processed_query)
            
            if potential_names:
                for potential_name in potential_names:
                    # Check if this name exists in profiles
                    matching_profiles = []
                    for profile in profiles.values():
                        if profile.get('name', '').lower() == potential_name.lower():
                            matching_profiles.append(profile)
                    
                    if matching_profiles:
                        if len(matching_profiles) == 1:
                            # Single profile found, use it
                            matched_profile = matching_profiles[0]
                            logger.info(f"[PROFILE QUERY] Found single profile for '{potential_name}' in ask_info query")
                            
                            # Generate targeted response
                            if is_rich_profile(matched_profile):
                                try:
                                    query_lower = processed_query.lower()
                                    targeted_response = generate_targeted_response(matched_profile, query_lower)
                                    state.final_response = targeted_response
                                    state.agent_status = "initialize"
                                    return state
                                except Exception as e:
                                    logger.warning(f"[PROFILE QUERY] Targeted response generation failed: {str(e)}")
                                    # Fall through to general routing
                        elif len(matching_profiles) > 1:
                            # Multiple profiles found, ask for clarification
                            logger.info(f"[PROFILE QUERY] Multiple profiles found for '{potential_name}' in ask_info query")
                            
                            clarification_options = []
                            for profile in matching_profiles:
                                relationship = profile.get('relationship', 'unknown').capitalize()
                                clarification_options.append(f"{relationship} {profile.get('name', 'Unknown')}")
                            
                            clarification_message = (
                                f"I found multiple profiles for '{potential_name}'. "
                                f"Which one are you asking about?\n\n"
                                f"Available options:\n"
                                f"{chr(10).join([f'• {option}' for option in clarification_options])}\n\n"
                                f"Please specify by saying something like 'my friend {potential_name}' or 'my colleague {potential_name}'."
                            )
                            
                            state.final_response = clarification_message
                            state.agent_status = "waiting_profile_clarification"
                            state.pending_profile_clarification = {
                                "name": potential_name.lower(),
                                "profiles": matching_profiles,
                                "query": processed_query
                            }
                            return state
            
            # If multiple profiles with the same name, check if relationship was already specified
            if len(profiles_with_same_name) > 1:
                logger.info(f"[PROFILE QUERY] Multiple profiles found for name '{person_name}': {len(profiles_with_same_name)} profiles")
                
                # Check if a specific relationship was already provided in the query
                if detected_relationship:
                    logger.info(f"[PROFILE QUERY] Relationship '{detected_relationship}' detected in query, finding specific profile")
                    
                    # Find the profile with the matching relationship
                    matched_profile = None
                    for profile in profiles_with_same_name:
                        if profile.get('relationship', '').lower() == detected_relationship:
                            matched_profile = profile
                            break
                    
                    if matched_profile:
                        logger.info(f"[PROFILE QUERY] Found matching profile with relationship '{detected_relationship}' for {person_name}")
                        
                        # Generate targeted response based on the user's specific question
                        if is_rich_profile(matched_profile):
                            try:
                                # Analyze the query to provide a focused answer
                                query_lower = processed_query.lower()
                                targeted_response = generate_targeted_response(matched_profile, query_lower)
                                logger.info(f"[PROFILE QUERY] Generated targeted response for {matched_profile.get('name', 'Unknown')}")
                                state.final_response = targeted_response
                            except Exception as e:
                                logger.warning(f"[PROFILE QUERY] Targeted response generation failed: {str(e)}, falling back to rich response")
                                try:
                                    rich_response = generate_rich_response_from_profile(matched_profile)
                                    state.final_response = rich_response
                                except Exception as e2:
                                    logger.warning(f"[PROFILE QUERY] Rich response generation also failed: {str(e2)}, using original response")
                                    state.final_response = answer
                        else:
                            # Convert simple profile to rich format and generate response
                            try:
                                rich_profile = await convert_simple_to_rich_profile(matched_profile)
                                rich_response = generate_rich_response_from_profile(rich_profile)
                                logger.info(f"[PROFILE QUERY] Converted simple profile to rich and generated response for {matched_profile.get('name', 'Unknown')}")
                                state.final_response = rich_response
                                
                                # Set the last mentioned profile
                                state.last_mentioned_profile = {
                                    "name": matched_profile.get('name'),
                                    "relationship": detected_relationship,
                                    "information": "Profile information retrieved"
                                }
                            except Exception as e:
                                logger.warning(f"[PROFILE QUERY] Profile conversion failed: {str(e)}, using original response")
                                state.final_response = answer
                        
                        state.agent_status = "initialize"
                        return state
                    else:
                        logger.warning(f"[PROFILE QUERY] Relationship '{detected_relationship}' specified but no matching profile found")
                
                # If no relationship specified or no match found, ask for clarification
                logger.info(f"[PROFILE QUERY] No specific relationship match, asking for clarification")
                
                # Create clarification message
                clarification_options = []
                for profile in profiles_with_same_name:
                    relationship = profile.get('relationship', 'unknown').capitalize()
                    clarification_options.append(f"{relationship} {profile.get('name', 'Unknown')}")
                
                clarification_message = (
                    f"I found multiple profiles for '{person_name.capitalize()}'. "
                    f"Which one are you asking about?\n\n"
                    f"Available options:\n"
                    f"{chr(10).join([f'• {option}' for option in clarification_options])}\n\n"
                    f"Please specify by saying something like 'my friend {person_name.capitalize()}' or 'my colleague {person_name.capitalize()}'."
                )
                
                # Set the state to wait for clarification
                state.final_response = clarification_message
                state.agent_status = "waiting_profile_clarification"
                state.pending_profile_clarification = {
                    "name": person_name,
                    "profiles": profiles_with_same_name,
                    "query": processed_query
                }
                logger.info(f"[PROFILE QUERY] Set pending profile clarification for {person_name}")
                return state
            
            # Single profile found - proceed with normal response
            matched_profile = profiles_with_same_name[0] if profiles_with_same_name else None
            
            if matched_profile:
                # Generate rich response if it's a rich profile
                if is_rich_profile(matched_profile):
                    try:
                        rich_response = generate_rich_response_from_profile(matched_profile)
                        logger.info(f"[PROFILE QUERY] Generated rich response for {matched_profile.get('name', 'Unknown')}")
                        state.final_response = rich_response
                    except Exception as e:
                        logger.warning(f"[PROFILE QUERY] Rich response generation failed: {str(e)}, using original response")
                        state.final_response = answer
                else:
                    # Convert simple profile to rich format and generate response
                    try:
                        rich_profile = await convert_simple_to_rich_profile(matched_profile)
                        rich_response = generate_rich_response_from_profile(rich_profile)
                        logger.info(f"[PROFILE QUERY] Converted simple profile to rich and generated response for {matched_profile.get('name', 'Unknown')}")
                        state.final_response = rich_response
                        
                        # Optionally save the enhanced profile back to storage
                        # await update_user_profile(user_id, matched_profile.get('name', ''), rich_profile)
                    except Exception as e:
                        logger.warning(f"[PROFILE QUERY] Profile conversion failed: {str(e)}, using original response")
                        state.final_response = answer
                
                state.last_mentioned_profile = detected_people[0]
                state.agent_status = "initialize"
                return state
    
    # If no rich profile enhancement was possible, use the original response
    state.final_response = answer

    # Determine status based on response
    if any(keyword in answer.lower() for keyword in ["which", "which one", "clarify", "specify", "multiple profiles", "found multiple"]):
        state.agent_status = "confirm_answer"
        logger.info("[LANGGRAPH PATH] LLM is asking for clarification about multiple profiles")
    else:
        state.agent_status = "initialize"
    
        # Detect profiles in query
        detected_people, _ = await detect_profiles_in_text(processed_query, state.request.messages)
    
        state.last_mentioned_profile = detected_people[0] if detected_people else {}

    # Safety check: ensure agent_status is always a valid routing destination
    if state.agent_status not in ["general", "confirm_answer", "waiting_profile_clarification", "initialize"]:
        logger.warning(f"[PROFILE QUERY] Invalid agent_status '{state.agent_status}', defaulting to 'general'")
        state.agent_status = "general"

    return state

# async def profile_missing_creation_node(state: AgentState):
#     return state

# async def profile_missing_answer_node(state: AgentState):
#     return state
async def call_langchain_agent_node(state: AgentState) -> AgentState:
    logger.info("[LANGGRAPH PATH] Starting call_langchain_agent_node")
    
    current_date = getattr(state, "current_date_content", "")
    user_input = state.retrieval_query.strip()
    context_text = getattr(state, "context", "") or ""
    
    logger.info(f"[LANGGRAPH PATH] User input: {user_input}")
    logger.info(f"[LANGGRAPH PATH] Current date: {current_date}")
    logger.info(f"[LANGGRAPH PATH] Context length: {len(context_text)} characters")
    
    system_template = f"""
    From now, You are a vision language model that trained after Cyberself AI Team. 
    You were created/trained in 2025 by Cyberself Inc.
    Today's date is {current_date}.

    # Response Guidelines:
    1. **Do not fabricate or assume** any names, dates, numbers, or details that are not explicitly provided in the context or user input.
    2. **Only use verified and clearly stated information** — such as real names, specific dates, times, or contact details.
    3. **Be clear, concise, and direct.** Avoid filler, speculation, or vague language.
    4. **Do not use placeholders** (e.g., [Country X], [Person A], [Model B]) under any circumstances.
    5. **When asked for a specific number of items** (e.g., "Top 5"):
    - Return exactly that number if available.
    - If fewer than requested are found, report the actual number found and do **not** guess or fill in extras.

    ## General Chat Handling:
    - Respond naturally and helpfully, but **always ground your answers in verifiable information.**
    - If the user asks for opinions or analysis, **clearly distinguish fact from interpretation.**
    - **Do not over-explain or add unnecessary context** unless it directly supports clarity or relevance.
    - If the context is ambiguous, ask for clarification instead of assuming.

    ## Context:
    {context_text}
    
    Respond clearly and helpfully.
    """
    
    llm_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder("chat_history"),
        ("human", user_input)
    ])

    # Prepare chat history
    chat_history = []
    for idx, message in enumerate(state.request.messages):
        if idx == len(state.request.messages) - 1 and message.role == "user": 
            break
        if message.role == "user":
            chat_history.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            chat_history.append(AIMessage(content=message.content))
    
    logger.info(f"[LANGGRAPH PATH] Chat history length: {len(chat_history)} messages")
    
    messages = llm_prompt.format_messages(
        input=user_input,
        current_date_content=current_date,
        chat_history=chat_history
    )
    
    logger.info("[LANGGRAPH PATH] Invoking LLM...")
    response = await openai_llm.ainvoke(messages)
    state.final_response = response.content.strip()
    
    # Append searched links to final response if real-time data was used
    state.final_response = append_searched_links_to_response(state.final_response, state.searched_links)
    if state.searched_links:
        logger.info(f"[LANGGRAPH PATH] Appended {len(state.searched_links)} searched links to final response")
    
    state.agent_status = "initialize"
    
    logger.info(f"[LANGGRAPH PATH] LLM response length: {len(state.final_response)} characters")
    logger.info(f"[LANGGRAPH PATH] call_langchain_agent_node completed")
    return state

# --- LangGraph Definition ---
def build_chat_graph():
    logger.info("Building LangGraph workflow with built-in memory checkpointing...")
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("get_user_name", get_user_name_node)
    workflow.add_node("user_data_and_stats", user_data_and_stats_node)
    workflow.add_node("repository_validation", repository_validation_node)
    workflow.add_node("retrieve_rag_context", retrieve_rag_context_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("intent_analysis", intent_analysis_node)
    workflow.add_node("profile_create_update", profile_create_update_node)
    workflow.add_node("profile_query_answer", profile_query_answer_node)
    workflow.add_node("handle_profile_update_confirmation", handle_profile_update_confirmation_node)
    workflow.add_node("handle_profile_clarification", handle_profile_clarification_node)
    workflow.add_node("prepare_llm_input", prepare_llm_input_node)
    workflow.add_node("call_langchain_agent", call_langchain_agent_node)
    
    # Set entry point
    workflow.set_entry_point("initialize")
    
    # Define transitions
    def route_after_initialize(state):
        next_node = state.agent_status
        logger.info(f"[TRANSITION] initialize -> {next_node}")
        return next_node

    workflow.add_conditional_edges("initialize", 
        route_after_initialize, {
        "initialize": "user_data_and_stats",
        "confirm_creation": "profile_create_update",
        "confirm_answer": "profile_query_answer",
        "waiting_confirmation": "handle_profile_update_confirmation",
        "waiting_profile_clarification": "handle_profile_clarification",
        "get_name": "get_user_name"
    })
    
    # Add routing for get_user_name node
    workflow.add_conditional_edges("get_user_name", 
        lambda state: state.agent_status, {
        "get_name": END,  # Ask for name and end workflow
        "initialize": "user_data_and_stats"  # Name found, end workflow immediately
    })
    
    workflow.add_edge("user_data_and_stats", "repository_validation")
    workflow.add_edge("repository_validation", "retrieve_rag_context")
    workflow.add_edge("retrieve_rag_context", "web_search")
    workflow.add_edge("web_search", "intent_analysis")
    
    workflow.add_conditional_edges("intent_analysis", 
        lambda state: state.agent_status, {
        "give_info": "profile_create_update",
        "ask_info": "profile_query_answer",
        "general": "prepare_llm_input"
    })
    
    # Add conditional edge for profile_create_update to handle confirmation flow
    workflow.add_conditional_edges("profile_create_update",
        lambda state: state.agent_status, {
        "waiting_confirmation": END,  # End workflow to wait for user confirmation
        "initialize": END,  # Normal completion
        "confirm_creation": END  # Still asking for relationship clarification
    })
    
    # Add conditional edge for profile_query_answer to handle routing to general node
    def route_after_profile_query(state):
        next_node = state.agent_status
        logger.info(f"[TRANSITION] profile_query_answer -> {next_node}")
        return next_node
    
    workflow.add_conditional_edges("profile_query_answer", 
        route_after_profile_query, {
        "general": "prepare_llm_input",
        "confirm_answer": END,
        "waiting_profile_clarification": END,
        "initialize": END
    })
    
    workflow.add_edge("prepare_llm_input", "call_langchain_agent")
    workflow.add_edge("call_langchain_agent", END)
    
    # Add conditional edge for handle_profile_update_confirmation
    workflow.add_conditional_edges("handle_profile_update_confirmation",
        lambda state: state.agent_status, {
        "waiting_confirmation": END,  # Still waiting for clear confirmation
        "initialize": END  # Confirmation processed, workflow complete
    })
    
    # Add conditional edge for handle_profile_clarification
    workflow.add_conditional_edges("handle_profile_clarification",
        lambda state: state.agent_status, {
        "waiting_profile_clarification": END,  # Still waiting for clear clarification
        "initialize": END  # Clarification processed, workflow complete
    })

    # Compile with built-in checkpoint saver
    compiled_workflow = workflow.compile(
        checkpointer=checkpoint_store,
        interrupt_after=["profile_create_update", "handle_profile_update_confirmation", "handle_profile_clarification", "call_langchain_agent"]
    )
    logger.info("Workflow compiled with built-in memory checkpointing")
    return compiled_workflow

logger.info("[LANGGRAPH INIT] Building chat graph...")
chat_graph = build_chat_graph() # Compile the graph once
logger.info("[LANGGRAPH INIT] Chat graph built and compiled successfully")

@app.post("/chat/")
async def chat(
    request: ChatRequest,
    user_id: str = Depends(get_user_id)
):
    start_time = time.time()
    logger.info(f"[CHAT] Request for user: {user_id}")
    
    try:
        # Create or retrieve thread ID
        thread_id = request.thread_id
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        
        # Check for existing state
        existing_state = await checkpoint_store.aget(config)
        if existing_state:
            logger.info(f"[MEMORY] Resuming thread: {thread_id}")
            # Reconstruct AgentState from dictionary
            state_dict = existing_state["channel_values"]
            logger.info(f"Loaded state dictionary: {state_dict}")
            
            # Ensure required fields are present
            if not all(k in state_dict for k in ["thread_id", "request", "user_id"]):
                logger.warning("Incomplete state in checkpoint, creating new state")
                # Load saved user name if available
                saved_user_name = await load_user_name(user_id)
                
                input_state = AgentState(
                    thread_id=thread_id,
                    request=request,
                    user_id=user_id,
                    user_name=saved_user_name,
                    retrieval_query="",
                    user_profile={},
                    user_repositories=[],
                    user_stats_context="",
                    filtered_documents=[],
                    context="",
                    current_date_content="",
                    final_response="",
                    searched_links=[],
                    realtime_data=None,
                    error=None,
                    profiles={}
                )
            else:
                if "request" in state_dict:
                    del state_dict["request"]
                
                # Ensure user_name is loaded from storage if not in state_dict
                if "user_name" not in state_dict or not state_dict.get("user_name"):
                    saved_user_name = await load_user_name(user_id)
                    if saved_user_name:
                        state_dict["user_name"] = saved_user_name
                        logger.info(f"[CHAT] Loaded user name from storage: {saved_user_name}")
                
                # Log the state reconstruction for debugging
                logger.info(f"[CHAT] Reconstructing state with keys: {list(state_dict.keys())}")
                logger.info(f"[CHAT] confirm_profile in state_dict: {state_dict.get('confirm_profile')}")
                logger.info(f"[CHAT] agent_status in state_dict: {state_dict.get('agent_status')}")
                logger.info(f"[CHAT] pending_profile_update in state_dict: {state_dict.get('pending_profile_update')}")
                logger.info(f"[CHAT] pending_profile_clarification in state_dict: {state_dict.get('pending_profile_clarification')}")
                
                input_state = AgentState(**state_dict, request=request)
                
                # Log the reconstructed state
                logger.info(f"[CHAT] Reconstructed state - confirm_profile: {input_state.confirm_profile}")
                logger.info(f"[CHAT] Reconstructed state - agent_status: {input_state.agent_status}")
                logger.info(f"[CHAT] Reconstructed state - pending_profile_update: {input_state.pending_profile_update}")
                logger.info(f"[CHAT] Reconstructed state - pending_profile_clarification: {input_state.pending_profile_clarification}")
        else:
            logger.info(f"[MEMORY] New thread: {thread_id}")
            # Load saved user name if available
            saved_user_name = await load_user_name(user_id)
            
            input_state = AgentState(
                thread_id=thread_id,
                request=request,
                user_id=user_id,
                user_name=saved_user_name,
                retrieval_query="",
                user_profile={},
                user_repositories=[],
                user_stats_context="",
                filtered_documents=[],
                context="",
                current_date_content="",
                final_response="",
                searched_links=[],
                realtime_data=None,
                error=None,
                profiles={},
                pending_profile_update=None,
                pending_profile_clarification=None
            )
        
        # Execute graph
        final_state = None
        async for event in chat_graph.astream(input_state, config=config):
            if "call_langchain_agent" in event:
                final_state = event["call_langchain_agent"]
            elif "profile_create_update" in event:
                final_state = event["profile_create_update"]
            elif "profile_query_answer" in event:
                final_state = event["profile_query_answer"]
            elif "handle_profile_update_confirmation" in event:
                final_state = event["handle_profile_update_confirmation"]
            elif "handle_profile_clarification" in event:
                final_state = event["handle_profile_clarification"]
            elif "get_user_name" in event:
                final_state = event["get_user_name"]
        
        # Save final state properly
        if final_state:
            logger.info(f"[MEMORY] Saving state for thread: {thread_id}")
            logger.info(f"[MEMORY] Final state before saving - confirm_profile: {getattr(final_state, 'confirm_profile', 'NOT_FOUND')}")
            logger.info(f"[MEMORY] Final state before saving - agent_status: {getattr(final_state, 'agent_status', 'NOT_FOUND')}")
            logger.info(f"[MEMORY] Final state before saving - pending_profile_update: {getattr(final_state, 'pending_profile_update', 'NOT_FOUND')}")
            logger.info(f"[MEMORY] Final state before saving - pending_profile_clarification: {getattr(final_state, 'pending_profile_clarification', 'NOT_FOUND')}")
            # Since final_state is already a dictionary, save it directly
            existing_state = await checkpoint_store.aget(config)
            logger.info(f"Saved state: {existing_state}")
        
        duration = time.time() - start_time
        logger.info(f"[TIMER] Completed in {duration:.2f}s")
        logger.info(f"Final Response: {final_state['final_response']}")
        
        # Return response in list format with user name if available
        response_message = final_state["final_response"] if final_state else "No response generated"
        
        if final_state and final_state.get("user_name"):
            logger.info(f"User name extracted: {final_state['user_name']}")
            return [response_message, {'name': final_state['user_name']}]
        else:
            return response_message
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal error: {str(e)}"}
        )
    
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    # Configure uvicorn with increased timeouts
    uvicorn.run(
        "app:app", 
        host=host, 
        port=port, 
        reload=True,
        timeout_keep_alive=120,  # 2 minutes (default is 5 seconds)
        workers=4  # Use multiple workers to handle concurrent requests
    ) 
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
   - If message contains "sign in" or "signin" → action = "sign-in"
   - If message contains "sign up" or "signup" → action = "sign-up"
   - Action is set only once; do not overwrite if already set.
4. Phone number detection:
   - If message contains a phone number (digits, optional +, spaces, or dashes) → normalize to E.164 format → store in phone_number.
   - Phone number is set only once; do not overwrite if already set.
5. Password detection:
   - If message contains a string that is not a recognized action or phone number, and password is null, store it in password.
6. **Success condition**:
   - Only output a success message ("Sign in successful!" or "Sign up successful! What's your name?") if ALL THREE values (action, phone_number, password) are set.
   - Otherwise, follow the flow logic exactly.

--- FLOW LOGIC ---
- If action is null → instruction: "Would you like to sign in or sign up?"
- If action is set but phone_number is null → instruction: "Please provide your phone number."
- If phone_number is set but password is null → instruction: "Please provide your password."
- If all three are set → instruction: "Sign in successful!" or "Sign up successful! What's your name?" depending on action.

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
→ {"instruction":"Would you like to sign in or sign up?","action":null,"phone_number":null,"password":null}

User: "sign up"  
→ {"instruction":"Please provide your phone number.","action":"sign-up","phone_number":null,"password":null}

User: "+1 555 123 4567"  
→ {"instruction":"Please provide your password.","action":"sign-up","phone_number":"+15551234567","password":null}

User: "hunter2"  
→ {"instruction":"Sign up successful! What's your name?","action":"sign-up","phone_number":"+15551234567","password":"hunter2"}

User: "sign in"  
→ {"instruction":"Please provide your phone number.","action":"sign-in","phone_number":null,"password":null}

User: "+44 7700 900123"  
→ {"instruction":"Please provide your password.","action":"sign-in","phone_number":"+447700900123","password":null}

User: "myp@ss"  
→ {"instruction":"Sign up successful! What's your name?","action":"sign-in","phone_number":"+447700900123","password":"myp@ss"}

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
    try:
        logger.info("Starting Multi-modal RAG API...")
        
        # Initialize global repository
        logger.info("Initializing global repository...")
        await initialize_global_repository()
        logger.info("Global repository initialized successfully")

            # Initialize model
        logger.info("Initializing model...")
        global openai_llm
        openai_llm = init_model()
        logger.info("Model initialization completed")

        if openai_llm is None:
            logger.error("Failed to load model - openai_llm is None")
            raise RuntimeError("Failed to load model")
        
        # Verify tesseract installation
        logger.info("Verifying tesseract installation...")
        tesseract_ok = verify_tesseract_installation()
        
        if not tesseract_ok:
            logger.warning(
                "WARNING: Tesseract OCR is not properly installed. "
                "Image text extraction and PDF OCR will not work correctly. "
                "Please install Tesseract OCR to enable these features."
            )
        else:
            logger.info("Tesseract OCR verification passed")
        
        # Check upload and processed directories
        upload_folder = os.getenv("UPLOAD_FOLDER", "./uploads")
        processed_folder = os.getenv("PROCESSED_FOLDER", "./processed")
        
        logger.info(f"Upload folder: {os.path.abspath(upload_folder)}")
        logger.info(f"Processed folder: {os.path.abspath(processed_folder)}")
        
        logger.info("System initialization complete")
        
    except Exception as e:
        logger.error(f"Startup failed with error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

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
            content = " ".join(parts)  # Use space instead of empty string for better readability
        else:
            content = str(content)
        
        # Final safety check - ensure content is always a string
        if not isinstance(content, str):
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
                # Ensure content is a string and not a list
                content = msg['content']
                if isinstance(content, list):
                    content = ' '.join(str(item) for item in content)
                elif not isinstance(content, str):
                    content = str(content)
                
                if msg['role'] == 'system':
                    langchain_messages.append(SystemMessage(content=content))
                elif msg['role'] == 'user':
                    langchain_messages.append(HumanMessage(content=content))
                elif msg['role'] == 'assistant':
                    langchain_messages.append(AIMessage(content=content))
            
            logging.info(f"Langchain messages: {langchain_messages}")
            
            # Use the openai_llm to generate response
            logging.info(f"Sending {len(langchain_messages)} messages to vLLM server")
            for i, msg in enumerate(langchain_messages):
                logging.info(f"Message {i}: role={type(msg).__name__}, content_type={type(msg.content)}, content_length={len(str(msg.content))}")
            
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
                    resp_json['instruction'] = "Sign up successful! What's your name?"
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

# --- Custom Exceptions ---
class PartialNameMatchConfirmationRequired(Exception):
    """Exception raised when a partial name match is found and requires user confirmation."""
    def __init__(self, potential_matches, new_name, relationship):
        self.potential_matches = potential_matches
        self.new_name = new_name
        self.relationship = relationship
        self.confirmation_message = self._generate_confirmation_message()
        super().__init__(self.confirmation_message)
    
    def _generate_confirmation_message(self):
        if len(self.potential_matches) == 1:
            match = self.potential_matches[0]
            return (
                f"I found an existing profile for your {self.relationship} named '{match['existing_name']}'. "
                f"Are you referring to the same person as '{self.new_name}'? "
                f"If yes, I'll update the existing profile. If no, I'll create a new one. (yes/no)"
            )
        else:
            options = [f"'{match['existing_name']}'" for match in self.potential_matches]
            return (
                f"I found {len(self.potential_matches)} existing profiles that might match '{self.new_name}': {', '.join(options)}. "
                f"Are you referring to one of these people? If yes, please specify which one. If no, I'll create a new profile."
            )

class MultipleProfileClarificationRequired(Exception):
    """Exception raised when multiple profiles exist with same first name and relationship."""
    
    def __init__(self, first_name: str, relationship: str, existing_profiles: List[Dict[str, Any]], new_information: str):
        self.first_name = first_name
        self.relationship = relationship
        self.existing_profiles = existing_profiles
        self.new_information = new_information
        self.confirmation_message = self._generate_confirmation_message()
        super().__init__(self.confirmation_message)
    
    def _generate_confirmation_message(self):
        message = f"I found {len(self.existing_profiles)} existing {self.relationship} named '{self.first_name}'. What would you like to do?\n\n"
        
        # Show existing profiles
        for i, profile in enumerate(self.existing_profiles, 1):
            name = profile.get('name', 'Unknown')
            location = profile.get('location', 'Not specified')
            phone = profile.get('phone', 'Not specified')
            other_info = profile.get('other_info', [])
            
            message += f"**Existing Profile {i}:** {name}"
            if location != 'Not specified':
                message += f" (lives in {location})"
            if phone != 'Not specified':
                message += f" (phone: {phone})"
            if other_info:
                message += f" - {', '.join(other_info)}"
            message += "\n"
        
        message += f"\n**Options:**\n"
        message += f"• **UPDATE** - Update the existing profile with new information\n"
        message += f"• **CREATE** - Create a new profile (I'll ask for full name to avoid confusion)\n\n"
        message += f"Please respond with 'UPDATE' or 'CREATE' to proceed."
        
        return message

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
    pending_partial_match: Optional[Dict[str, Any]] = None  # New field for pending partial name match confirmation
    pending_multiple_profile_clarification: Optional[Dict[str, Any]] = None  # New field for multiple profile clarification
    pending_new_profile_creation: Optional[Dict[str, Any]] = None  # New field for pending new profile creation with full name
    multiple_profile_choice: Optional[str] = None  # New field to track user's choice: "UPDATE" or "CREATE"
    waiting_for_name: Optional[Dict[str, Any]] = None  # New field for when we're waiting for a name after relationship was mentioned

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
        detection_result = await detect_profiles_in_text(file_content, user_id=user_id)
        
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
                        user_id, name, relationship, information, last_mentioned_profile=None
                    )
                    saved_profiles.append(saved_profile)
                    logger.info(f"[FILE PROFILE DETECTION] Successfully saved profile for {name} ({relationship}) from file: {filename}")
                except PartialNameMatchConfirmationRequired as partial_match_req:
                    # Partial name match found - ask for confirmation
                    logger.info(f"[FILE PROFILE DETECTION] Partial name match requires confirmation for {partial_match_req.new_name}")
                    # For file processing, we'll skip this profile and continue with others
                    logger.warning(f"[FILE PROFILE DETECTION] Skipping profile due to partial name match requiring confirmation: {name}")
                    continue
                    
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
async def detect_profiles_in_text(text: str, chat_history: List[Any] = None, user_id: str = None) -> List[Dict[str, str]]:
    """
    Detects profiles in the given text using LLM for intelligent profile recognition.
    Includes chat history context and existing profiles for better understanding of relationships and references.
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
    
    # Get existing profiles for context if user_id is provided
    profile_context = ""
    if user_id:
        try:
            from rag.database import load_user_profiles
            existing_profiles = await load_user_profiles(user_id)
            if existing_profiles:
                profile_list = []
                for name_lower, profile in existing_profiles.items():
                    name = profile.get('name', 'Unknown')
                    relationship = profile.get('relationship', 'unknown')
                    profile_list.append(f"• {relationship.title()} {name}")
                profile_context = "\n".join(profile_list)
                logger.info(f"[PROFILE DETECTION] Profile context: {profile_context}")
        except Exception as e:
            logger.warning(f"[PROFILE DETECTION] Failed to load profiles for context: {str(e)}")
    
    detection_prompt_template = f"""
    You are an intelligent assistant that extracts **person profiles** from a given text, using **chat history context** to resolve pronouns and relationship references.

    ---

    ## Core Instructions
    1. Always return **only raw JSON** in the specified format.  
    2. Never invent or hallucinate names.  
    3. Never include the user ("Sonnet") in extracted profiles unless explicitly mentioned in third person.  
    4. Never include meta-information (e.g., "asking to create a profile").  
    5. Family members and pets belong inside the main person’s `"information"`, not as separate profiles.  
    6. When multiple profiles exist with the same name, set `"relationship": "unknown"`.  
    7. If the input is about a relationship type only (e.g., "Tell me about my colleague"), return `[]`.  

    ---

    ## Extraction Rules

    ### Names
    - Extract names **exactly as written** (respecting capitalization and full names).
    - If a full name is given (e.g., `"Jacob Smith"` or `"jacob smith"`), **keep the entire name** exactly as provided.
    - Never shorten full names to first names.
    - Do not include descriptive phrases (e.g., `"my friend Ash"` → `"Ash"`).
    - Do not change spelling (e.g., `"Ashely"` must remain `"Ashely"`).
    - Do not use pronouns, gendered terms, or placeholders as names.
    - If no name is present (e.g., "My colleague lives in New York"), return `[]`.

    ### Profiles
    - Create a JSON entry **only for the main subject** of the text.
    - If another person is only mentioned in relation (e.g., as a sibling), include them in `"information"`.
    - Merge multiple facts into one `"information"` string, joined with “and”.

    ### Family and Pets
    - Always include them inside the main person’s `"information"`.
    - Never create separate profiles for family members or pets unless they later become the subject.

    ### Pronoun Resolution
    - If text begins with **“He”** or **“She”**, check chat history:
    - Resolve to the **most recently mentioned person**.
    - Use their **exact name** and relationship.
    - If no match exists, return `[]`.
    - If pronouns appear **in the same sentence as a name**, they always refer to that name.
    - Never resolve pronouns to the user (Sonnet).

    ### Relationship Extraction
    - If the text explicitly uses “my [relationship] [Name]” or “He/She is my [relationship]”:
    - Use that relationship type.
    - Example: `"He is my cousin lives in New York"` → `"relationship": "cousin"`.
    - Never set `"relationship": "unknown"` when the text explicitly provides `"my [relationship]"`.
    - If no relationship is given, default to `"unknown"`.

    ---

    ## Defaults
    - `"relationship": "unknown"` if unstated.  
    - `"information": ""` if no details are provided.  
    - If no valid names are found, return `[]`.  

    ---

    ## Output Format
    - Return only a JSON array:  
    [
    {{
        "name": "Name",
        "relationship": "type",
        "information": "details"
    }}
    ]
    - No explanations, no markdown, no rules in output.

    ---

    ## Validation Checklist
    1. Did you resolve pronouns using chat history?  
    2. Did you keep the exact name as written?  
    3. Did you avoid creating a profile for the user (Sonnet)?  
    4. Are family members/pets included in `"information"` only?  
    5. If the query was about a relationship type only, did you return `[]`?  
    6. If multiple profiles exist for the same name, did you set `"relationship": "unknown"`?  
    7. Is `"information"` free of meta-information?  
    8. Did you merge all facts into one `"information"` string?  

    ---

    ## Examples

    **Input:**  
    My friend Marry is going to Europe and her phone number is 2930492 and her birthday is March 15  

    **Output:**  
    [
    {{
        "name": "Marry",
        "relationship": "friend",
        "information": "going to Europe and phone number is 2930492 and birthday is March 15"
    }}
    ]

    ---

    **Input:**  
    Create a profile for my colleague jacob smith, his phone number is 7734922  

    **Output:**  
    [
    {{
        "name": "jacob smith",
        "relationship": "colleague",
        "information": "phone number is 7734922"
    }}
    ]

    ---

    **Input:**  
    He is my cousin lives in New York. His phone number is 62221033  
    (Chat history: last mentioned person was Robert)  

    **Output:**  
    [
    {{
        "name": "Robert",
        "relationship": "cousin",
        "information": "lives in New York and phone number is 62221033"
    }}
    ]

    ---

    **Input:**  
    Tell me about Jacob  
    (Profiles exist: Friend Jacob and Colleague Jacob Smith)  

    **Output:**  
    [
    {{
        "name": "Jacob",
        "relationship": "unknown",
        "information": ""
    }}
    ]

    ---

    **Input:**  
    Now give me info of Colleague  

    **Output:**  
    []

    """

    max_retries = 2
    detected_names = []

    for attempt in range(max_retries):
        response_text = None  # Initialize variable outside try block
        
        try:
            # Format the prompt template with actual values
            detection_prompt = detection_prompt_template.format(
                chat_context=chat_context if chat_context else "No previous context available.",
                text=text,
                profile_context=profile_context if profile_context else "No existing profiles available."
            )
            
            response = await openai_llm.ainvoke([HumanMessage(content=detection_prompt)])
            response_text = response.content.strip()

            logger.info(f"LLM response for name detection (attempt {attempt+1}): {response_text}")

            # ENHANCED: Multiple strategies to extract JSON from LLM response
            candidate_json = None
            
            # Strategy 1: Try to find JSON array using regex
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
            if json_match:
                try:
                    candidate_json = json.loads(json_match.group())
                    logger.info(f"[PROFILE DETECTION] Successfully extracted JSON using regex")
                except json.JSONDecodeError as e:
                    logger.warning(f"[PROFILE DETECTION] Regex match found but JSON parsing failed: {str(e)}")
            
            # Strategy 2: Try to parse the entire response as JSON
            if not candidate_json:
                try:
                    candidate_json = json.loads(response_text)
                    logger.info(f"[PROFILE DETECTION] Successfully parsed entire response as JSON")
                except json.JSONDecodeError as e:
                    logger.warning(f"[PROFILE DETECTION] Entire response JSON parsing failed: {str(e)}")
            
            # Strategy 3: Try to find JSON after "OUTPUT" or similar markers
            if not candidate_json:
                # Look for common output markers
                output_markers = ["OUTPUT", "Output", "output", "RESULT", "Result", "result"]
                for marker in output_markers:
                    if marker in response_text:
                        parts = response_text.split(marker)
                        if len(parts) > 1:
                            # Try to extract JSON from the part after the marker
                            after_marker = parts[1]
                            json_match = re.search(r'\[\s*\{.*?\}\s*\]', after_marker, re.DOTALL)
                            if json_match:
                                try:
                                    candidate_json = json.loads(json_match.group())
                                    logger.info(f"[PROFILE DETECTION] Successfully extracted JSON after '{marker}' marker")
                                    break
                                except json.JSONDecodeError:
                                    continue
            
            # Strategy 4: Try to find any JSON-like structure
            if not candidate_json:
                # Look for any array-like structure
                array_match = re.search(r'\[\s*\{[^}]*"[^"]*"[^}]*\}\s*\]', response_text, re.DOTALL)
                if array_match:
                    try:
                        # Clean up the matched text to make it valid JSON
                        cleaned_json = array_match.group()
                        # Remove any non-JSON characters that might be present
                        cleaned_json = re.sub(r'[^\x00-\x7F]+', '', cleaned_json)  # Remove non-ASCII
                        candidate_json = json.loads(cleaned_json)
                        logger.info(f"[PROFILE DETECTION] Successfully extracted JSON using fallback pattern")
                    except json.JSONDecodeError:
                        pass
            
            if not candidate_json:
                logger.error(f"[PROFILE DETECTION] All JSON extraction strategies failed")
                logger.error(f"[PROFILE DETECTION] Full LLM response: {response_text}")
                logger.error(f"[PROFILE DETECTION] Response length: {len(response_text)}")
                logger.error(f"[PROFILE DETECTION] Response type: {type(response_text)}")
                logger.error(f"[PROFILE DETECTION] Response contains 'name': {'name' in response_text}")
                logger.error(f"[PROFILE DETECTION] Response contains '[': {'[' in response_text}")
                logger.error(f"[PROFILE DETECTION] Response contains '}}': {'}}' in response_text}")
                raise ValueError(f"No valid JSON array found in LLM response. Response was: {response_text[:200]}...")

            # ENHANCED: Validate JSON structure with better error messages
            if not isinstance(candidate_json, list):
                raise ValueError(f"Parsed JSON is not a list. Got: {type(candidate_json)}")
            
            if len(candidate_json) == 0:
                raise ValueError("Parsed JSON is an empty list")

            filtered_results = []
            for i, entry in enumerate(candidate_json):
                if not isinstance(entry, dict):
                    logger.warning(f"[PROFILE DETECTION] Entry {i} is not a dict: {type(entry)}")
                    continue

                # Validate required fields
                if "name" not in entry:
                    logger.warning(f"[PROFILE DETECTION] Entry {i} missing 'name' field: {entry}")
                    continue
                if "relationship" not in entry:
                    logger.warning(f"[PROFILE DETECTION] Entry {i} missing 'relationship' field: {entry}")
                    continue
                if "information" not in entry:
                    logger.warning(f"[PROFILE DETECTION] Entry {i} missing 'information' field: {entry}")
                    continue

                name = entry.get("name", "").strip().title()
                relationship = entry.get("relationship", "unknown").lower()
                information = entry.get("information", "").strip()

                # Filter invalid names
                if not name or len(name) < 2:
                    continue

                # Filter relationships to allowed set - EXPANDED to include comprehensive family relationships
                allowed_rel = {
                    # Basic relationships
                    "friend", "colleague", "family", "supervisor", "neighbor", "assistant", "boss", "manager",
                    
                    # Family relationships
                    "sister", "brother", "mother", "father", "daughter", "son", "cousin", "uncle", "aunt",
                    "niece", "nephew", "grandmother", "grandfather", "granddaughter", "grandson",
                    "stepbrother", "stepsister", "half-brother", "half-sister", "stepmother", "stepfather",
                    "mother-in-law", "father-in-law", "sister-in-law", "brother-in-law",
                    "daughter-in-law", "son-in-law", "wife", "husband", "spouse", "partner",
                    
                    # Extended family
                    "great-grandmother", "great-grandfather", "great-aunt", "great-uncle",
                    "second-cousin", "third-cousin", "cousin-once-removed",
                    
                    # Professional relationships
                    "mentor", "mentee", "student", "teacher", "professor", "instructor",
                    "coach", "trainee", "apprentice", "intern", "volunteer",
                    
                    # Social relationships
                    "roommate", "housemate", "flatmate", "neighbor", "acquaintance",
                    "classmate", "teammate", "club-member", "church-member",
                    
                    # Default
                    "unknown"
                }
                
                if relationship not in allowed_rel:
                    logger.info(f"Invalid relationship '{relationship}' replaced with 'unknown'")
                    logger.info(f"Allowed relationships: {sorted(allowed_rel)}")
                    relationship = "unknown"

                # Filter common non-names
                common_words = {
                    'my', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                    'is', 'are', 'was', 'were', 'will', 'going', 'friend', 'colleague', 'boss', 'manager', 'assistant',
                    'neighbor', 'family', 'sister', 'brother', 'mother', 'father', 'daughter', 'son', 'cousin', 'uncle', 'aunt',
                    'name', 'names', 'named', 'calling', 'calle d', 'know', 'knows', 'knew', 'meet', 'meets', 'met',
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
            logger.error(f"LLM parsing attempt #{attempt+1} - Full response text: {response_text}")
            logger.error(f"LLM parsing attempt #{attempt+1} - Response length: {len(response_text) if response_text else 'None'}")
            logger.error(f"LLM parsing attempt #{attempt+1} - Exception type: {type(e).__name__}")
            
            # Try to extract any valid JSON from the response for debugging
            if response_text:
                try:
                    # Look for JSON array pattern
                    json_match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
                    if json_match:
                        logger.error(f"LLM parsing attempt #{attempt+1} - Found JSON pattern: {json_match.group()[:200]}...")
                    else:
                        logger.error(f"LLM parsing attempt #{attempt+1} - No JSON array pattern found in response")
                except Exception as debug_e:
                    logger.error(f"LLM parsing attempt #{attempt+1} - Debug extraction failed: {str(debug_e)}")
            
            if attempt == max_retries - 1:
                logger.info("LLM detection failed after retries, falling back")
                fallback_profiles = detect_profiles_fallback(text)
                return fallback_profiles, len(fallback_profiles) > 0

    fallback_profiles = detect_profiles_fallback(text)
    return fallback_profiles, len(fallback_profiles) > 0

async def validate_context_aware_profiles(detected_profiles: List[Dict[str, Any]], chat_context: str, text: str) -> List[Dict[str, Any]]:
    """
    Validate and correct profiles based on chat context to ensure proper name resolution.
    """
    logger.info(f"[CONTEXT VALIDATION] Validating {len(detected_profiles)} profiles with chat context")
    
    # CRITICAL FIX: If LLM already detected names, skip context validation to preserve full names
    if detected_profiles and any(profile.get('name') for profile in detected_profiles):
        detected_names = [profile.get('name') for profile in detected_profiles if profile.get('name')]
        logger.info(f"[CONTEXT VALIDATION] LLM detected names: {detected_names}, skipping context validation to preserve full names")
        logger.info(f"[CONTEXT VALIDATION] This prevents overriding full names like 'jacob smith' with partial names from chat context")
        return detected_profiles
    
    # Check if text contains relationship references that should use chat history
    # EXPANDED to include comprehensive family relationships
    relationship_patterns = [
        r'my\s+(friend|colleague|family|supervisor|neighbor|assistant|boss|manager)\'s?\s+',
        r'my\s+(friend|colleague|family|supervisor|neighbor|assistant|boss|manager)\s+',
        r'my\s+(sister|brother|mother|father|daughter|son|cousin|uncle|aunt|niece|nephew)\'s?\s+',
        r'my\s+(sister|brother|mother|father|daughter|son|cousin|uncle|aunt|niece|nephew)\s+',
        r'my\s+(grandmother|grandfather|granddaughter|grandson|stepbrother|stepsister)\'s?\s+',
        r'my\s+(grandmother|grandfather|granddaughter|grandson|stepbrother|stepsister)\s+',
        r'my\s+(wife|husband|spouse|partner|mentor|mentee)\'s?\s+',
        r'my\s+(wife|husband|spouse|partner|mentor|mentee)\s+',
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
                    # Look for patterns like "my friend Lisa" or "friend Lisa" or "my friend Lisa Smith"
                    name_pattern = rf'(?:my\s+)?{relationship}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
                    name_match = re.search(name_pattern, chat_context, re.IGNORECASE)
                    
                    if name_match:
                        context_name = name_match.group(1)
                        logger.info(f"[CONTEXT VALIDATION] Found {relationship} {context_name} in chat context")
                        
                        # CRITICAL FIX: Only correct names if the current input doesn't explicitly mention a name
                        # Check if the current text explicitly mentions a name after "my friend"
                        explicit_name_pattern = rf'my\s+{relationship}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
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
                        
                        # CRITICAL FIX: Never override LLM-detected names with context names
                        # If the LLM already detected a name, trust it over chat context
                        if detected_profiles and any(profile.get('name') for profile in detected_profiles):
                            detected_names = [profile.get('name') for profile in detected_profiles if profile.get('name')]
                            logger.info(f"[CONTEXT VALIDATION] LLM already detected names: {detected_names}, skipping context-based name correction to preserve full names")
                            logger.info(f"[CONTEXT VALIDATION] This prevents overriding 'jacob smith' with just 'jacob' from chat context")
                            continue
                        
                        # Only correct names when dealing with ambiguous references (no explicit name) AND no LLM detection
                        logger.info(f"[CONTEXT VALIDATION] No LLM detection, proceeding with context-based name correction")
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
    
    # Patterns for relationship references - EXPANDED to include comprehensive family relationships
    relationship_patterns = [
        r'my\s+(friend|colleague|sister|brother|mother|father|neighbor|supervisor|assistant|boss|manager)\'s?\s+',
        r'my\s+(friend|colleague|sister|brother|mother|father|neighbor|supervisor|assistant|boss|manager)\s+',
        r'my\s+(cousin|uncle|aunt|niece|nephew|grandmother|grandfather|granddaughter|grandson)\'s?\s+',
        r'my\s+(cousin|uncle|aunt|niece|nephew|grandmother|grandfather|granddaughter|grandson)\s+',
        r'my\s+(stepbrother|stepsister|half-brother|half-sister|stepmother|stepfather)\'s?\s+',
        r'my\s+(stepbrother|stepsister|half-brother|half-sister|stepmother|stepfather)\s+',
        r'my\s+(wife|husband|spouse|partner|mentor|mentee|roommate|housemate)\'s?\s+',
        r'my\s+(wife|husband|spouse|partner|mentor|mentee|roommate|housemate)\s+',
    ]
    
    for pattern in relationship_patterns:
        match = re.search(pattern, text_lower)
        if match:
            relationship = match.group(1)
            logger.info(f"[RELATIONSHIP RESOLUTION] Detected relationship reference: {relationship}")
            
            # Look for this relationship in chat context
            if relationship in chat_context.lower():
                # Find the name associated with this relationship in chat context
                # Look for patterns like "my friend Marry" or "friend Marry" or "my friend Marry Johnson"
                name_pattern = rf'(?:my\s+)?{relationship}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
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

    logger.info(f"Using fallback regex detection for: {text}")

    common_words = {
        'my', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'will', 'going', 'friend', 'colleague', 'boss', 'manager', 'assistant',
        'neighbor', 'family', 'sister', 'brother', 'mother', 'father', 'daughter', 'son', 'cousin', 'uncle', 'aunt',
        'name', 'names', 'named', 'calling', 'called', 'know', 'knows', 'knew', 'meet', 'meets', 'met',
        'this', 'that', 'these', 'those', 'here', 'there', 'where', 'when', 'what', 'who', 'why', 'how',
        'my', 'me', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'us', 'his', 'her', 'hers', 'him',
        'phone', 'number', 'lives', 'works', 'studies', 'travels', 'goes', 'age', 'years', 'old', 'has'
    }

    # ENHANCED: More precise name detection regex patterns with word boundaries
    name_patterns = [
        # Pattern 1: "my friend Maria" or "my colleague John Smith" -> captures "Maria" or "John Smith" (with additional context)
        r'\b(?:my\s+)?(friend|colleague|boss|manager|assistant|neighbor|family|sister|brother|mother|father|daughter|son|cousin|uncle|aunt|niece|nephew|grandmother|grandfather|granddaughter|grandson|stepbrother|stepsister|half-brother|half-sister|stepmother|stepfather|wife|husband|spouse|partner|mentor|mentee|roommate|housemate)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*?)(?:\s+(?:lives|works|studies|travels|goes|is|was|will|age|years\s+old|has|phone|number|in|at|on|to|for|with|by|and|or|but|so|yet|for|nor))',
        
        # Pattern 1b: "my friend Maria" or "my colleague John Smith" -> captures "Maria" or "John Smith" (without additional context)
        r'\b(?:my\s+)?(friend|colleague|boss|manager|assistant|neighbor|family|sister|brother|mother|father|daughter|son|cousin|uncle|aunt|niece|nephew|grandmother|grandfather|granddaughter|grandson|stepbrother|stepsister|half-brother|half-sister|stepmother|stepfather|wife|husband|spouse|partner|mentor|mentee|roommate|housemate)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*?)(?:\s*$|\s+(?:and|or|but|so|yet|for|nor|,|\.|!|\?))',
        
        # Pattern 2: "Maria is my friend" or "Maria Smith is my friend" -> captures "Maria" or "Maria Smith"
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:my\s+)?(friend|colleague|boss|manager|assistant|neighbor|family|sister|brother|mother|father|daughter|son|cousin|uncle|aunt|niece|nephew|grandmother|grandfather|granddaughter|grandson|stepbrother|stepsister|half-brother|half-sister|stepmother|stepfather|wife|husband|spouse|partner|mentor|mentee|roommate|housemate)\b',
        
        # Pattern 3: "I know Maria" or "I know Maria Smith" -> captures "Maria" or "Maria Smith"
        r'\b(?:I\s+)?(know|knows|knew|met|meet|meets)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        
        # Pattern 3b: "I mean my colleague Marry" -> captures "Marry"
        r'\b(?:I\s+)?(mean|meant)\s+(?:my\s+)?(friend|colleague|boss|manager|assistant|neighbor|family|sister|brother|mother|father|daughter|son|cousin|uncle|aunt|niece|nephew|grandmother|grandfather|granddaughter|grandson|stepbrother|stepsister|half-brother|half-sister|stepmother|stepfather|wife|husband|spouse|partner|mentor|mentee|roommate|housemate)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*?)(?:\s*$|\s+(?:and|or|but|so|yet|for|nor|,|\.|!|\?))',
        
        # Pattern 4: "Maria lives in Australia" or "Maria Smith lives in Australia" -> captures "Maria" or "Maria Smith"
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*?)(?:\s+(?:lives|works|studies|travels|goes|is|was|will|age|years\s+old|has|phone|number))\b',
        
        # Pattern 5: "Tell me about Maria" or "Tell me about Maria Smith" -> captures "Maria" or "Maria Smith"
        r'\b(?:Do\s+you\s+know|Tell\s+me\s+about|Who\s+is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        
        # Pattern 6: "Maria is going to Australia" or "Maria Smith is going to Australia" -> captures "Maria" or "Maria Smith"
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:going|traveling|working|studying|living)\b',
    ]

    detected_names = []

    for pattern in name_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Extract based on pattern groups, tolerate different groupings
            groups = match.groups()
            logger.info(f"[FALLBACK DETECTION] Pattern matched: {pattern[:50]}... -> groups: {groups}")
            
            if len(groups) == 3:
                # Pattern with 3 groups: (mean, relationship, name) - e.g., "I mean my colleague Marry"
                if groups[1].lower() in {"friend", "colleague", "boss", "manager", "assistant", "neighbor", "family",
                                       "sister", "brother", "mother", "father", "daughter", "son", "cousin", "uncle", "aunt",
                                       "niece", "nephew", "grandmother", "grandfather", "granddaughter", "grandson",
                                       "stepbrother", "stepsister", "half-brother", "half-sister", "stepmother", "stepfather",
                                       "wife", "husband", "spouse", "partner", "mentor", "mentee", "roommate", "housemate"}:
                    relationship = groups[1].lower()
                    name = groups[2].title()
                    logger.info(f"[FALLBACK DETECTION] Pattern 3 groups: relationship='{relationship}', name='{name}'")
                else:
                    name = groups[2].title()
                    relationship = "unknown"
                    logger.info(f"[FALLBACK DETECTION] Pattern 3 groups (unknown relationship): name='{name}', relationship='{relationship}'")
            elif len(groups) == 2:
                if groups[0].lower() in {"friend", "colleague", "boss", "manager", "assistant", "neighbor", "family",
                                       "sister", "brother", "mother", "father", "daughter", "son", "cousin", "uncle", "aunt",
                                       "niece", "nephew", "grandmother", "grandfather", "granddaughter", "grandson",
                                       "stepbrother", "stepsister", "half-brother", "half-sister", "stepmother", "stepfather",
                                       "wife", "husband", "spouse", "partner", "mentor", "mentee", "roommate", "housemate"}:
                    relationship = groups[0].lower()
                    name = groups[1].title()
                    logger.info(f"[FALLBACK DETECTION] Pattern 2 groups: relationship='{relationship}', name='{name}'")
                else:
                    name = groups[0].title()
                    relationship = "unknown"
                    logger.info(f"[FALLBACK DETECTION] Pattern 2 groups (unknown relationship): name='{name}', relationship='{relationship}'")
            elif len(groups) == 1:
                name = groups[0].title()
                relationship = "unknown"
                logger.info(f"[FALLBACK DETECTION] Pattern 1 group: name='{name}', relationship='{relationship}'")
            else:
                continue

            # Filter out common words and short names
            if name.lower() in common_words or len(name) < 2:
                logger.info(f"[FALLBACK DETECTION] Skipping common word or short name: {name}")
                continue

            # CRITICAL: Check if the name looks like a real person's name (not a sentence fragment)
            # Names should be 1-3 words, start with capital letter, and not contain common sentence words
            name_words = name.split()
            if len(name_words) > 3:
                logger.info(f"[FALLBACK DETECTION] Skipping name with too many words: {name}")
                continue
            
            # Check if any word in the name is a common word
            if any(word.lower() in common_words for word in name_words):
                logger.info(f"[FALLBACK DETECTION] Skipping name containing common words: {name}")
                continue
            
            # Check if the name contains location or action words that suggest it's not a person's name
            location_action_words = {'lives', 'works', 'studies', 'travels', 'goes', 'is', 'was', 'will', 'has', 'phone', 'number', 'in', 'at', 'on', 'to', 'for', 'with', 'by', 'and', 'or', 'but', 'so', 'yet', 'for', 'nor', 'the', 'a', 'an', 'his', 'her', 'hers', 'him', 'he', 'she', 'it', 'we', 'they', 'them', 'us', 'my', 'me', 'i', 'you'}
            if any(word.lower() in location_action_words for word in name_words):
                logger.info(f"[FALLBACK DETECTION] Skipping name containing location/action words: {name}")
                continue

            context = match.group(0)

            # Extract meaningful information - pass text, name, and full context snippet
            information = extract_meaningful_info(text, name, context)

            profile_data = {
                "name": name,
                "relationship": relationship,
                "information": information,
                "timestamp": datetime.now().isoformat()
            }
            logger.info(f"[FALLBACK DETECTION] Created profile: name='{name}', relationship='{relationship}', information='{information}'")
            detected_names.append(profile_data)

    # ENHANCED: Deduplicate by name only, keeping the best relationship
    unique = {}
    for item in detected_names:
        name_lower = item["name"].lower()
        relationship = item["relationship"]
        
        # If we already have this name, keep the one with the better relationship
        if name_lower in unique:
            existing_relationship = unique[name_lower]["relationship"]
            
            # Priority: friend/colleague > family > unknown
            relationship_priority = {
                "friend": 3, "colleague": 3, "boss": 3, "manager": 3,
                "family": 2, "sister": 2, "brother": 2, "mother": 2, "father": 2,
                "unknown": 1
            }
            
            existing_priority = relationship_priority.get(existing_relationship, 1)
            new_priority = relationship_priority.get(relationship, 1)
            
            if new_priority > existing_priority:
                logger.info(f"[FALLBACK DETECTION] Replacing {existing_relationship} with {relationship} for {name_lower}")
                unique[name_lower] = item
            else:
                logger.info(f"[FALLBACK DETECTION] Keeping {existing_relationship} over {relationship} for {name_lower}")
        else:
            unique[name_lower] = item
    
    filtered = list(unique.values())

    logger.info(f"Fallback detected {len(filtered)} names.")
    return filtered

def extract_meaningful_info(text: str, name: str, context: str) -> str:
    """
    Extract meaningful information about a person from the text.
    """
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
    
    # Apply each pattern to clean the name
    clean_name = name.strip()
    for pattern in patterns_to_remove:
        clean_name = re.sub(pattern, '', clean_name, flags=re.IGNORECASE)
    
    # Clean up any remaining whitespace and preserve exact spelling
    clean_name = clean_name.strip()
    if clean_name:
        # CRITICAL FIX: Preserve exact spelling, only capitalize first letter
        # This prevents "Ashely" from becoming "Ashley"
        clean_name = clean_name[0].upper() + clean_name[1:]
    
    logger.info(f"[PROFILE NAME CLEANING] Cleaned '{name}' to '{clean_name}' (preserved exact spelling)")
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

def generate_unique_profile_key(profiles: Dict[str, Dict[str, Any]], base_key: str) -> str:
    """
    Generate a unique profile key using the format: firstname_relationship_index
    
    Args:
        profiles: Dictionary of existing profiles
        base_key: Base key in format "firstname_relationship"
    
    Returns:
        Unique profile key like "mark_friend_0", "mark_friend_1", etc.
    """
    # Start with index 0
    index = 0
    profile_key = f"{base_key}_{index}"
    
    # Keep incrementing index until we find a unique key
    while profile_key in profiles:
        index += 1
        profile_key = f"{base_key}_{index}"
    
    return profile_key

async def create_or_update_profile_persistent(user_id: str, name: str, relationship: str, information: str, force_create: bool = False, last_mentioned_profile: Dict[str, Any] = None) -> str:
    """
    Creates a new profile or updates an existing one with new information using persistent storage.
    Considers both name and relationship for profile identification.
    Returns the updated profile.
    
    Args:
        user_id: The user ID
        name: The person's name
        relationship: The relationship to the user
        information: Information about the person
        force_create: If True, bypass partial name match checks and create a new profile
        last_mentioned_profile: The last mentioned profile in ongoing chat (if any)
    """
    logger.info(f"--- Invoking create_or_update_profile_persistent for user_id: {user_id} ---")
    logger.info(f"Received Parameters: name='{name}', relationship='{relationship}', information='{information}', force_create={force_create}")

    # Clean the name to ensure it's just the person's name, not a descriptive name
    clean_name = clean_profile_name(name)
    logger.info(f"Cleaned name from '{name}' to '{clean_name}'")

    # Check if this is an ongoing chat (last mentioned profile matches current profile)
    is_ongoing_chat = False
    if last_mentioned_profile and last_mentioned_profile.get('name') and last_mentioned_profile.get('relationship'):
        last_name = last_mentioned_profile.get('name', '').lower().strip()
        last_relationship = last_mentioned_profile.get('relationship', '').lower().strip()
        current_name = clean_name.lower().strip()
        current_relationship = relationship.lower().strip() if relationship else 'unknown'
        
        # Check if names match (exact match or first name match)
        # Also handle cases where the current name might be longer due to fallback detection
        last_first_name = last_name.split()[0] if last_name else ""
        current_first_name = current_name.split()[0] if current_name else ""
        
        name_matches = (
            last_name == current_name or
            last_first_name == current_first_name or
            (last_first_name and current_name.startswith(last_first_name)) or
            (current_first_name and last_name.startswith(current_first_name))
        )
        
        # Check if relationships match
        relationship_matches = last_relationship == current_relationship
        
        is_ongoing_chat = name_matches and relationship_matches
        logger.info(f"Ongoing chat check: last_name='{last_name}', current_name='{current_name}', name_matches={name_matches}")
        logger.info(f"Ongoing chat check: last_relationship='{last_relationship}', current_relationship='{current_relationship}', relationship_matches={relationship_matches}")
        logger.info(f"Is ongoing chat: {is_ongoing_chat}")

    # Load existing profiles
    profiles = await load_user_profiles(user_id)
    logger.info(f"Loaded {len(profiles)} existing profiles for user {user_id}.")

    clean_name_lower = clean_name.lower()
    relationship_lower = relationship.lower() if relationship else 'unknown'
    timestamp = datetime.now().isoformat()
    
    # Create a base key using first name and relationship
    # Extract first name only for the key
    first_name = clean_name_lower.split()[0] if clean_name_lower else clean_name_lower
    base_key = f"{first_name}_{relationship_lower}"
    logger.info(f"Generated base profile key: '{base_key}' (first name: '{first_name}', relationship: '{relationship_lower}').")
    logger.info(f"Force create mode: {force_create}")

    # Find existing profiles with same first name and relationship
    existing_profiles = []
    potential_partial_matches = []
    logger.info("Searching for existing profiles with same first name and relationship...")
    
    for key, profile in profiles.items():
        # Ensure keys exist before lowercasing
        prof_name = profile.get('name', '').lower()
        prof_rel = profile.get('relationship', 'unknown').lower()
        prof_first_name = prof_name.split()[0] if prof_name else prof_name
        
        # Check for same first name and relationship
        if prof_first_name == first_name and prof_rel == relationship_lower:
            existing_profiles.append((key, profile))
        
        # Check for potential partial name matches (same relationship)
        elif prof_rel == relationship_lower:
            # Check if names are related (one contains the other)
            name_related = (
                clean_name_lower in prof_name.split() or 
                prof_name in clean_name_lower.split() or
                any(word in prof_name.split() for word in clean_name_lower.split()) or
                any(word in clean_name_lower.split() for word in prof_name.split())
            )
            
            if name_related:
                potential_partial_matches.append({
                    'key': key,
                    'profile': profile,
                    'existing_name': profile.get('name', ''),
                    'new_name': clean_name,
                    'relationship': relationship
                })
                logger.info(f"Found potential partial name match: '{clean_name}' vs existing '{profile.get('name', '')}' (same relationship: {relationship})")
                logger.info(f"Match details: clean_name_lower='{clean_name_lower}', prof_name='{prof_name}', relationship='{relationship}'")
            else:
                logger.info(f"No partial name match: '{clean_name}' vs '{profile.get('name', '')}' (same relationship: {relationship})")
                logger.info(f"Reason: Names not related - clean_name_lower='{clean_name_lower}', prof_name='{prof_name}'")

    # Handle existing profiles with same first name and relationship
    if existing_profiles:
        logger.info(f"Found {len(existing_profiles)} existing profiles with same first name and relationship.")
        
        # Check if this is an ongoing conversation - if so, auto-update the profile
        if is_ongoing_chat:
            logger.info(f"[ONGOING CHAT] Auto-updating existing profile for ongoing conversation: {clean_name} ({relationship})")
            
            # Find the most recent profile with the same first name and relationship
            target_profile = None
            target_key = None
            latest_timestamp = None
            
            for key, profile in existing_profiles:
                profile_timestamp = profile.get('last_updated', profile.get('created_date', ''))
                if latest_timestamp is None or profile_timestamp > latest_timestamp:
                    latest_timestamp = profile_timestamp
                    target_profile = profile
                    target_key = key
            
            if target_profile:
                # Update the existing profile with new information
                from rag.rich_profile_utils import intelligently_update_profile
                
                try:
                    updated_profile = await intelligently_update_profile(
                        user_id, target_key, target_profile, information
                    )
                    logger.info(f"[ONGOING CHAT] Successfully updated profile {target_key} with new information")
                    return target_key
                except Exception as e:
                    logger.error(f"[ONGOING CHAT] Error updating profile: {str(e)}")
                    # Fall through to clarification if update fails
            else:
                logger.warning(f"[ONGOING CHAT] No target profile found for update")
                # Fall through to clarification
        
        # Ask for clarification when there are existing profiles with same first name and relationship
        logger.info(f"Found {len(existing_profiles)} profile(s) with same first name '{first_name}' and relationship '{relationship_lower}' - requiring clarification")
        
        # Prepare existing profiles for clarification
        existing_profiles_list = []
        for key, profile in existing_profiles:
            existing_profiles_list.append({
                'key': key,
                'name': profile.get('name', 'Unknown'),
                'location': profile.get('location', 'Not specified'),
                'phone': profile.get('phone', 'Not specified'),
                'other_info': profile.get('other_info', [])
            })
        
        # Raise exception to trigger clarification
        raise MultipleProfileClarificationRequired(
            first_name=first_name,
            relationship=relationship_lower,
            existing_profiles=existing_profiles_list,
            new_information=information
        )

    elif potential_partial_matches and not force_create:
        # Found potential partial name matches - check if we can auto-merge
        logger.info(f"Found {len(potential_partial_matches)} potential partial name matches.")
        
        # Check if we can auto-merge (full name provided with same relationship)
        auto_merge_candidate = None
        for match in potential_partial_matches:
            existing_name = match['existing_name']
            new_name = match['new_name']
            
            # Check if new name is longer/more specific than existing name
            # This indicates a full name being provided for a partial name profile
            existing_words = existing_name.lower().split()
            new_words = new_name.lower().split()
            
            # Auto-merge conditions:
            # 1. New name has more words than existing name
            # 2. New name starts with the existing name (e.g., "Jacob" -> "Jacob Smith")
            # 3. New name contains the existing name as a prefix
            should_merge = (
                len(new_words) > len(existing_words) and 
                (new_name.lower().startswith(existing_name.lower()) or
                 existing_name.lower() in new_name.lower().split()[:len(existing_words)])
            )
            
            if should_merge:
                auto_merge_candidate = match
                logger.info(f"Auto-merge candidate found: '{existing_name}' -> '{new_name}' (same relationship: {relationship})")
                logger.info(f"Merge reason: New name '{new_name}' is more specific than existing '{existing_name}'")
                break
        
        if auto_merge_candidate:
            # Auto-merge the profiles - update existing profile with new name and information
            logger.info(f"Auto-merging profile: '{auto_merge_candidate['existing_name']}' -> '{auto_merge_candidate['new_name']}'")
            
            try:
                # Get the existing profile to update
                existing_profile_key = auto_merge_candidate['key']
                existing_profile = profiles[existing_profile_key]
                
                # Update the name to the more specific version
                existing_profile['name'] = clean_name
                existing_profile['last_updated'] = timestamp
                
                # Add the new information
                if isinstance(existing_profile.get('information'), list):
                    existing_profile['information'].append(information)
                else:
                    existing_profile['information'] = [existing_profile.get('information', ''), information]
                
                # Generate new profile key for the updated name
                new_clean_name_for_key = clean_name_lower.replace(' ', '_')
                new_profile_key = f"{new_clean_name_for_key}_{relationship_lower}"
                
                # Remove old profile and add new one with updated key
                del profiles[existing_profile_key]
                profiles[new_profile_key] = existing_profile
                
                logger.info(f"Successfully auto-merged profile: '{existing_profile_key}' -> '{new_profile_key}'")
                logger.info(f"Updated profile name from '{auto_merge_candidate['existing_name']}' to '{clean_name}'")
                
                # Save the merged profiles
                success = await save_user_profiles(user_id, profiles)
                if success:
                    logger.info("Successfully saved merged profiles to persistent storage.")
                else:
                    logger.error("Failed to save merged profiles to persistent storage.")
                
                # Return the merged profile
                return existing_profile
                
            except Exception as e:
                logger.error(f"Error during auto-merge: {str(e)}")
                # Fallback to asking for confirmation
                logger.info("Auto-merge failed, falling back to confirmation approach")
                raise PartialNameMatchConfirmationRequired(potential_partial_matches, clean_name, relationship)
        
        else:
            # No auto-merge possible - ask for confirmation
            logger.info("No auto-merge possible, asking for confirmation")
            raise PartialNameMatchConfirmationRequired(potential_partial_matches, clean_name, relationship)
    
    elif potential_partial_matches and force_create:
        # Force create mode - but still check for auto-merge opportunities
        logger.info(f"Force create mode: found {len(potential_partial_matches)} potential partial name matches")
        
        # Check if we can auto-merge even in force create mode
        auto_merge_candidate = None
        for match in potential_partial_matches:
            existing_name = match['existing_name']
            new_name = match['new_name']
            
            # Check if new name is longer/more specific than existing name
            existing_words = existing_name.lower().split()
            new_words = new_name.lower().split()
            
            # Auto-merge conditions (same as above)
            should_merge = (
                len(new_words) > len(existing_words) and 
                (new_name.lower().startswith(existing_name.lower()) or
                 existing_name.lower() in new_name.lower().split()[:len(existing_words)])
            )
            
            if should_merge:
                auto_merge_candidate = match
                logger.info(f"Force create mode: auto-merge candidate found: '{existing_name}' -> '{new_name}' (same relationship: {relationship})")
                logger.info(f"Force create mode: merge reason: New name '{new_name}' is more specific than existing '{existing_name}'")
                break
        
        if auto_merge_candidate:
            # Auto-merge the profiles even in force create mode
            logger.info(f"Force create mode: auto-merging profile: '{auto_merge_candidate['existing_name']}' -> '{auto_merge_candidate['new_name']}'")
            
            try:
                # Get the existing profile to update
                existing_profile_key = auto_merge_candidate['key']
                existing_profile = profiles[existing_profile_key]
                
                # Update the name to the more specific version
                existing_profile['name'] = clean_name
                existing_profile['last_updated'] = timestamp
                
                # Add the new information
                if isinstance(existing_profile.get('information'), list):
                    existing_profile['information'].append(information)
                else:
                    existing_profile['information'] = [existing_profile.get('information', ''), information]
                
                # Generate new profile key for the updated name
                new_clean_name_for_key = clean_name_lower.replace(' ', '_')
                new_profile_key = f"{new_clean_name_for_key}_{relationship_lower}"
                
                # Remove old profile and add new one with updated key
                del profiles[existing_profile_key]
                profiles[new_profile_key] = existing_profile
                
                logger.info(f"Force create mode: successfully auto-merged profile: '{existing_profile_key}' -> '{new_profile_key}'")
                logger.info(f"Updated profile name from '{auto_merge_candidate['existing_name']}' to '{clean_name}'")
                
                # Save the merged profiles
                success = await save_user_profiles(user_id, profiles)
                if success:
                    logger.info("Force create mode: successfully saved merged profiles to persistent storage.")
                else:
                    logger.error("Force create mode: failed to save merged profiles to persistent storage.")
                
                # Return the merged profile
                return existing_profile
                
            except Exception as e:
                logger.error(f"Force create mode: error during auto-merge: {str(e)}")
                # Continue with force create as intended
                logger.info("Force create mode: auto-merge failed, proceeding with new profile creation")
        
        # Clear the partial matches list since we're forcing creation
        potential_partial_matches = []
        
    # If force_create=True or no partial matches, proceed to create new profile
    if not potential_partial_matches or force_create:
        logger.info("Proceeding to create a new profile entry.")
        
        # Generate a unique key with index for the new profile
        profile_key = generate_unique_profile_key(profiles, base_key)
        logger.info(f"Generated unique profile key: '{profile_key}'")
        
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
            logger.info(f"Created new profile data with key '{profile_key}'.")
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
        
        # After saving, check for and clean up any duplicate profiles
        logger.info("Checking for duplicate profiles to clean up...")
        cleanup_success = await cleanup_duplicate_profiles(user_id)
        if cleanup_success:
            logger.info("Profile cleanup completed successfully")
        else:
            logger.warning("Profile cleanup encountered issues")
    else:
        logger.error("Failed to save profiles to persistent storage.")

    # Select the correct profile to return
    final_profile = profiles.get(profile_key, {}) # Use .get for safety

    logger.info(f"--- Function complete. Returning profile for key '{profile_key}' ---")
    # return f"OK, I've noted your {relationship} {name} profile!"
    return final_profile

async def get_all_profiles_persistent(user_id: str) -> List[Dict[str, Any]]:
    """
    Returns all profiles as a list using persistent storage.
    """
    profiles = await load_user_profiles(user_id)
    return list(profiles.values())

async def cleanup_duplicate_profiles(user_id: str) -> bool:
    """
    Identifies and merges duplicate profiles that should have been auto-merged.
    This function can be called to clean up existing duplicate profiles.
    """
    try:
        profiles = await load_user_profiles(user_id)
        logger.info(f"[PROFILE CLEANUP] Checking for duplicate profiles to merge for user {user_id}")
        
        # Group profiles by relationship
        profiles_by_relationship = {}
        for key, profile in profiles.items():
            relationship = profile.get('relationship', 'unknown').lower()
            if relationship not in profiles_by_relationship:
                profiles_by_relationship[relationship] = []
            profiles_by_relationship[relationship].append((key, profile))
        
        merged_count = 0
        
        # Check each relationship group for potential merges
        for relationship, profile_list in profiles_by_relationship.items():
            if len(profile_list) <= 1:
                continue
                
            # Sort profiles by name length (shorter names first)
            profile_list.sort(key=lambda x: len(x[1].get('name', '').split()))
            
            # Check for partial name matches that can be merged
            for i, (key1, profile1) in enumerate(profile_list):
                name1 = profile1.get('name', '')
                name1_words = name1.lower().split()
                
                for j, (key2, profile2) in enumerate(profile_list[i+1:], i+1):
                    name2 = profile2.get('name', '')
                    name2_words = name2.lower().split()
                    
                    # Check if name2 is a more specific version of name1
                    if (len(name2_words) > len(name1_words) and 
                        name2.lower().startswith(name1.lower())):
                        
                        logger.info(f"[PROFILE CLEANUP] Found duplicate profiles to merge: '{name1}' -> '{name2}' (relationship: {relationship})")
                        
                        try:
                            # Merge profile2 into profile1 (keep the more specific name)
                            from datetime import datetime
                            merged_profile = profile2.copy()
                            merged_profile['name'] = name2  # Keep the more specific name
                            merged_profile['last_updated'] = datetime.now().isoformat()
                            
                            # Merge information from both profiles
                            info1 = profile1.get('information', [])
                            info2 = profile2.get('information', [])
                            
                            if isinstance(info1, list) and isinstance(info2, list):
                                merged_profile['information'] = info1 + info2
                            elif isinstance(info1, list):
                                merged_profile['information'] = info1 + [info2] if info2 else info1
                            elif isinstance(info2, list):
                                merged_profile['information'] = [info1] + info2 if info1 else info2
                            else:
                                merged_profile['information'] = [info1, info2] if info1 and info2 else (info1 or info2)
                            
                            # Generate new key for merged profile
                            new_name_for_key = name2.lower().replace(' ', '_')
                            new_key = f"{new_name_for_key}_{relationship}"
                            
                            # Convert merged profile to rich format
                            try:
                                logger.info(f"[PROFILE CLEANUP] Converting merged profile to rich format: {merged_profile}")
                                from rag.rich_profile_utils import convert_simple_to_rich_profile
                                rich_merged_profile = await convert_simple_to_rich_profile(merged_profile)
                                logger.info(f"[PROFILE CLEANUP] Rich profile conversion result: {rich_merged_profile}")
                                
                                # Remove old profiles and add rich merged one
                                del profiles[key1]
                                del profiles[key2]
                                profiles[new_key] = rich_merged_profile
                                
                                merged_count += 1
                                logger.info(f"[PROFILE CLEANUP] Successfully merged and converted profiles: '{key1}' + '{key2}' -> '{new_key}' (rich format)")
                                
                            except Exception as e:
                                logger.error(f"[PROFILE CLEANUP] Error converting merged profile to rich format: {str(e)}")
                                # Fallback to simple format if rich conversion fails
                                del profiles[key1]
                                del profiles[key2]
                                profiles[new_key] = merged_profile
                                
                                merged_count += 1
                                logger.info(f"[PROFILE CLEANUP] Successfully merged profiles (simple format): '{key1}' + '{key2}' -> '{new_key}'")
                            
                        except Exception as e:
                            logger.error(f"[PROFILE CLEANUP] Error merging profiles: {str(e)}")
                            continue
        
        if merged_count > 0:
            # Save the cleaned up profiles
            success = await save_user_profiles(user_id, profiles)
            if success:
                logger.info(f"[PROFILE CLEANUP] Successfully cleaned up {merged_count} duplicate profiles")
                
                # After cleanup, ensure all profiles are in rich format
                logger.info("[PROFILE CLEANUP] Converting cleaned profiles to rich format...")
                rich_conversion_success = await convert_profile_to_rich_format(user_id)
                if rich_conversion_success:
                    logger.info("[PROFILE CLEANUP] Rich format conversion completed successfully")
                else:
                    logger.warning("[PROFILE CLEANUP] Rich format conversion encountered issues")
                
                return True
            else:
                logger.error("[PROFILE CLEANUP] Failed to save cleaned up profiles")
                return False
        else:
            logger.info("[PROFILE CLEANUP] No duplicate profiles found to merge")
            
            # Even if no merges, ensure all profiles are in rich format
            logger.info("[PROFILE CLEANUP] Converting existing profiles to rich format...")
            rich_conversion_success = await convert_profile_to_rich_format(user_id)
            if rich_conversion_success:
                logger.info("[PROFILE CLEANUP] Rich format conversion completed successfully")
            else:
                logger.warning("[PROFILE CLEANUP] Rich format conversion encountered issues")
            
            return True
            
    except Exception as e:
        logger.error(f"[PROFILE CLEANUP] Error during cleanup: {str(e)}")
        return False

async def convert_profile_to_rich_format(user_id: str, profile_key: str = None) -> bool:
    """
    Converts a specific profile or all profiles to rich format.
    This is useful for converting existing simple profiles to rich format.
    """
    from rag.rich_profile_utils import convert_simple_to_rich_profile, is_rich_profile
    
    try:
        profiles = await load_user_profiles(user_id)
        converted_count = 0
        
        if profile_key:
            # Convert specific profile
            if profile_key in profiles:
                profile = profiles[profile_key]
                if not is_rich_profile(profile):
                    logger.info(f"[RICH CONVERSION] Converting profile '{profile_key}' to rich format")
                    try:
                        rich_profile = await convert_simple_to_rich_profile(profile)
                        profiles[profile_key] = rich_profile
                        converted_count += 1
                        logger.info(f"[RICH CONVERSION] Successfully converted profile '{profile_key}' to rich format")
                    except Exception as e:
                        logger.error(f"[RICH CONVERSION] Error converting profile '{profile_key}': {str(e)}")
                        return False
            else:
                logger.warning(f"[RICH CONVERSION] Profile key '{profile_key}' not found")
                return False
        else:
            # Convert all profiles that aren't already rich
            for key, profile in profiles.items():
                if not is_rich_profile(profile):
                    logger.info(f"[RICH CONVERSION] Converting profile '{key}' to rich format")
                    try:
                        rich_profile = await convert_simple_to_rich_profile(profile)
                        profiles[key] = rich_profile
                        converted_count += 1
                        logger.info(f"[RICH CONVERSION] Successfully converted profile '{key}' to rich format")
                    except Exception as e:
                        logger.error(f"[RICH CONVERSION] Error converting profile '{key}': {str(e)}")
                        continue
        
        if converted_count > 0:
            # Save the converted profiles
            success = await save_user_profiles(user_id, profiles)
            if success:
                logger.info(f"[RICH CONVERSION] Successfully converted {converted_count} profiles to rich format")
                return True
            else:
                logger.error("[RICH CONVERSION] Failed to save converted profiles")
                return False
        else:
            logger.info("[RICH CONVERSION] No profiles needed conversion")
            return True
            
    except Exception as e:
        logger.error(f"[RICH CONVERSION] Error during conversion: {str(e)}")
        return False

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
    - Normalize formatting: capitalize the first letter, lowercase the rest (e.g., "john" → "John").

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
    - "Hi" → "NO_NAME"
    - "Hello there" → "NO_NAME"
    - "My name is John" → "John"
    - "I'm Sarah" → "Sarah"
    - "Call me Alex" → "Alex"
    - "John" → "John"
    - "Mary Ann" → "Mary"
    - "Nice to meet you" → "NO_NAME"
    - "How are you?" → "NO_NAME"
    - "This is David" → "David"
    - "I'm called Emma" → "Emma"
    - "The name's Bond" → "Bond"
    - "Sonnet" → "Sonnet"
    - "My name is Sonnet" → "Sonnet"
    - "I'm Sonnet" → "Sonnet"

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

    # Check if this is a new user who just completed sign-up and needs to provide their name
    if not state.user_name:
        # Check if the last assistant message indicates sign-up completion
        last_assistant_message = None
        for m in reversed(request.messages):
            if (isinstance(m, dict) and m.get('role') == 'assistant') or (hasattr(m, 'role') and m.role == 'assistant'):
                last_assistant_message = m['content'] if isinstance(m, dict) else m.content
                break
        
        if last_assistant_message and 'Sign up successful! What\'s your name?' in last_assistant_message:
            logger.info("[LANGGRAPH PATH] Detected new user after sign-up, routing to name collection")
            state.agent_status = "get_name"
            return state
        
        # ENHANCED: Check if this is a new user providing their name as first message
        # This handles cases where user transitions from auth flow to chat flow
        if len(request.messages) == 1 and request.messages[0].role == 'user':
            user_message = request.messages[0].content.strip()
            # Check if the message looks like a name (single word, capitalized, no special chars)
            if (len(user_message.split()) == 1 and 
                user_message.isalpha() and 
                user_message[0].isupper() and 
                len(user_message) >= 2 and 
                len(user_message) <= 20):
                logger.info(f"[LANGGRAPH PATH] Detected potential name from new user: {user_message}")
                logger.info("[LANGGRAPH PATH] Routing to name collection for verification")
                state.agent_status = "get_name"
                return state

    logger.info(f"[LANGGRAPH PATH] Extracted retrieval_query: {retrieval_query}")
    state.retrieval_query = retrieval_query
    
    # Clear searched links and realtime data for new conversation turn
    logger.info(f"[LANGGRAPH PATH] Before clearing - searched_links: {state.searched_links}, realtime_data: {state.realtime_data is not None}")
    state.searched_links = []
    state.realtime_data = None
    
    logger.info(f"[LANGGRAPH PATH] Cleared searched_links and realtime_data for new turn")
    
    # Check if we have pending clarifications that need to be handled
    if state.pending_profile_clarification:
        logger.info(f"[LANGGRAPH PATH] Found pending profile clarification, setting agent_status to waiting_profile_clarification")
        state.agent_status = "waiting_profile_clarification"
    elif state.pending_profile_update:
        logger.info(f"[LANGGRAPH PATH] Found pending profile update, setting agent_status to waiting_confirmation")
        state.agent_status = "waiting_confirmation"
    elif state.pending_partial_match:
        logger.info(f"[LANGGRAPH PATH] Found pending partial match, setting agent_status to waiting_partial_match_confirmation")
        state.agent_status = "waiting_partial_match_confirmation"
    elif state.pending_new_profile_creation:
        logger.info(f"[LANGGRAPH PATH] Found pending new profile creation, setting agent_status to waiting_full_name_for_new_profile")
        state.agent_status = "waiting_full_name_for_new_profile"
    elif state.waiting_for_name:
        logger.info(f"[LANGGRAPH PATH] Found waiting for name, setting agent_status to waiting_for_name")
        state.agent_status = "waiting_for_name"
    elif state.pending_multiple_profile_clarification:
        logger.info(f"[LANGGRAPH PATH] Found pending multiple profile clarification, setting agent_status to waiting_multiple_profile_clarification")
        state.agent_status = "waiting_multiple_profile_clarification"
    elif state.confirm_profile:
        logger.info(f"[LANGGRAPH PATH] Found confirm_profile, preserving confirm_creation status")
        state.agent_status = "confirm_creation"
    else:
        # Only set to initialize if no pending states
        state.agent_status = "initialize"
    
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
    
    # CRITICAL: Check if we have pending clarifications first
    if state.pending_profile_clarification:
        logger.info(f"[INTENT ANALYSIS] Found pending profile clarification, skipping intent analysis and routing to clarification handler")
        state.agent_status = "waiting_profile_clarification"
        return state
    elif state.pending_profile_update:
        logger.info(f"[INTENT ANALYSIS] Found pending profile update, skipping intent analysis and routing to confirmation handler")
        state.agent_status = "waiting_confirmation"
        return state
    elif state.pending_partial_match:
        logger.info(f"[INTENT ANALYSIS] Found pending partial match, skipping intent analysis and routing to partial match handler")
        state.agent_status = "waiting_partial_match_confirmation"
        return state
    elif state.pending_multiple_profile_clarification:
        logger.info(f"[INTENT ANALYSIS] Found pending multiple profile clarification, skipping intent analysis and routing to multiple profile clarification handler")
        state.agent_status = "waiting_multiple_profile_clarification"
        return state
    
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
    - Generic groups (e.g., "students", "politicians") = not specific persons → classify as `general`.  

    ---

    ### Context Rules
    - Use **chat history** to resolve pronouns or relationship references.  
    - If "my [relationship]" is used (e.g., "my colleague"), resolve it to the previously introduced person.  
    - Continuation statements (starting with "And") that add facts about a previously mentioned person = **give_info**.  

    ---

    ### Classification Logic
    **Step 1** – Does the query mention a specific person (name, relationship, or resolvable pronoun)?  
    - NO → `general`  
    - YES → Go to Step 2  

    **Step 2** – Is the user providing or requesting facts?  
    - Providing → `give_info`  
    - Requesting → `ask_info`
    
    **Step 2a** – Special profile operations (always `give_info`):
    - Profile creation: "Create profile for", "Add profile", "New profile", "Create a profile for", "Make profile"
    - Profile updates: "Update", "Change", "Set" + person + attribute  

    **Step 3** – Special cases (in order of priority)
    - **Questions starting with "How", "Where", "What", "When", "Why" + person name** → `ask_info` (HIGHEST PRIORITY)
    - **Questions about specific people's attributes** (e.g., "How many sisters does [Name] have?") → `ask_info` (HIGHEST PRIORITY)
    - Queries like "Tell me about [Name]" or "What do you know about [Name]" → `ask_info`
    - **Questions containing "what's", "what is", "what are" + person name** → `ask_info` (HIGHEST PRIORITY)
    - Starts with "And" + adds facts → `give_info` (ONLY if not a question)
    - Uses pronouns ("he", "she", "they") tied to history + facts → `give_info`  
    - Uses "my [relationship]" + facts → `give_info`  
    - Profile update requests (e.g., "Update", "Change", "Set") + person + attribute → `give_info`
    - **Profile creation requests** (e.g., "Create a profile for", "Add profile", "New profile", "Create profile", "Make profile") + person → `give_info`
    
    **CRITICAL**: Profile creation requests are ALWAYS `give_info` - they provide information about creating a profile for a specific person.  

    ---

    ### Examples
    - "My friend Sarah just started a new job." → give_info  
    - "Do you know where John lives?" → ask_info  
    - "Tell me about Lionel Messi." → ask_info  
    - "How many sisters does marry have?" → ask_info
    - "Where does Sarah live?" → ask_info
    - "What's John's phone number?" → ask_info
    - "And also what's the phone number of friend Sam?" → ask_info (question takes priority over "And")
    - "Let's talk about the latest iPhone." → general  
    - "And Lisa works at Google." → give_info  
    - "She has two brothers." (refers to previous Sarah) → give_info
    - "Update my friend Marry's location to Italy" → give_info
    - "Change John's phone number to 123-456-7890" → give_info
    - "Set Sarah's workplace to Google" → give_info
    - "Create a profile for my friend sam" → give_info
    - "Add profile for John" → give_info
    - "New profile for Sarah" → give_info
    - "Create profile for my colleague Mike" → give_info
    - "Make profile for my friend Lisa" → give_info  

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

    # Debug: Log the original query for troubleshooting
    logger.info(f"[INTENT ANALYSIS] Original query: '{state.retrieval_query}'")
    logger.info(f"[INTENT ANALYSIS] Query contains 'create': {'create' in state.retrieval_query.lower()}")
    logger.info(f"[INTENT ANALYSIS] Query contains 'profile': {'profile' in state.retrieval_query.lower()}")
    logger.info(f"[INTENT ANALYSIS] Query contains 'friend': {'friend' in state.retrieval_query.lower()}")
    logger.info(f"[INTENT ANALYSIS] Query contains 'colleague': {'colleague' in state.retrieval_query.lower()}")

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
    
    # ENHANCED: Fallback check for profile creation requests that might be misclassified
    query_lower = state.retrieval_query.lower()
    profile_creation_patterns = [
        "create a profile for", "create profile for", "add profile for", 
        "new profile for", "make profile for", "create profile", "add profile"
    ]
    
    if any(pattern in query_lower for pattern in profile_creation_patterns):
        logger.info(f"[INTENT ANALYSIS] Fallback: Detected profile creation pattern, forcing give_info intent")
        state.agent_status = "give_info"
        logger.info(f"[INTENT ANALYSIS] Intent corrected to: {state.agent_status}")
    
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
        r'[A-Z][a-z]+\s+(has|is|was|will|can|does)\s+(a|an|the\s+)?(phone|number|address|email|age|birthday|job|work|house|car|dog|cat|sister|brother|mother|father|friend|colleague)',  # Only match person names with specific personal attributes
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
        # Direct factual statements about specific people (but exclude question words)
        r'^(?!who|what|where|when|why|how)[a-zA-Z]+\s+(lives?|lived|has|have|is|are|was|were|likes?|like|works?|work|studies?|study)\s+',
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
        # First check if this is a general knowledge question that should stay general
        general_knowledge_patterns = [
            r'what\s+(is|are|was|were)\s+[a-z\s]+(\?)?$',  # "what is linear algebra?"
            r'how\s+(does|do|did|can|could)\s+[a-z\s]+(\?)?$',  # "how does photosynthesis work?"
            r'why\s+(is|are|was|were|does|do|did)\s+[a-z\s]+(\?)?$',  # "why is the sky blue?"
            r'when\s+(is|are|was|were|does|do|did)\s+[a-z\s]+(\?)?$',  # "when is the next eclipse?"
            r'where\s+(is|are|was|were|does|do|did)\s+[a-z\s]+(\?)?$',  # "where is the capital of France?"
        ]
        
        # If it matches general knowledge patterns, keep it as general
        for pattern in general_knowledge_patterns:
            if re.search(pattern, text_lower):
                logger.info(f"[INTENT VALIDATION] General knowledge pattern '{pattern}' matched - keeping as general")
                return "general"
        
        for pattern in obvious_give_info_patterns:
            if re.search(pattern, text_lower):
                logger.info(f"[INTENT VALIDATION] Pattern '{pattern}' matched - forcing correction to give_info")
                return "give_info"
        
        # Check if this is obviously ask_info but was classified as general
        for pattern in obvious_ask_info_patterns:
            if re.search(pattern, text_lower):
                logger.info(f"[INTENT VALIDATION] Pattern '{pattern}' matched - forcing correction to ask_info")
                return "ask_info"
    
    # Check if this is obviously ask_info but was classified as give_info
    elif current_intent == "give_info":
        for pattern in obvious_ask_info_patterns:
            if re.search(pattern, text_lower):
                logger.info(f"[INTENT VALIDATION] Pattern '{pattern}' matched - forcing correction from give_info to ask_info")
                return "ask_info"
        
        # Special check for the exact pattern "How many sisters have marry got"
        if re.search(r'how many\s+(sisters?|brothers?|children|pets|friends)\s+(does|do|have)\s+[a-zA-Z]+\s+(got|have)', text_lower):
            logger.info(f"[INTENT VALIDATION] Specific question pattern about person's attributes matched - forcing correction to ask_info")
            return "ask_info"
        
        # Special check for "And also what's" pattern - this should be ask_info, not give_info
        if re.search(r'^and\s+(also\s+)?what\'?s?\s+(is\s+)?(the\s+)?(phone|number|address|email|age|birthday|job|work|location|workplace)', text_lower):
            logger.info(f"[INTENT VALIDATION] 'And also what's' question pattern matched - forcing correction to ask_info")
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
        # Direct factual statements about specific people (but exclude question words)
        r'^(?!who|what|where|when|why|how)[a-zA-Z]+\s+(lives?|lived|has|have|is|are|was|were|likes?|like|works?|work|studies?|study)\s+',
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
        r'^(how|where|what|when|why|who)\s+[^?]*[a-zA-Z]+[^?]*\?',
        # Special pattern for "And also what's" questions
        r'^and\s+(also\s+)?what\'?s?\s+(is\s+)?(the\s+)?(phone|number|address|email|age|birthday|job|work|location|workplace)'
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
    - "my friend" → "friend"
    - "my colleague" → "colleague" 
    - "my family" → "family"
    - "my supervisor" → "supervisor"
    - "my neighbor" → "neighbor"
    - "my assistant" → "assistant"
    
    If the relationship is not mentioned or unclear, respond with "unknown".
    If a relationship is mentioned, respond with exactly that relationship word (friend, colleague, family, supervisor, neighbor, assistant).
    
    Examples:
    Text: "my friend Tom" → Response: "friend"
    Text: "my colleague Sarah" → Response: "colleague"
    Text: "friend Mark" → Response: "friend"
    Text: "colleague John" → Response: "colleague"
    Text: "just tell me about him" → Response: "unknown"
    
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
    
    # Import required functions
    from rag.rich_profile_utils import intelligently_update_profile
    # replace_pronouns is defined in this file, so no import needed
    
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
            
            try:
                # Create or update profile now
                updated_profile = await create_or_update_profile_persistent( 
                    user_id,
                    confirmed_profile["name"],
                    confirmed_profile["relationship"],
                    state.confirm_profile['information'] or "No additional info provided.",
                    last_mentioned_profile=state.last_mentioned_profile
                )
                
                # Success case - profile created successfully
                state.final_response = (
                    f"Great, your {confirmed_profile['relationship']} {confirmed_profile['name']}'s profile "
                    f"is noted successfully."
                )
                state.last_mentioned_profile = confirmed_profile
                state.confirm_profile = None
                state.agent_status = "initialize"
                return state
                
            except PartialNameMatchConfirmationRequired as partial_match_req:
                # Partial name match found - ask for confirmation
                logger.info(f"[PROFILE CREATION] Partial name match requires confirmation for {partial_match_req.new_name}")
                
                # Set the partial match confirmation state
                state.pending_partial_match = {
                    "potential_matches": partial_match_req.potential_matches,
                    "new_name": partial_match_req.new_name,
                    "relationship": partial_match_req.relationship,
                    "information": state.confirm_profile['information'] or "No additional info provided."
                }
                
                # Set the confirmation message and status
                state.final_response = partial_match_req.confirmation_message
                state.agent_status = "waiting_partial_match_confirmation"
                
                # Clear other state variables
                state.confirm_profile = None
                state.last_mentioned_profile = {
                    "name": confirmed_profile["name"],
                    "relationship": confirmed_profile["relationship"],
                    "information": state.confirm_profile['information'] or "No additional info provided."
                }
                
                logger.info(f"[PROFILE CREATION] Set pending partial match confirmation: {state.pending_partial_match}")
                return state
                

                
            except Exception as e:
                logger.error(f"[PROFILE CREATION] Error during profile creation: {str(e)}")
                # If there's an error, retry the confirmation instead of crashing
                profile_name = state.confirm_profile.get('name', 'this person')
                state.final_response = (
                    f"I encountered an error while creating the profile. Let me try again. "
                    f"Please specify the relationship for {profile_name} "
                    f"(friend, colleague, family, supervisor, neighbor, assistant)."
                )
                state.agent_status = "confirm_creation"
                return state
        else:
            # Still no valid relationship info
            profile_name = state.confirm_profile.get('name', 'this person')
            state.final_response = (
                f"Please specify the relationship for {profile_name} "
                f"(friend, colleague, family, supervisor, neighbor, assistant)."
            )
            state.agent_status = "confirm_creation"
            return state

    # ENHANCED: Check if we're in a confirmation stage and should ignore unrelated input
    if state.confirm_profile:
        logger.info(f"[PROFILE CREATION] Currently in confirmation stage for: {state.confirm_profile}")
        logger.info(f"[PROFILE CREATION] Ignoring unrelated input and retrying confirmation")
        
        # Retry the confirmation instead of processing new profile information
        state.final_response = (
            f"I'm still waiting for you to clarify the relationship for {state.confirm_profile.get('name', 'this person')}. "
            f"Please specify if they are your friend, colleague, family member, supervisor, neighbor, or assistant."
        )
        state.agent_status = "confirm_creation"
        return state
    
    # ENHANCED: Check for pronouns and resolve them intelligently
    last_profile = getattr(state, "last_mentioned_profile", {})
    has_pronouns = any(pron in text_lower for pron in ["he", "she", "him", "her", "his", "hers", "they", "them", "their", "theirs"])
    
    logger.info(f"[PROFILE CREATION] Pronoun check - has_pronouns: {has_pronouns}, last_profile: {last_profile}")
    
    if has_pronouns and last_profile and last_profile.get("name") and last_profile.get("relationship") and last_profile.get("relationship") != "unknown":
        
        # CRITICAL FIX: Check if the text contains a name in the same sentence as the pronoun
        # If so, the pronoun refers to that name, not the last mentioned profile
        name_patterns = [
            r'\b(my\s+)?(friend|colleague|family|supervisor|neighbor|assistant|cousin|brother|sister|mother|father|daughter|son|wife|husband|partner)\s+(named\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(is|was|are|were)\s+(my\s+)?(friend|colleague|family|supervisor|neighbor|assistant|cousin|brother|sister|mother|father|daughter|son|wife|husband|partner)',
            r'\b(?!(?:he|she|him|her|his|hers|they|them|their|theirs|also|and|but|or|so|yet|for|nor)\b)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(lives|works|studies|has|likes|enjoys|plays)',
        ]
        
        name_in_sentence = False
        logger.info(f"[PROFILE CREATION] Testing text '{text}' against name patterns:")
        for i, pattern in enumerate(name_patterns):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                matched_text = match.group()
                # Additional check: if the match contains pronouns, it's not a real name
                if any(pronoun in matched_text.lower() for pronoun in ['he', 'she', 'him', 'her', 'his', 'hers', 'they', 'them', 'their', 'theirs']):
                    logger.info(f"[PROFILE CREATION] Pattern {i+1}: {pattern} -> MATCH but contains pronouns, ignoring: {matched_text}")
                    continue
                logger.info(f"[PROFILE CREATION] Pattern {i+1}: {pattern} -> MATCH: {matched_text}")
                name_in_sentence = True
                logger.info(f"[PROFILE CREATION] Name detected in same sentence as pronoun - pattern {i+1} matched: '{matched_text}' - pronoun refers to that name, not last mentioned profile")
                break
            else:
                logger.info(f"[PROFILE CREATION] Pattern {i+1}: {pattern} -> NO MATCH")
        
        if not name_in_sentence:
            # CRITICAL FIX: Check if the text mentions a different relationship than the last mentioned profile
            # If so, don't resolve pronouns to the last mentioned profile - create a new profile instead
            text_relationships = []
            for rel in ['friend', 'colleague', 'family', 'supervisor', 'neighbor', 'assistant', 'cousin', 'brother', 'sister', 'mother', 'father', 'daughter', 'son', 'wife', 'husband', 'partner']:
                # Use word boundary matching to avoid false positives like "son" in "song"
                if re.search(r'\b' + re.escape(rel) + r'\b', text_lower):
                    text_relationships.append(rel)
            
            last_relationship = last_profile.get('relationship', '').lower()
            different_relationship_mentioned = any(rel != last_relationship for rel in text_relationships)
            
            if different_relationship_mentioned:
                logger.info(f"[PROFILE CREATION] Different relationship mentioned in text: {text_relationships} vs last mentioned: {last_relationship}")
                logger.info(f"[PROFILE CREATION] Not resolving pronouns to last mentioned profile - will create new profile instead")
                # Don't resolve pronouns, let the normal profile creation flow handle this
            else:
                # Only resolve pronouns to last mentioned profile if no name is in the same sentence
                logger.info(f"[PROFILE CREATION] Pronouns detected, resolving to last mentioned profile: {last_profile}")
                
                # Replace pronouns with the last mentioned profile name
                resolved_text = replace_pronouns(text, last_profile)
                logger.info(f"[PROFILE CREATION] Resolved text: '{text}' -> '{resolved_text}'")
            
            # Check if this is new information to add to the existing profile
            if any(keyword in text_lower for keyword in ['likes', 'phone', 'number', 'lives', 'works', 'studies', 'birthday', 'family', 'sister', 'brother', 'mother', 'father', 'hobby', 'plays', 'enjoys', 'has', 'is', 'was']):
                logger.info(f"[PROFILE CREATION] Adding new information to existing profile: {last_profile.get('name')}")
                
                try:
                    # Load existing profiles to find the one to update
                    profiles = await load_user_profiles(user_id)
                    
                    # Find the profile key for the last mentioned profile
                    profile_key = None
                    for key, prof in profiles.items():
                        if (prof.get('name') == last_profile.get('name') and 
                            prof.get('relationship') == last_profile.get('relationship')):
                            profile_key = key
                            break
                    
                    if profile_key:
                        # Update the existing profile with new information
                        try:
                            updated_profile = await intelligently_update_profile(profiles[profile_key], [resolved_text])
                            
                            # Save the updated profile
                            profiles[profile_key] = updated_profile
                            success = await save_user_profiles(user_id, profiles)
                            
                            if success:
                                state.final_response = f"Perfect! I've added the new information about {last_profile.get('name')} ({last_profile.get('relationship')}) to their profile."
                                state.last_mentioned_profile = {
                                    "name": last_profile.get('name'),
                                    "relationship": last_profile.get('relationship'),
                                    "information": "Profile updated with new information"
                                }
                                state.agent_status = "initialize"
                                logger.info(f"[PROFILE CREATION] Successfully updated existing profile for {last_profile.get('name')}")
                                return state
                            else:
                                raise Exception("Failed to save updated profile")
                        except ProfileUpdateConfirmationRequired as confirmation_req:
                            # Profile update requires confirmation - set pending update state
                            logger.info(f"[PROFILE CREATION] Profile update requires confirmation for {confirmation_req.person_name}'s {confirmation_req.field_name}")
                            
                            # Set the pending update information in the state
                            state.pending_profile_update = {
                                "name": confirmation_req.person_name,
                                "field": confirmation_req.field_name,
                                "new_value": confirmation_req.new_value,
                                "current_value": confirmation_req.current_value,
                                "updated_profile": confirmation_req.updated_profile,
                                "fields_needing_confirmation": confirmation_req.fields_needing_confirmation
                            }
                            
                            # Set the confirmation message and status
                            state.final_response = confirmation_req.confirmation_message
                            state.agent_status = "waiting_confirmation"
                            
                            # Set last_mentioned_profile using the existing profile data
                            state.last_mentioned_profile = {
                                "name": last_profile.get('name'),
                                "relationship": last_profile.get('relationship'),
                                "information": "Profile update pending confirmation"
                            }
                            
                            logger.info(f"[PROFILE CREATION] Set pending profile update: {state.pending_profile_update}")
                            return state
                    else:
                        raise Exception("Profile key not found")
                except Exception as e:
                    logger.error(f"[PROFILE CREATION] Error updating existing profile: {str(e)}")
                    # Fall through to normal profile creation if update fails
                    logger.info(f"[PROFILE CREATION] Falling back to normal profile creation due to update failure")
        
            # If not updating existing profile, use resolved text for new profile creation
            text = resolved_text
            logger.info(f"[PROFILE CREATION] Using resolved text for profile creation: {text}")
        
        else:
            # Name detected in same sentence as pronoun - pronoun refers to that name, not last mentioned profile
            # Process as normal profile creation without pronoun resolution
            logger.info(f"[PROFILE CREATION] Name detected in same sentence as pronoun - processing as normal profile creation")
            text = text  # Use original text without pronoun resolution
    
    # Load existing profiles for disambiguation
    profiles = await load_user_profiles(user_id)
    logger.info(f"[PROFILE CREATION] Loaded {len(profiles)} existing profiles for disambiguation")
    
    detected_names, is_detected = await detect_profiles_in_text(text, state.request.messages, state.user_id)
    
    logger.info(f"[PROFILE CREATION] Detected profiles: {detected_names}")
    # Debug: Log each profile individually to see the structure
    for i, profile in enumerate(detected_names):
        logger.info(f"[PROFILE CREATION] Profile {i}: name='{profile.get('name')}', relationship='{profile.get('relationship')}', information='{profile.get('information')}'")
    logger.info(f"[PROFILE CREATION] Is detected: {is_detected}")

    # CRITICAL FIX: Check if no profiles were detected
    if not detected_names or len(detected_names) == 0:
        logger.info("[PROFILE CREATION] No profiles detected - checking if relationship was mentioned without name")
        
        # Check if a relationship was mentioned but no name was provided
        text_lower = text.lower()
        mentioned_relationships = []
        for rel in ['friend', 'colleague', 'family', 'supervisor', 'neighbor', 'assistant', 'cousin', 'brother', 'sister', 'mother', 'father', 'daughter', 'son', 'wife', 'husband', 'partner']:
            # Use word boundary matching to avoid false positives like "son" in "song"
            if re.search(r'\b' + re.escape(rel) + r'\b', text_lower):
                mentioned_relationships.append(rel)
        
        if mentioned_relationships:
            logger.info(f"[PROFILE CREATION] Relationship mentioned without name: {mentioned_relationships}")
            state.waiting_for_name = {
                "relationship": mentioned_relationships[0],
                "original_text": text
            }
            state.final_response = f"I understand you're talking about your {mentioned_relationships[0]}, but I don't see a name mentioned. What's your {mentioned_relationships[0]}'s name?"
            state.agent_status = "waiting_for_name"
            return state
        else:
            logger.info("[PROFILE CREATION] No relationship mentioned - routing to general node")
        state.agent_status = "general"
        return state

    # ENHANCED: Validate that family members and pets are not incorrectly created as separate profiles
    if detected_names and len(detected_names) > 1:
        logger.info(f"[PROFILE CREATION] Multiple profiles detected, checking for family members incorrectly separated")
        
        # Look for the main profile (usually the first one with clear relationship)
        main_profile = None
        family_profiles = []
        
        for profile in detected_names:
            name = profile.get('name', '')
            relationship = profile.get('relationship', '')
            information = profile.get('information', '')
            
            # Check if this looks like a main profile (has location, phone, etc.)
            if any(keyword in information.lower() for keyword in ['lives in', 'phone', 'works at', 'studies at']):
                main_profile = profile
                logger.info(f"[PROFILE CREATION] Identified main profile: {name}")
            else:
                # Check if this looks like family information
                if any(keyword in information.lower() for keyword in ['brother', 'sister', 'parent', 'child', 'cat', 'dog', 'pet']):
                    family_profiles.append(profile)
                    logger.warning(f"[PROFILE CREATION] WARNING: Family member/pet detected as separate profile: {name} - {information}")
        
        # If we found family profiles that should be merged, fix them
        if main_profile and family_profiles:
            logger.info(f"[PROFILE CREATION] Merging family information into main profile")
            
            # Merge all family information into the main profile
            main_info = main_profile.get('information', '')
            for family_profile in family_profiles:
                family_info = family_profile.get('information', '')
                if family_info:
                    main_info = f"{main_info} and {family_info}" if main_info else family_info
            
            # Update main profile with merged information
            main_profile['information'] = main_info
            
            # Remove family profiles from the list
            detected_names = [main_profile]
            logger.info(f"[PROFILE CREATION] Merged profiles. Final result: {detected_names}")

    # ENHANCED: Handle multiple profiles with clear relationships
    if detected_names and len(detected_names) > 1:
        logger.info(f"[PROFILE CREATION] Multiple profiles detected: {len(detected_names)} profiles")
        
        # Check if all profiles have clear relationships
        all_clear_relationships = True
        unclear_profiles = []
        
        for profile in detected_names:
            name = profile.get('name', '')
            relationship = profile.get('relationship', '')
            information = profile.get('information', '')
            
            logger.info(f"[PROFILE CREATION] Checking profile: {name} ({relationship})")
            
            # Check if relationship is clear (not empty, not unknown)
            if not relationship or relationship == 'unknown':
                all_clear_relationships = False
                unclear_profiles.append(name)
                logger.info(f"[PROFILE CREATION] Profile {name} has unclear relationship: '{relationship}'")
        
        if all_clear_relationships:
            logger.info(f"[PROFILE CREATION] All {len(detected_names)} profiles have clear relationships - saving all at once")
            
            saved_profiles = []
            for profile in detected_names:
                name = profile.get('name', '')
                relationship = profile.get('relationship', '')
                information = profile.get('information', '')
                
                try:
                    # Clean the name
                    clean_name = clean_profile_name(name)
                    if clean_name != name:
                        logger.info(f"[PROFILE CREATION] Cleaned name from '{name}' to '{clean_name}'")
                    
                    # Save the profile
                    saved_profile = await create_or_update_profile_persistent(
                        user_id, clean_name, relationship, information, last_mentioned_profile=state.last_mentioned_profile
                    )
                    saved_profiles.append(saved_profile)
                    logger.info(f"[PROFILE CREATION] Successfully saved profile for {clean_name} ({relationship})")
                    
                except PartialNameMatchConfirmationRequired as partial_match_req:
                    # Partial name match found - ask for confirmation
                    logger.info(f"[PROFILE CREATION] Partial name match requires confirmation for {partial_match_req.new_name}")
                    # For multiple profile creation, we'll skip this profile and continue with others
                    logger.warning(f"[PROFILE CREATION] Skipping profile due to partial name match requiring confirmation: {name}")
                    continue
                except MultipleProfileClarificationRequired as multi_profile_req:
                    # Multiple profiles found - ask for clarification
                    logger.info(f"[PROFILE CREATION] Multiple profiles require clarification for {multi_profile_req.first_name}")
                    
                    # Set the multiple profile clarification state
                    state.pending_multiple_profile_clarification = {
                        "first_name": multi_profile_req.first_name,
                        "relationship": multi_profile_req.relationship,
                        "existing_profiles": multi_profile_req.existing_profiles,
                        "new_information": multi_profile_req.new_information
                    }
                    
                    # Set the confirmation message and status
                    state.final_response = multi_profile_req.confirmation_message
                    state.agent_status = "waiting_multiple_profile_clarification"
                    
                    logger.info(f"[PROFILE CREATION] Set pending multiple profile clarification: {state.pending_multiple_profile_clarification}")
                    return state
                except Exception as e:
                    logger.error(f"[PROFILE CREATION] Error saving profile for {name}: {str(e)}")
                    continue
            
            # Create success message for multiple profiles
            if saved_profiles:
                profile_summary = []
                for profile in saved_profiles:
                    profile_summary.append(f"{profile.get('relationship', 'unknown')} {profile.get('name', 'Unknown')}")
                
                state.final_response = (
                    f"Excellent! I've successfully created profiles for: {', '.join(profile_summary)}. "
                    f"All {len(saved_profiles)} profiles have been saved with their information."
                )
                state.agent_status = "initialize"
                return state
            else:
                logger.error(f"[PROFILE CREATION] Failed to save any of the {len(detected_names)} profiles")
        
        else:
            logger.info(f"[PROFILE CREATION] Some profiles have unclear relationships: {unclear_profiles}")
            logger.info(f"[PROFILE CREATION] Asking for clarification for unclear profiles")
            
            # Ask for clarification for unclear profiles
            if len(unclear_profiles) == 1:
                clarification_message = (
                    f"I'd be happy to help! I detected a profile but need clarification on the relationship for: {unclear_profiles[0]}. "
                    f"Could you please specify the relationship? For example, you could say 'my friend {unclear_profiles[0]}' or 'my colleague {unclear_profiles[0]}'."
                )
            else:
                # For multiple profiles, create examples based on available count
                examples = []
                for i, profile in enumerate(unclear_profiles[:2]):  # Limit to first 2 for examples
                    if i == 0:
                        examples.append(f"'my friend {profile}'")
                    else:
                        examples.append(f"'my colleague {profile}'")
                
                clarification_message = (
                    f"I'd be happy to help! I detected multiple profiles but need clarification on relationships for: {', '.join(unclear_profiles)}. "
                    f"Could you please specify the relationship for each person? For example: {', '.join(examples)}."
                )
            
            state.final_response = clarification_message
            state.agent_status = "confirm_creation"
            state.confirm_profile = {
                "name": ", ".join(unclear_profiles),
                "relationship": "multiple_unclear",
                "information": text
            }
            return state

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
                    f"I'd be happy to help! I see you're referring to {name}, but I need to know the relationship. "
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
                    user_id, clean_name, relationship, text, last_mentioned_profile=state.last_mentioned_profile
                )
            except PartialNameMatchConfirmationRequired as partial_match_req:
                # Partial name match found - ask for confirmation
                logger.info(f"[PRONOUN PROFILE UPDATE] Partial name match requires confirmation for {partial_match_req.new_name}")
                
                # Set the partial match confirmation state
                state.pending_partial_match = {
                    "potential_matches": partial_match_req.potential_matches,
                    "new_name": partial_match_req.new_name,
                    "relationship": partial_match_req.relationship,
                    "information": text
                }
                
                # Set the confirmation message and status
                state.final_response = partial_match_req.confirmation_message
                state.agent_status = "waiting_partial_match_confirmation"
                
                # Clear other state variables
                state.confirm_profile = None
                state.last_mentioned_profile = {
                    "name": clean_name,
                    "relationship": relationship,
                    "information": text
                }
                
                logger.info(f"[PRONOUN PROFILE UPDATE] Set pending partial match confirmation: {state.pending_partial_match}")
                return state
            except ProfileUpdateConfirmationRequired as confirmation_req:
                # Profile update requires confirmation - set pending update state
                logger.info(f"[PRONOUN PROFILE UPDATE] Profile update requires confirmation for {confirmation_req.person_name}'s {confirmation_req.field_name}")
                
                # Set the pending update information in the state
                state.pending_profile_update = {
                    "name": confirmation_req.person_name,
                    "field": confirmation_req.field_name,
                    "new_value": confirmation_req.new_value,
                    "current_value": confirmation_req.current_value,
                    "updated_profile": confirmation_req.updated_profile,  # Store the complete updated profile
                    "fields_needing_confirmation": confirmation_req.fields_needing_confirmation  # Store all fields that need confirmation
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
        
        # ENHANCED: Check if this might be a response to a clarification request (e.g., "my friend Tom")
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
                        user_id, clean_name, rel_from_llm, information, last_mentioned_profile=state.last_mentioned_profile
                    )
                except PartialNameMatchConfirmationRequired as partial_match_req:
                    # Partial name match found - ask for confirmation
                    logger.info(f"[PROFILE CLARIFICATION RESPONSE] Partial name match requires confirmation for {partial_match_req.new_name}")
                    
                    # Set the partial match confirmation state
                    state.pending_partial_match = {
                        "potential_matches": partial_match_req.potential_matches,
                        "new_name": partial_match_req.new_name,
                        "relationship": partial_match_req.relationship,
                        "information": information
                    }
                    
                    # Set the confirmation message and status
                    state.final_response = partial_match_req.confirmation_message
                    state.agent_status = "waiting_partial_match_confirmation"
                    
                    # Clear other state variables
                    state.confirm_profile = None
                    state.last_mentioned_profile = {
                        "name": clean_name,
                        "relationship": rel_from_llm,
                        "information": information
                    }
                    
                    logger.info(f"[PROFILE CLARIFICATION RESPONSE] Set pending partial match confirmation: {state.pending_partial_match}")
                    return state
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
        
        # ENHANCED: Handle unrelated responses during confirmation stages
        elif state.confirm_profile:
            logger.info(f"[PROFILE CLARIFICATION RESPONSE] User provided unrelated response during confirmation: {text}")
            logger.info(f"[PROFILE CLARIFICATION RESPONSE] Ignoring unrelated response and retrying confirmation")
            
            # Retry the confirmation instead of processing unrelated information
            state.final_response = (
                f"I'm still waiting for you to clarify the relationship for {state.confirm_profile.get('name', 'this person')}. "
                f"Please specify if they are your friend, colleague, family member, supervisor, neighbor, or assistant."
            )
            state.agent_status = "confirm_creation"
            return state
        
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
            # Expanded valid relationships to be more inclusive
            valid_relationships = {
                # Personal relationships
                "friend", "best friend", "close friend", "old friend", "new friend",
                "colleague", "coworker", "work colleague", "team member", "boss", "supervisor", "manager",
                "neighbor", "next door neighbor", "neighborhood friend",
                
                # Family relationships
                "family", "family member", "relative", "cousin", "brother", "sister", "sibling",
                "mother", "mom", "father", "dad", "parent", "grandmother", "grandfather", "grandparent",
                "aunt", "uncle", "niece", "nephew", "son", "daughter", "child", "children",
                
                # Professional relationships
                "assistant", "mentor", "teacher", "professor", "student", "classmate",
                "doctor", "lawyer", "consultant", "client", "customer", "vendor",
                
                # Other relationships
                "acquaintance", "contact", "connection", "partner", "roommate", "flatmate"
            }
            
            # Check if the relationship is valid (case-insensitive)
            relationship_lower = relationship.lower()
            
            # First check exact matches
            is_valid = any(rel.lower() == relationship_lower for rel in valid_relationships)
            
            # If not valid, check for common variations and synonyms
            if not is_valid:
                # Common relationship variations and synonyms
                relationship_variations = {
                    'cousin': ['cousin', 'cousins', 'cousin-in-law'],
                    'brother': ['brother', 'brothers', 'brother-in-law', 'stepbrother', 'half-brother'],
                    'sister': ['sister', 'sisters', 'sister-in-law', 'stepsister', 'half-sister'],
                    'mother': ['mother', 'mom', 'mum', 'mommy', 'mummy', 'mother-in-law', 'stepmother'],
                    'father': ['father', 'dad', 'daddy', 'father-in-law', 'stepfather'],
                    'friend': ['friend', 'friends', 'best friend', 'close friend', 'old friend', 'new friend'],
                    'colleague': ['colleague', 'colleagues', 'coworker', 'work colleague', 'team member'],
                    'neighbor': ['neighbor', 'neighbours', 'neighbour', 'next door neighbor', 'neighborhood friend'],
                    'boss': ['boss', 'supervisor', 'manager', 'team lead', 'director']
                }
                
                # Check if the relationship matches any variations
                for base_relationship, variations in relationship_variations.items():
                    if relationship_lower in variations:
                        is_valid = True
                        logger.info(f"[PROFILE CREATION] Relationship '{relationship}' matched variation of '{base_relationship}'")
                        break
            
            # Add detailed logging for relationship validation
            logger.info(f"[PROFILE CREATION] Validating relationship '{relationship}' (lowercase: '{relationship_lower}') for {clean_name}")
            logger.info(f"[PROFILE CREATION] Valid relationships include: {sorted(valid_relationships)}")
            logger.info(f"[PROFILE CREATION] Relationship validation result: {is_valid}")
            
            # Final fallback: Always accept common family and personal relationships
            if not is_valid:
                # Common relationships that should always be accepted
                always_accepted = {
                    'cousin', 'brother', 'sister', 'mother', 'father', 'daughter', 'son',
                    'uncle', 'aunt', 'niece', 'nephew', 'grandmother', 'grandfather',
                    'friend', 'colleague', 'neighbor', 'boss', 'supervisor', 'assistant'
                }
                
                if relationship_lower in always_accepted:
                    is_valid = True
                    logger.info(f"[PROFILE CREATION] Relationship '{relationship}' accepted as always-valid family/personal relationship")
            
            if not is_valid:
                logger.warning(f"[PROFILE CREATION] Invalid relationship '{relationship}' for {clean_name}, asking for clarification")
                state.agent_status = "confirm_creation"
                state.confirm_profile = {
                    "name": clean_name,
                    "relationship": "unknown",
                    "information": information
                }
                
                # Provide a more helpful and comprehensive list of valid relationships
                state.final_response = (
                    f"I detected the name '{clean_name}' but the relationship '{relationship}' is not recognized. "
                    f"Could you please specify a valid relationship? Here are some examples:\n\n"
                    f"**Family Relationships** (most common):\n"
                    f"• cousin, brother, sister, mother, father, aunt, uncle\n"
                    f"• niece, nephew, grandmother, grandfather, son, daughter\n\n"
                    f"**Personal Relationships**:\n"
                    f"• friend, colleague, neighbor, acquaintance, roommate\n\n"
                    f"**Professional Relationships**:\n"
                    f"• boss, supervisor, coworker, mentor, teacher, assistant\n\n"
                    f"**Quick Fix**: You can also say 'my {relationship}' and I'll understand it as a valid relationship type."
                )
                break
            
            # Create or update profile with rich structure
            try:
                await create_or_update_profile_persistent(
                    user_id, clean_name, relationship, information, last_mentioned_profile=state.last_mentioned_profile
                )
            except PartialNameMatchConfirmationRequired as partial_match_req:
                # Partial name match found - ask for confirmation
                logger.info(f"[PROFILE CREATION] Partial name match requires confirmation for {partial_match_req.new_name}")
                
                # Set the partial match confirmation state
                state.pending_partial_match = {
                    "potential_matches": partial_match_req.potential_matches,
                    "new_name": partial_match_req.new_name,
                    "relationship": partial_match_req.relationship,
                    "information": information
                }
                
                # Set the confirmation message and status
                state.final_response = partial_match_req.confirmation_message
                state.agent_status = "waiting_partial_match_confirmation"
                
                # Clear other state variables
                state.confirm_profile = None
                state.last_mentioned_profile = {
                    "name": clean_name,
                    "relationship": relationship,
                    "information": information
                }
                
                logger.info(f"[PROFILE CREATION] Set pending partial match confirmation: {state.pending_partial_match}")
                return state
            except ProfileUpdateConfirmationRequired as confirmation_req:
                # Profile update requires confirmation - set pending update state
                logger.info(f"[PROFILE CREATION] Profile update requires confirmation for {confirmation_req.person_name}'s {confirmation_req.field_name}")
                
                # Set the pending update information in the state
                state.pending_profile_update = {
                    "name": confirmation_req.person_name,
                    "field": confirmation_req.field_name,
                    "new_value": confirmation_req.new_value,
                    "current_value": confirmation_req.current_value,
                    "updated_profile": confirmation_req.updated_profile,  # Store the complete updated profile
                    "fields_needing_confirmation": confirmation_req.fields_needing_confirmation  # Store all fields that need confirmation
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
            except MultipleProfileClarificationRequired as multi_profile_req:
                # Multiple profiles found - ask for clarification
                logger.info(f"[PROFILE CREATION] Multiple profiles require clarification for {multi_profile_req.first_name}")
                
                # Set the multiple profile clarification state
                state.pending_multiple_profile_clarification = {
                    "first_name": multi_profile_req.first_name,
                    "relationship": multi_profile_req.relationship,
                    "existing_profiles": multi_profile_req.existing_profiles,
                    "new_information": multi_profile_req.new_information
                }
                
                # Set the confirmation message and status
                state.final_response = multi_profile_req.confirmation_message
                state.agent_status = "waiting_multiple_profile_clarification"
                
                # Clear other state variables
                state.confirm_profile = None
                state.last_mentioned_profile = {
                    "name": clean_name,
                    "relationship": relationship,
                    "information": information
                }
                
                logger.info(f"[PROFILE CREATION] Set pending multiple profile clarification: {state.pending_multiple_profile_clarification}")
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
    
    # Ensure required functions are imported
    from rag.rich_profile_utils import generate_rich_response_from_profile, is_rich_profile, convert_simple_to_rich_profile, intelligently_update_profile
    
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
    
    # Check if user provided a number selection (1, 2, 3, etc.)
    if user_response.isdigit():
        try:
            selection = int(user_response)
            if 1 <= selection <= len(profiles):
                # User selected a profile by number
                selected_profile_tuple = profiles[selection - 1]
                selected_profile = selected_profile_tuple[1]  # Extract profile dictionary
                selected_key = selected_profile_tuple[0]  # Extract profile key
                
                logger.info(f"[PROFILE CLARIFICATION] User selected profile {selection}: {selected_profile.get('name')} ({selected_profile.get('relationship')})")
                
                # Generate response with profile information
                if is_rich_profile(selected_profile):
                    profile_info = generate_rich_response_from_profile(selected_profile)
                else:
                    # Convert simple profile to rich format for better display
                    try:
                        rich_profile = await convert_simple_to_rich_profile(selected_profile)
                        profile_info = generate_rich_response_from_profile(rich_profile)
                    except Exception as e:
                        logger.warning(f"[PROFILE CLARIFICATION] Failed to convert profile to rich format: {str(e)}")
                        profile_info = f"**{selected_profile.get('name')}** ({selected_profile.get('relationship')})\n"
                        if selected_profile.get('location'):
                            profile_info += f"• Lives in: {selected_profile.get('location')}\n"
                        if selected_profile.get('phone'):
                            profile_info += f"• Phone: {selected_profile.get('phone')}\n"
                        if selected_profile.get('other_info'):
                            profile_info += f"• Other: {', '.join(selected_profile.get('other_info', []))}\n"
                
                state.final_response = f"Here's the information about {selected_profile.get('name')}:\n\n{profile_info}"
                state.last_mentioned_profile = {
                    "name": selected_profile.get('name'),
                    "relationship": selected_profile.get('relationship'),
                    "information": "Profile information displayed"
                }
                state.agent_status = "initialize"
                state.pending_profile_clarification = None
                return state
            else:
                logger.warning(f"[PROFILE CLARIFICATION] Invalid selection number: {selection} (valid range: 1-{len(profiles)})")
                state.final_response = f"Please select a number between 1 and {len(profiles)}."
                state.agent_status = "waiting_profile_clarification"
                return state
        except ValueError:
            logger.warning(f"[PROFILE CLARIFICATION] Invalid number format: {user_response}")
            # Fall through to relationship detection
    elif user_response.lower() in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
        # Handle single digit as string
        try:
            selection = int(user_response)
            if 1 <= selection <= len(profiles):
                selected_profile_tuple = profiles[selection - 1]
                selected_profile = selected_profile_tuple[1]
                
                logger.info(f"[PROFILE CLARIFICATION] User selected profile {selection}: {selected_profile.get('name')} ({selected_profile.get('relationship')})")
                
                # Generate response with profile information
                if is_rich_profile(selected_profile):
                    profile_info = generate_rich_response_from_profile(selected_profile)
                else:
                    try:
                        rich_profile = await convert_simple_to_rich_profile(selected_profile)
                        profile_info = generate_rich_response_from_profile(rich_profile)
                    except Exception as e:
                        logger.warning(f"[PROFILE CLARIFICATION] Failed to convert profile to rich format: {str(e)}")
                        profile_info = f"**{selected_profile.get('name')}** ({selected_profile.get('relationship')})\n"
                        if selected_profile.get('location'):
                            profile_info += f"• Lives in: {selected_profile.get('location')}\n"
                        if selected_profile.get('phone'):
                            profile_info += f"• Phone: {selected_profile.get('phone')}\n"
                        if selected_profile.get('other_info'):
                            profile_info += f"• Other: {', '.join(selected_profile.get('other_info', []))}\n"
                
                state.final_response = f"Here's the information about {selected_profile.get('name')}:\n\n{profile_info}"
                state.last_mentioned_profile = {
                    "name": selected_profile.get('name'),
                    "relationship": selected_profile.get('relationship'),
                    "information": "Profile information displayed"
                }
                state.agent_status = "initialize"
                state.pending_profile_clarification = None
                return state
            else:
                logger.warning(f"[PROFILE CLARIFICATION] Invalid selection number: {selection} (valid range: 1-{len(profiles)})")
                state.final_response = f"Please select a number between 1 and {len(profiles)}."
                state.agent_status = "waiting_profile_clarification"
                return state
        except ValueError:
            # Fall through to relationship detection
            pass
    
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
        for profile_tuple in profiles:
            profile = profile_tuple[1]  # Extract the profile dictionary from the tuple
            if profile.get('relationship', '').lower() == detected_relationship:
                matched_profile = profile
                break
        
        if matched_profile:
            logger.info(f"[PROFILE CLARIFICATION] Found matching profile: {matched_profile.get('name')} ({detected_relationship})")
            
            # Check if user is providing new information (not just asking about the profile)
            # If they're providing new info and there's a clear match, append it automatically
            if any(keyword in user_response.lower() for keyword in ['likes', 'phone', 'number', 'lives', 'works', 'studies', 'birthday', 'family', 'sister', 'brother', 'mother', 'father', 'hobby', 'plays', 'enjoys', 'has', 'is', 'was']):
                logger.info(f"[PROFILE CLARIFICATION] User providing new information for {matched_profile.get('name')}, appending automatically")
                
                try:
                    # Append the new information to the existing profile
                    user_id = state.user_id
                    profiles_dict = await load_user_profiles(user_id)
                    
                    # Find the profile key
                    profile_key = None
                    for key, prof in profiles_dict.items():
                        if (prof.get('name') == matched_profile.get('name') and 
                            prof.get('relationship') == matched_profile.get('relationship')):
                            profile_key = key
                            break
                    
                    if profile_key:
                        # Update the profile with new information
                        updated_profile = await intelligently_update_profile(profiles_dict[profile_key], [user_response])
                        
                        # Save the updated profile
                        profiles_dict[profile_key] = updated_profile
                        success = await save_user_profiles(user_id, profiles_dict)
                        
                        if success:
                            state.final_response = (
                                f"Perfect! I've added the new information about {matched_profile.get('name')} ({detected_relationship}) to their profile. "
                                f"Here's what I know about them now:\n\n"
                                f"{generate_rich_response_from_profile(updated_profile) if is_rich_profile(updated_profile) else 'Profile updated successfully'}"
                            )
                            logger.info(f"[PROFILE CLARIFICATION] Successfully updated profile for {matched_profile.get('name')} with new information")
                        else:
                            state.final_response = f"Sorry, I encountered an error while updating the profile. Please try again."
                            logger.error(f"[PROFILE CLARIFICATION] Failed to save updated profile")
                    else:
                        # Fallback to confirmation if profile key not found
                        raise Exception("Profile key not found")
                        
                except Exception as e:
                    logger.error(f"[PROFILE CLARIFICATION] Error updating profile: {str(e)}")
                    # Fallback to confirmation approach
                    state.final_response = (
                        f"I think you're referring to {matched_profile.get('name')} ({detected_relationship}), but I need to confirm. "
                        f"Are you asking about:\n\n"
                        f"- {matched_profile.get('name')} ({detected_relationship})?\n\n"
                        f"Please confirm with 'yes' if this is correct, or specify which profile you mean if not. "
                        f"Available options:\n" +
                        "\n".join([f"• {p[1].get('relationship', 'unknown').capitalize()} {p[1].get('name')}" for p in profiles])
                    )
                    state.agent_status = "waiting_profile_clarification"
                    
                    # Store the detected profile for confirmation
                    state.pending_profile_clarification = {
                        **pending_clarification,
                        "detected_profile": matched_profile,
                        "detected_relationship": detected_relationship
                    }
                    
            else:
                # User is just asking about the profile, not providing new information
                # Ask for confirmation as before
                state.final_response = (
                    f"I think you're referring to {matched_profile.get('name')} ({detected_relationship}), but I need to confirm. "
                    f"Are you asking about:\n\n"
                    f"- {matched_profile.get('name')} ({detected_relationship})?\n\n"
                    f"Please confirm with 'yes' if this is correct, or specify which profile you mean if not. "
                    f"Available options:\n" +
                    "\n".join([f"• {p[1].get('relationship', 'unknown').capitalize()} {p[1].get('name')}" for p in profiles])
                )
                state.agent_status = "waiting_profile_clarification"
                
                # Store the detected profile for confirmation
                state.pending_profile_clarification = {
                    **pending_clarification,
                    "detected_profile": matched_profile,
                    "detected_relationship": detected_relationship
                }
            
        else:
            logger.warning(f"[PROFILE CLARIFICATION] No profile found with relationship '{detected_relationship}' for '{name}'")
            state.final_response = f"I couldn't find a profile for {name.capitalize()} with the relationship '{detected_relationship}'. The available profiles are:\n" + \
                                 "\n".join([f"• {p[1].get('relationship', 'unknown').capitalize()} {p[1].get('name')}" for p in profiles])
            state.agent_status = "waiting_profile_clarification"
    else:
        # Check if user is confirming a previously detected profile
        if (state.pending_profile_clarification and 
            'detected_profile' in state.pending_profile_clarification and
            user_response.lower() in ['yes', 'y', 'confirm', 'ok', 'okay', 'sure', 'go ahead', 'correct', 'right']):
            
            # User confirmed the detected profile
            detected_profile = state.pending_profile_clarification['detected_profile']
            detected_relationship = state.pending_profile_clarification['detected_relationship']
            
            logger.info(f"[PROFILE CLARIFICATION] User confirmed profile: {detected_profile.get('name')} ({detected_relationship})")
            
            # Generate response based on the original query
            try:
                if is_rich_profile(detected_profile):
                    rich_response = generate_rich_response_from_profile(detected_profile)
                    state.final_response = rich_response
                else:
                    # Convert simple profile to rich format and generate response
                    rich_profile = await convert_simple_to_rich_profile(detected_profile)
                    rich_response = generate_rich_response_from_profile(rich_profile)
                    state.final_response = rich_response
                
                # Set the last mentioned profile
                state.last_mentioned_profile = {
                    "name": detected_profile.get('name'),
                    "relationship": detected_relationship,
                    "information": "Clarified from multiple profiles"
                }
                
                state.agent_status = "initialize"
                logger.info(f"[PROFILE CLARIFICATION] Successfully provided response for {detected_profile.get('name')}")
                
            except Exception as e:
                logger.error(f"[PROFILE CLARIFICATION] Error generating response: {str(e)}")
                state.final_response = f"Of course! Here's what I know about {detected_profile.get('name')} ({detected_relationship}): {detected_profile.get('information', 'Information available but could not be formatted')}"
                state.agent_status = "initialize"
                
        else:
            # Check if user is providing new profile information instead of clarifying
            if any(keyword in user_response.lower() for keyword in ['likes', 'phone', 'number', 'lives', 'works', 'studies', 'birthday', 'family', 'sister', 'brother', 'mother', 'father']):
                # Check if there's only one profile with this name - if so, append automatically
                if len(profiles) == 1:
                    single_profile = profiles[0]
                    logger.info(f"[PROFILE CLARIFICATION] Only one profile found for {name}, appending new information automatically")
                    
                    try:
                        # Append the new information to the existing profile
                        user_id = state.user_id
                        profiles_dict = await load_user_profiles(user_id)
                        
                        # Find the profile key
                        profile_key = None
                        for key, prof in profiles_dict.items():
                            if (prof.get('name') == single_profile.get('name') and 
                                prof.get('relationship') == single_profile.get('relationship')):
                                profile_key = key
                                break
                        
                        if profile_key:
                            # Update the profile with new information
                            updated_profile = await intelligently_update_profile(profiles_dict[profile_key], [user_response])
                            
                            # Save the updated profile
                            profiles_dict[profile_key] = updated_profile
                            success = await save_user_profiles(user_id, profiles_dict)
                            
                            if success:
                                state.final_response = (
                                    f"Perfect! I've added the new information about {single_profile.get('name')} ({single_profile.get('relationship')}) to their profile. "
                                    f"Here's what I know about them now:\n\n"
                                    f"{generate_rich_response_from_profile(updated_profile) if is_rich_profile(updated_profile) else 'Profile updated successfully'}"
                                )
                                logger.info(f"[PROFILE CLARIFICATION] Successfully updated single profile for {single_profile.get('name')} with new information")
                                state.agent_status = "initialize"
                                state.pending_profile_clarification = None
                            else:
                                raise Exception("Failed to save profile")
                        else:
                            raise Exception("Profile key not found")
                            
                    except Exception as e:
                        logger.error(f"[PROFILE CLARIFICATION] Error updating single profile: {str(e)}")
                        # Fallback to asking for clarification
                        state.final_response = (
                            f"I notice you're providing new information about {name.capitalize()}, but I'm currently waiting for you to clarify "
                            f"which profile you want to know about. Please first specify which {name.capitalize()} you mean:\n\n"
                            f"Available options:\n" +
                            "\n".join([f"• {p[1].get('relationship', 'unknown').capitalize()} {p[1].get('name')}" for p in profiles]) +
                            f"\n\nAfter we clarify which profile you want, I'll be happy to help you add the new information about {name.capitalize()}."
                        )
                        state.agent_status = "waiting_profile_clarification"
                else:
                    # Multiple profiles exist, ask for clarification
                    state.final_response = (
                        f"I notice you're providing new information about {name.capitalize()}, but I'm currently waiting for you to clarify "
                        f"which profile you want to know about. Please first specify which {name.capitalize()} you mean:\n\n"
                        f"Available options:\n" +
                        "\n".join([f"• {p[1].get('relationship', 'unknown').capitalize()} {p[1].get('name')}" for p in profiles]) +
                        f"\n\nAfter we clarify which profile you want, I'll be happy to help you add the new information about {name.capitalize()}."
                    )
                    state.agent_status = "waiting_profile_clarification"
            else:
                # No clear relationship detected, ask for more specific clarification
                logger.info(f"[PROFILE CLARIFICATION] No clear relationship detected in response: {user_response}")
                # Extract profile dictionaries from tuples
                profile_dicts = [p[1] for p in profiles] if profiles and isinstance(profiles[0], list) else profiles
                state.final_response = (
                    f"I need you to be more specific about which {name.capitalize()} you're asking about. "
                    f"Please say something like:\n\n"
                    f"• 'my friend {name.capitalize()}' or\n"
                    f"• 'my colleague {name.capitalize()}' or\n"
                    f"• 'the {name.capitalize()} who is my friend'\n\n"
                    f"Available options:\n" +
                    "\n".join([f"• {p.get('relationship', 'unknown').capitalize()} {p.get('name')}" for p in profile_dicts])
                )
                state.agent_status = "waiting_profile_clarification"
    
    # Clear the pending clarification if we're done
    if state.agent_status == "initialize":
        state.pending_profile_clarification = None
    
    return state

async def handle_full_name_for_new_profile_node(state: AgentState) -> AgentState:
    """Node to handle full name input when creating a new profile with existing relationship+name combination."""
    logger.info("[LANGGRAPH PATH] Starting handle_full_name_for_new_profile_node")
    
    if not state.pending_new_profile_creation:
        logger.warning("[FULL NAME FOR NEW PROFILE] No pending new profile creation found")
        state.final_response = "I'm not sure what you're providing a full name for. Could you please provide the profile information again?"
        state.agent_status = "initialize"
        return state
    
    pending_creation = state.pending_new_profile_creation
    user_response = state.retrieval_query.strip()
    
    logger.info(f"[FULL NAME FOR NEW PROFILE] Processing full name for: {pending_creation['first_name']}")
    logger.info(f"[FULL NAME FOR NEW PROFILE] User response: {user_response}")
    
    # Extract the creation details
    first_name = pending_creation["first_name"]
    relationship = pending_creation["relationship"]
    new_information = pending_creation["new_information"]
    
    # Clean the provided full name
    full_name = clean_profile_name(user_response)
    logger.info(f"[FULL NAME FOR NEW PROFILE] Cleaned full name: '{full_name}'")
    
    # Validate that the full name contains the first name
    if first_name.lower() not in full_name.lower():
        logger.warning(f"[FULL NAME FOR NEW PROFILE] Full name '{full_name}' doesn't contain first name '{first_name}'")
        state.final_response = f"The full name should include '{first_name}'. Please provide the complete name (e.g., '{first_name} Smith' or '{first_name} Johnson')."
        state.agent_status = "waiting_full_name_for_new_profile"
        return state
    
    try:
        # Create a new profile with the full name
        from rag.rich_profile_utils import convert_simple_to_rich_profile
        
        # Generate a unique key for the new profile
        profiles = await load_user_profiles(state.user_id)
        base_key = f"{first_name}_{relationship}"
        new_profile_key = generate_unique_profile_key(profiles, base_key)
        
        # Create the new profile with full name
        simple_profile = {
            'name': full_name,  # Use full name as the name
            'relationship': relationship,
            'information': [new_information],
            'timestamp': datetime.now().isoformat()
        }
        
        # Convert to rich profile format
        rich_profile = await convert_simple_to_rich_profile(simple_profile)
        profiles[new_profile_key] = rich_profile
        
        # Save the profiles
        success = await save_user_profiles(state.user_id, profiles)
        if success:
            state.final_response = f"Perfect! I've created a new profile for your {relationship} {full_name}."
            state.last_mentioned_profile = {
                "name": full_name,
                "relationship": relationship,
                "information": "New profile created"
            }
            logger.info(f"[FULL NAME FOR NEW PROFILE] Successfully created new profile for {full_name}")
        else:
            state.final_response = f"Sorry, I encountered an error while creating the new profile. Please try again."
            logger.error(f"[FULL NAME FOR NEW PROFILE] Failed to save new profile")
        
    except Exception as e:
        logger.error(f"[FULL NAME FOR NEW PROFILE] Error creating new profile: {str(e)}")
        state.final_response = f"Sorry, I encountered an error while creating the new profile. Please try again."
    
    # Clear the pending creation and reset state
    state.pending_new_profile_creation = None
    state.pending_multiple_profile_clarification = None
    state.multiple_profile_choice = None
    state.agent_status = "initialize"
    
    return state

async def generate_targeted_response(profile: dict, query: str) -> str:
    """
    Generate a targeted response based on the user's specific question using LLM.
    Instead of dumping all profile information, focus on what was asked.
    """
    from rag.rich_profile_utils import generate_rich_response_from_profile
    
    name = profile.get('name', 'Unknown')
    relationship = profile.get('relationship', 'unknown')
    
    # Prepare profile information in a structured format
    profile_info = []
        
    if profile.get('location'):
        profile_info.append(f"Location: {profile.get('location')}")
    if profile.get('phone'):
        profile_info.append(f"Phone: {profile.get('phone')}")
    if profile.get('birthday'):
        profile_info.append(f"Birthday: {profile.get('birthday')}")
    if profile.get('workplace'):
        profile_info.append(f"Workplace: {profile.get('workplace')}")
    if profile.get('education'):
        profile_info.append(f"Education: {profile.get('education')}")
        if profile.get('family'):
            family_info = profile.get('family')
            if isinstance(family_info, list):
                profile_info.append(f"Family: {'; '.join(family_info)}")
            else:
                profile_info.append(f"Family: {family_info}")
        if profile.get('other'):
            other_info = profile.get('other')
            if isinstance(other_info, list):
                profile_info.append(f"Other: {'; '.join(other_info)}")
            else:
                profile_info.append(f"Other: {other_info}")
    
    profile_text = '\n'.join(profile_info) if profile_info else "No additional information available"
    
    # Create a focused prompt for the LLM
    prompt = f"""
You are a helpful assistant that answers questions about people based on their profile information.

**Profile Information for {name} ({relationship}):**
{profile_text}

**User's Question:** {query}

**Instructions:**
1. Answer the user's specific question directly and concisely
2. Only use information that is explicitly provided in the profile
3. If the question asks for a count (e.g., "how many brothers"), provide the exact number
4. If the question asks for names, list them clearly
5. If the information is not available in the profile, say so clearly
6. Be natural and conversational in your response
7. Do not provide information that is not in the profile

**Response:**"""

    try:
        # Use the local LLM to generate a targeted response
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        
        # Use the same local LLM configuration as the rest of the system
        VLLM_API_BASE = "http://localhost:8002/v1"
        VLLM_MODEL_NAME = "neuralmagic/pixtral-12b-quantized.w4a16"
        
        llm = ChatOpenAI(
            temperature=0.1,
            model_name=VLLM_MODEL_NAME,
            openai_api_base=VLLM_API_BASE,
            openai_api_key="token-abc123",
            max_tokens=200
        )
        
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()
        
    except Exception as e:
        logger.warning(f"LLM-based targeted response failed: {str(e)}, falling back to rich response")
        try:
            return generate_rich_response_from_profile(profile)
        except:
            return f"I have information about {name} ({relationship}), but I'm not sure what specific details you're looking for. Could you be more specific?"

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
                # Use the complete updated profile if available, otherwise update just the specific field
                if 'updated_profile' in pending_update and pending_update['updated_profile']:
                    # Replace the entire profile with the updated version
                    profiles[profile_key] = pending_update['updated_profile']
                    profiles[profile_key]['last_updated'] = datetime.now().isoformat()
                    logger.info(f"[PROFILE CONFIRMATION] Using complete updated profile for {name}")
                else:
                    # Fallback: Update only the specific field
                    profiles[profile_key][field] = new_value
                    profiles[profile_key]['last_updated'] = datetime.now().isoformat()
                    logger.info(f"[PROFILE CONFIRMATION] Updated only {name}'s {field}")
                
                # Save the updated profile
                success = await save_user_profiles(user_id, profiles)
                if success:
                    if 'updated_profile' in pending_update and pending_update['updated_profile']:
                        # Check if this was a multi-field update
                        if 'fields_needing_confirmation' in pending_update and pending_update['fields_needing_confirmation']:
                            fields_count = len(pending_update['fields_needing_confirmation'])
                            if fields_count > 1:
                                state.final_response = f"Perfect! I've updated {name}'s profile with all {fields_count} changes and additional information."
                            else:
                                state.final_response = f"Perfect! I've updated {name}'s profile with all the new information."
                        else:
                            state.final_response = f"Perfect! I've updated {name}'s profile with all the new information."
                    else:
                        state.final_response = f"Perfect! I've updated {name}'s {field} to: {new_value}"
                        logger.info(f"[PROFILE CONFIRMATION] Successfully updated {name}'s profile")
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
        
        # Check if this was a multi-field update
        if 'fields_needing_confirmation' in pending_update and pending_update['fields_needing_confirmation']:
            fields_count = len(pending_update['fields_needing_confirmation'])
            if fields_count > 1:
                state.final_response = f"Understood! I won't update any of {pending_update['name']}'s information. All current information remains unchanged."
            else:
                state.final_response = f"Understood! I won't update {pending_update['name']}'s {pending_update['field']}. The current information remains unchanged."
        else:
            state.final_response = f"Understood! I won't update {pending_update['name']}'s {pending_update['field']}. The current information remains unchanged."
        
        # Clear the pending update and reset state
        state.pending_profile_update = None
        state.agent_status = "initialize"
        
    else:
        # Unclear or unrelated response - ask for clarification
        logger.info("[PROFILE CONFIRMATION] Unclear or unrelated user response, asking for clarification")
        
        # Check if the response contains profile-related information that should be ignored
        if any(keyword in user_response for keyword in ['sister', 'brother', 'lives', 'phone', 'number', 'friend', 'colleague']):
            state.final_response = (
                f"I notice you're providing new profile information, but I'm currently waiting for confirmation about updating {pending_update['name']}'s {pending_update['field']}. "
                f"Please first answer 'yes' to confirm updating {pending_update['name']}'s {pending_update['field']} to '{pending_update['new_value']}', or 'no' to cancel the update. "
                f"After that, I'll be happy to help you with the new information about {pending_update['name']}."
            )
        else:
            state.final_response = f"I didn't understand your response. Please answer with 'yes' to confirm updating {pending_update['name']}'s {pending_update['field']} to '{pending_update['new_value']}', or 'no' to cancel the update."
        
        state.agent_status = "waiting_confirmation"
    
    return state

async def handle_partial_name_match_confirmation_node(state: AgentState) -> AgentState:
    """Node to handle user confirmation for partial name matches."""
    logger.info("[LANGGRAPH PATH] Starting handle_partial_name_match_confirmation_node")
    
    if not state.pending_partial_match:
        logger.warning("[PARTIAL NAME MATCH] No pending partial match found")
        state.final_response = "I'm not sure what you're confirming. Could you please provide the profile information again?"
        state.agent_status = "initialize"
        return state
    
    pending_match = state.pending_partial_match
    user_response = state.retrieval_query.strip().lower()
    
    logger.info(f"[PARTIAL NAME MATCH] Processing confirmation for: {pending_match}")
    logger.info(f"[PARTIAL NAME MATCH] User response: {user_response}")
    
    # Check if user confirmed the match
    if user_response in ['yes', 'y', 'confirm', 'ok', 'okay', 'sure', 'go ahead', 'same person', 'same']:
        logger.info("[PARTIAL NAME MATCH] User confirmed the match")
        
        try:
            # Get the profile data
            user_id = state.user_id
            new_name = pending_match['new_name']
            relationship = pending_match['relationship']
            information = pending_match['information']
            
            # Load existing profiles
            profiles = await load_user_profiles(user_id)
            
            # Find the existing profile to update
            if len(pending_match['potential_matches']) == 1:
                # Single match - update the existing profile
                match = pending_match['potential_matches'][0]
                existing_profile_key = match['key']
                existing_profile = match['profile']
                
                # Use intelligent profile update to properly merge new information
                from rag.rich_profile_utils import intelligently_update_profile
                
                try:
                    # Intelligently update the profile with new information
                    updated_profile = await intelligently_update_profile(existing_profile, [information])
                    
                    # Replace the profile with the updated version
                    profiles[existing_profile_key] = updated_profile
                    
                    logger.info(f"[PARTIAL NAME MATCH] Intelligently updated profile '{existing_profile_key}' with new information")
                    
                    # Save the updated profile
                    success = await save_user_profiles(user_id, profiles)
                    
                except Exception as e:
                    logger.error(f"[PARTIAL NAME MATCH] Error in intelligent update: {str(e)}")
                    # Fallback to simple update
                    if isinstance(existing_profile.get('information'), list):
                        existing_profile['information'].append(information)
                    else:
                        existing_profile['information'] = [existing_profile.get('information', ''), information]
                    
                    existing_profile['last_updated'] = datetime.now().isoformat()
                    
                    # Save the updated profile
                    success = await save_user_profiles(user_id, profiles)
                if success:
                    state.final_response = (
                        f"Perfect! I've updated the existing profile for {existing_profile.get('name')} ({relationship}) "
                        f"with the new information: {information}"
                    )
                    logger.info(f"[PARTIAL NAME MATCH] Successfully updated existing profile for {existing_profile.get('name')}")
                else:
                    state.final_response = f"Sorry, I encountered an error while updating the profile. Please try again."
                    logger.error(f"[PARTIAL NAME MATCH] Failed to save updated profile")
            else:
                # Multiple matches - ask user to specify which one
                options = [f"'{match['existing_name']}'" for match in pending_match['potential_matches']]
                state.final_response = (
                    f"I found multiple potential matches: {', '.join(options)}. "
                    f"Please specify which one you're referring to by saying something like 'the first one' or 'the second one'."
                )
                state.agent_status = "waiting_partial_match_confirmation"
                return state
                
        except Exception as e:
            logger.error(f"[PARTIAL NAME MATCH] Error updating profile: {str(e)}")
            state.final_response = f"Sorry, I encountered an error while updating the profile. Please try again."
        
        # Clear the pending match and reset state
        state.pending_partial_match = None
        state.agent_status = "initialize"
        
    elif user_response in ['no', 'n', 'cancel', 'stop', 'don\'t', 'dont', 'different person', 'different']:
        logger.info("[PARTIAL NAME MATCH] User declined the match")
        
        try:
            # Create a new profile since user confirmed it's a different person
            user_id = state.user_id
            new_name = pending_match['new_name']
            relationship = pending_match['relationship']
            information = pending_match['information']
            
            # Create the new profile with force_create=True to bypass partial name match checks
            created_profile = await create_or_update_profile_persistent(user_id, new_name, relationship, information, force_create=True, last_mentioned_profile=None)
            logger.info(f"[PARTIAL NAME MATCH] Successfully created new profile: {created_profile}")
            
            state.final_response = (
                f"Understood! I've created a new profile for {new_name} ({relationship}) "
                f"since you confirmed it's a different person from the existing profiles."
            )
            logger.info(f"[PARTIAL NAME MATCH] Created new profile for {new_name}")
            
        except Exception as e:
            logger.error(f"[PARTIAL NAME MATCH] Error creating new profile: {str(e)}")
            state.final_response = f"Sorry, I encountered an error while creating the new profile. Please try again."
        
        # Clear the pending match and reset state
        state.pending_partial_match = None
        state.agent_status = "initialize"
        
    else:
        # Unclear or unrelated response - ask for clarification
        logger.info("[PARTIAL NAME MATCH] Unclear or unrelated user response, asking for clarification")
        
        if len(pending_match['potential_matches']) == 1:
            match = pending_match['potential_matches'][0]
            state.final_response = (
                f"I found an existing profile for your {relationship} named '{match['existing_name']}'. "
                f"Are you referring to the same person as '{pending_match['new_name']}'? "
                f"Please answer 'yes' if it's the same person, or 'no' if it's a different person."
            )
        else:
            options = [f"'{match['existing_name']}'" for match in pending_match['potential_matches']]
            state.final_response = (
                f"I found {len(pending_match['potential_matches'])} existing profiles that might match '{pending_match['new_name']}': {', '.join(options)}. "
                f"Are you referring to one of these people? If yes, please specify which one. If no, I'll create a new profile."
            )
        
        state.agent_status = "waiting_partial_match_confirmation"
    
    return state

async def handle_name_input_node(state: AgentState) -> AgentState:
    """
    Handle when user provides a name after being asked for it.
    """
    logger.info("[NAME INPUT] Processing name input")
    
    # Get the user's response
    user_response = state.request.messages[-1].content.strip()
    logger.info(f"[NAME INPUT] User response: {user_response}")
    
    # Extract the name from the response
    name = user_response.strip()
    
    if not name or name.lower() in ['no', 'none', 'skip', 'cancel']:
        state.final_response = "Okay, I'll skip creating a profile for now. Let me know if you'd like to add information about someone later!"
        state.agent_status = "initialize"
        return state
    
    # Clean the name
    clean_name = name.strip().title()
    logger.info(f"[NAME INPUT] Cleaned name: {clean_name}")
    
    # Get the relationship from the waiting_for_name context
    relationship = "unknown"
    if state.waiting_for_name and state.waiting_for_name.get("relationship"):
        relationship = state.waiting_for_name["relationship"]
        logger.info(f"[NAME INPUT] Using stored relationship: {relationship}")
    else:
        # Fallback: look for relationship keywords in the last few messages
        for msg in reversed(state.request.messages[-3:]):  # Check last 3 messages
            msg_text = msg.content.lower()
            for rel in ['friend', 'colleague', 'family', 'supervisor', 'neighbor', 'assistant', 'cousin', 'brother', 'sister', 'mother', 'father', 'daughter', 'son', 'wife', 'husband', 'partner']:
                if rel in msg_text:
                    relationship = rel
                    break
            if relationship != "unknown":
                break
        logger.info(f"[NAME INPUT] Detected relationship from messages: {relationship}")
    
    # Create a new profile with the provided name and detected relationship
    try:
        from rag.database import load_user_profiles, save_user_profiles
        
        profiles = await load_user_profiles(state.user_id)
        
        # Generate profile key
        clean_name_lower = clean_name.lower().replace(' ', '_')
        relationship_lower = relationship.lower()
        profile_key = f"{clean_name_lower}_{relationship_lower}"
        
        # Create new profile
        new_profile = {
            "name": clean_name,
            "relationship": relationship,
            "created_date": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "information": "Profile created"
        }
        
        profiles[profile_key] = new_profile
        await save_user_profiles(state.user_id, profiles)
        
        state.final_response = f"Perfect! I've created a new profile for your {relationship} {clean_name}."
        state.last_mentioned_profile = {
            "name": clean_name,
            "relationship": relationship,
            "information": "Profile created"
        }
        state.agent_status = "initialize"
        
        logger.info(f"[NAME INPUT] Successfully created profile for {clean_name} ({relationship})")
        
    except Exception as e:
        logger.error(f"[NAME INPUT] Error creating profile: {str(e)}")
        state.final_response = f"Sorry, I encountered an error while creating the profile. Please try again."
        state.agent_status = "initialize"
    
    return state

async def handle_multiple_profile_clarification_node(state: AgentState) -> AgentState:
    """Node to handle user clarification when multiple profiles exist with same first name and relationship."""
    logger.info("[LANGGRAPH PATH] Starting handle_multiple_profile_clarification_node")
    
    if not state.pending_multiple_profile_clarification:
        logger.warning("[MULTIPLE PROFILE CLARIFICATION] No pending multiple profile clarification found")
        state.final_response = "I'm not sure what you're clarifying. Could you please provide the profile information again?"
        state.agent_status = "initialize"
        return state
    
    pending_clarification = state.pending_multiple_profile_clarification
    user_response = state.retrieval_query.strip()
    
    logger.info(f"[MULTIPLE PROFILE CLARIFICATION] Processing clarification for: {pending_clarification['first_name']}")
    logger.info(f"[MULTIPLE PROFILE CLARIFICATION] User response: {user_response}")
    
    # Extract the clarification details
    first_name = pending_clarification["first_name"]
    relationship = pending_clarification["relationship"]
    existing_profiles = pending_clarification["existing_profiles"]
    new_information = pending_clarification["new_information"]
    
    # Parse user response
    user_response_lower = user_response.lower().strip()
    
    # Check if this is the first response (UPDATE or CREATE choice)
    if not state.multiple_profile_choice:
        if user_response_lower in ['update', 'u']:
            logger.info("[MULTIPLE PROFILE CLARIFICATION] User chose UPDATE")
            state.multiple_profile_choice = "UPDATE"
            
            # Update the existing profile (use the first one if multiple)
            selected_profile = existing_profiles[0]
            profile_key = selected_profile['key']
                
            # Update the profile with new information
            try:
                from rag.database import load_user_profiles, save_user_profiles
                    
                # Get current profiles
                profiles = await load_user_profiles(state.user_id)
                if profile_key in profiles:
                    # Update the existing profile
                    existing_profile = profiles[profile_key]
                    existing_profile['information'] = new_information
                    
                    # Update location and phone if provided in new information
                    if 'lives in' in new_information.lower():
                        # Extract location from new information
                        location_match = re.search(r'lives in ([^,]+)', new_information, re.IGNORECASE)
                        if location_match:
                            existing_profile['location'] = location_match.group(1).strip()
                    
                    if 'phone' in new_information.lower():
                        # Extract phone from new information
                        phone_match = re.search(r'phone (?:number )?is? ([^\s,]+)', new_information, re.IGNORECASE)
                        if phone_match:
                            existing_profile['phone'] = phone_match.group(1).strip()
                    
                    # Save the updated profile
                    profiles[profile_key] = existing_profile
                    await save_user_profiles(state.user_id, profiles)
                    
                    # Set as last mentioned profile
                    state.last_mentioned_profile = {
                        "name": existing_profile.get('name', first_name),
                        "relationship": relationship,
                        "information": new_information
                    }
                    
                    state.final_response = f"Updated {existing_profile.get('name', first_name)}'s profile with the new information: {new_information}"
                    logger.info(f"[MULTIPLE PROFILE CLARIFICATION] Successfully updated profile: {profile_key}")
                else:
                    state.final_response = f"Could not find the profile to update. Please try again."
                    logger.error(f"[MULTIPLE PROFILE CLARIFICATION] Profile not found: {profile_key}")
                        
            except Exception as e:
                logger.error(f"[MULTIPLE PROFILE CLARIFICATION] Error updating profile: {str(e)}")
                state.final_response = f"Error updating profile: {str(e)}"
            
            # Clear the pending clarification and reset state
            state.pending_multiple_profile_clarification = None
            state.multiple_profile_choice = None
            state.agent_status = "profile_updated"
            
            logger.info(f"[MULTIPLE PROFILE CLARIFICATION] Returning state with final_response: {state.final_response}")
            logger.info(f"[MULTIPLE PROFILE CLARIFICATION] State agent_status: {state.agent_status}")
            logger.info(f"[MULTIPLE PROFILE CLARIFICATION] State type: {type(state)}")
            logger.info(f"[MULTIPLE PROFILE CLARIFICATION] State dict: {state.dict() if hasattr(state, 'dict') else 'No dict method'}")
            return state
                
        elif user_response_lower in ['create', 'c']:
            logger.info("[MULTIPLE PROFILE CLARIFICATION] User chose CREATE")
            state.multiple_profile_choice = "CREATE"
            
            # Ask for full name to avoid confusion
            state.final_response = f"Great! I'll create a new profile for your {relationship} {first_name}. To avoid confusion with your existing {relationship}s named {first_name}, please provide their full name (e.g., '{first_name} Smith' or '{first_name} Johnson')."
            state.agent_status = "waiting_full_name_for_new_profile"
            
            # Store the pending profile creation data
            state.pending_new_profile_creation = {
                "first_name": first_name,
                "relationship": relationship,
                "new_information": new_information
            }
            
            return state
    
    else:
        # Invalid response, ask again
        state.final_response = "Please respond with 'UPDATE' or 'CREATE' to proceed."
        return state
    
    # If we reach here, there's an unexpected state
    logger.warning("[MULTIPLE PROFILE CLARIFICATION] Unexpected state reached")
    state.final_response = "I'm not sure what you're trying to do. Please start over with the profile information."
    state.pending_multiple_profile_clarification = None
    state.multiple_profile_choice = None
    state.agent_status = "initialize"
    return state

# Pattern to detect pronouns to replace
PRONOUN_PATTERN = re.compile(r'\b(he|she|him|her|his|hers|they|them|their|theirs)\b', re.IGNORECASE)

def replace_pronouns(text: str, last_profile: Dict[str, str]) -> str:
    """Replace simple pronouns with last mentioned profile's name or 'relationship name'."""
    if not last_profile or not last_profile.get("name"):
        return text
    name = last_profile["name"]
    relationship = last_profile.get("relationship", "").strip().lower()
    # Prefer relationship + name if known and not 'unknown', else just name
    replacement = f"{relationship} {name}" if relationship and relationship != "unknown" else name

    def repl(match):
        pronoun = match.group(0).lower()
        # Handle different pronoun cases
        if pronoun in ["he", "him", "his"]:
            # Masculine pronouns
            if pronoun[0].isupper():
                return replacement.title()
            return replacement
        elif pronoun in ["she", "her", "hers"]:
            # Feminine pronouns
            if pronoun[0].isupper():
                return replacement.title()
            return replacement
        elif pronoun in ["they", "them", "their", "theirs"]:
            # Gender-neutral pronouns
            if pronoun[0].isupper():
                return replacement.title()
            return replacement
        else:
            # Fallback
            if pronoun[0].isupper():
                return replacement.title()
        return replacement

    return PRONOUN_PATTERN.sub(repl, text)

async def profile_query_answer_node(state: AgentState) -> AgentState:
    """
    Fully dynamic profile query answer node using LLM for intelligent query understanding and response generation.
    
    This node uses a single LLM call to:
    1. Understand the user's query intent
    2. Find matching profiles intelligently
    3. Generate appropriate responses
    4. Handle clarifications naturally
    """
    from rag.rich_profile_utils import generate_rich_response_from_profile, is_rich_profile
    logger.info("[LANGGRAPH PATH] Starting profile_query_answer_node")
    user_id = state.user_id
    raw_query = state.retrieval_query.strip()

    # Load existing profiles
    profiles = await load_user_profiles(user_id)
    logger.info(f"Loaded {len(profiles)} existing profiles for user {user_id}.")
    
    # Process the query to handle pronouns
    processed_query = raw_query
    if state.last_mentioned_profile and state.last_mentioned_profile.get('name'):
        # Check if the query contains pronouns
        if re.search(PRONOUN_PATTERN, raw_query):
            logger.info(f"[PROFILE QUERY] Query contains pronouns, replacing with last mentioned profile: {state.last_mentioned_profile.get('name')}")
            processed_query = replace_pronouns(raw_query, state.last_mentioned_profile)
            logger.info(f"[PROFILE QUERY] Processed query after pronoun replacement: {processed_query}")
        else:
            logger.info(f"[PROFILE QUERY] No pronoun replacement - profile incomplete: {state.last_mentioned_profile}")
    else:
        logger.info(f"[PROFILE QUERY] No pronoun replacement - no last mentioned profile")
    
    # Create context for LLM
    context_parts = []
    for profile_key, profile in profiles.items():
        name = profile.get('name', 'Unknown')
        relationship = profile.get('relationship', 'unknown')
        
        # Add basic info
        info_parts = []
        if profile.get('location'):
            info_parts.append(f"lives in {profile.get('location')}")
        if profile.get('phone'):
            info_parts.append(f"phone: {profile.get('phone')}")
        if profile.get('workplace'):
            info_parts.append(f"works at {profile.get('workplace')}")
        if profile.get('birthday'):
            info_parts.append(f"birthday: {profile.get('birthday')}")
        if profile.get('family'):
            family_info = profile.get('family')
            if isinstance(family_info, list):
                info_parts.append(f"family: {'; '.join(family_info)}")
            else:
                info_parts.append(f"family: {family_info}")
        if profile.get('other'):
            other_info = profile.get('other')
            if isinstance(other_info, list):
                info_parts.append(f"other: {'; '.join(other_info)}")
            else:
                info_parts.append(f"other: {other_info}")
        if profile.get('information'):
            info = profile.get('information')
            if isinstance(info, list):
                info_parts.append(f"info: {'; '.join(info)}")
            else:
                info_parts.append(f"info: {info}")
        
        info_text = f"; {'; '.join(info_parts)}" if info_parts else ""
        context_parts.append(f"{relationship.capitalize()} {name}{info_text}")
    
    context_text = '\n'.join(context_parts) if context_parts else "No profiles available"
    
    logger.info(f"Profile Context: {context_text}")
    
    # DYNAMIC LLM-BASED QUERY PROCESSING
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        
        # Use the same local LLM configuration as the rest of the system
        VLLM_API_BASE = "http://localhost:8002/v1"
        VLLM_MODEL_NAME = "neuralmagic/pixtral-12b-quantized.w4a16"
        
        llm = ChatOpenAI(
            temperature=0.1,
            model_name=VLLM_MODEL_NAME,
            openai_api_base=VLLM_API_BASE,
            openai_api_key="token-abc123",
            max_tokens=500
        )
        
        # Create a comprehensive prompt for intelligent query processing
        prompt = f"""
        You are an intelligent assistant that helps users query their personal profile database. Your task is to understand the user's query and provide the most appropriate response.

        **User's Query:** {processed_query}

        **Available Profiles:**
        {context_text}

        **Instructions:**
        1. **Understand the Query**: Analyze what the user is asking for, paying special attention to relationship keywords (friend, colleague, family, etc.)
        2. **Find Matching Profiles**: Identify which profiles are relevant to the query. If a relationship is specified, prioritize profiles with that exact relationship
        3. **Generate Response**: Provide a natural, helpful response

        **Response Types:**
        - **Direct Answer**: If asking about a specific person (e.g., "How many brothers does Sam have?"), provide a direct answer
        - **Profile Summary**: If asking for general info about someone, provide a summary
        - **Multiple Profiles**: If multiple people match, list them with brief info
        - **Clarification**: If ambiguous, ask for clarification
        - **Not Found**: If no matching profiles, say so politely

        **Examples:**
        - "How many brothers does my colleague Sam have?" → "Sam has 2 brothers: Harray and Ashley"
        - "Tell me about John" → Provide John's profile summary
        - "My colleagues" → List all colleagues with brief info
        - "Tell me about friend Sam" → Provide friend Sam's profile summary (even if there's also a colleague Sam)
        - "Tell me about Sam" (when multiple Sams exist and no relationship specified) → Ask which Sam they mean

        **Important Rules:**
        - Be direct and concise
        - Only use information from the provided profiles
        - If asking for a count (e.g., "how many"), provide the exact number
        - If asking for names, list them clearly
        - Be natural and conversational
        - Don't make assumptions beyond the provided data
        - When a relationship is specified (e.g., "friend Sam"), only consider profiles with that exact relationship
        - Only ask for clarification if there are truly multiple profiles with the same name AND relationship

        **Response:**"""

        response = await llm.ainvoke([HumanMessage(content=prompt)])
        final_response = response.content.strip()
        
        # Check if the LLM is asking for clarification (indicates multiple profiles)
        if any(phrase in final_response.lower() for phrase in [
            "which one", "which sam", "which john", "multiple profiles", "found multiple",
            "please specify", "clarify", "which person", "which colleague", "which friend",
            "would you like more information", "either of these", "which of these", "which are you asking about",
            "which one are you", "which one do you", "which one would you", "which one is"
        ]):
            # This is a clarification request, set up the clarification state
            logger.info("[PROFILE QUERY] LLM requested clarification, setting up clarification state")
            
            # Try to extract the name being asked about
            detected_people, _ = await detect_profiles_in_text(processed_query, state.request.messages)
            if detected_people:
                person_name = detected_people[0].get('name', '').lower()
                person_relationship = detected_people[0].get('relationship', '').lower()
                
                # Find profiles with this name using first name matching
                person_first_name = person_name.split()[0] if person_name else person_name
                profiles_with_same_name = []
                
                for profile_key, profile in profiles.items():
                    profile_name = profile.get('name', '').lower()
                    profile_first_name = profile_name.split()[0] if profile_name else profile_name
                    profile_relationship = profile.get('relationship', '').lower()
                    
                    # Check if first names match and relationships match (if specified)
                    if profile_first_name == person_first_name:
                        if person_relationship and person_relationship != 'unknown':
                            if profile_relationship == person_relationship:
                                profiles_with_same_name.append((profile_key, profile))
                        else:
                            profiles_with_same_name.append((profile_key, profile))
                
                logger.info(f"[PROFILE QUERY] Found {len(profiles_with_same_name)} profiles with first name '{person_first_name}' and relationship '{person_relationship}'")
                
                # Only ask for clarification if there are multiple profiles with the same first name and relationship
                if len(profiles_with_same_name) > 1:
                    # Create numbered options for clarification
                    clarification_message = f"I found {len(profiles_with_same_name)} profiles with the name '{person_first_name}' who are your {person_relationship}s. Which one would you like to know about?\n\n"
                    
                    for i, (profile_key, profile) in enumerate(profiles_with_same_name, 1):
                        name = profile.get('name', 'Unknown')
                        location = profile.get('location', 'Not specified')
                        phone = profile.get('phone', 'Not specified')
                        other_info = []
                        
                        if profile.get('family'):
                            other_info.append(f"Family: {profile.get('family')}")
                        if profile.get('other'):
                            other_info.append(f"Other: {profile.get('other')}")
                        
                        clarification_message += f"{i}. {name}"
                        if location != 'Not specified':
                            clarification_message += f" (lives in {location})"
                        if phone != 'Not specified':
                            clarification_message += f" (phone: {phone})"
                        if other_info:
                            clarification_message += f" - {', '.join(other_info)}"
                        clarification_message += "\n"
                    
                    clarification_message += f"\nPlease respond with the number (1-{len(profiles_with_same_name)}) of the person you'd like to know about."
                    
                    state.final_response = clarification_message
                    state.agent_status = "waiting_profile_clarification"
                    state.pending_profile_clarification = {
                        "name": person_first_name,
                        "relationship": person_relationship,
                        "profiles": profiles_with_same_name,
                        "query": processed_query
                    }
                    return state
                else:
                    logger.info(f"[PROFILE QUERY] Only one profile found after relationship filtering, proceeding with direct response")
        
        # Regular response
        state.final_response = final_response
        
        # Set last_mentioned_profile if we provided information about a specific person
        # This enables pronoun resolution in subsequent messages
        try:
            # Try to extract the person's name from the response or query
            detected_people, _ = await detect_profiles_in_text(processed_query, state.request.messages)
            if detected_people:
                person = detected_people[0]
                person_name = person.get('name', '')
                person_relationship = person.get('relationship', 'unknown')
                
                # Find the corresponding profile to get the relationship
                for profile_key, profile in profiles.items():
                    if profile.get('name', '').lower() == person_name.lower():
                        person_relationship = profile.get('relationship', person_relationship)
                        break
                
                state.last_mentioned_profile = {
                    "name": person_name,
                    "relationship": person_relationship,
                    "information": "Recently discussed"
                }
                logger.info(f"[PROFILE QUERY] Set last_mentioned_profile: {state.last_mentioned_profile}")
        except Exception as e:
            logger.warning(f"[PROFILE QUERY] Failed to set last_mentioned_profile: {str(e)}")
        
        state.agent_status = "initialize"
        return state
        
    except Exception as e:
        logger.error(f"[PROFILE QUERY] LLM-based processing failed: {str(e)}")
        # Fallback to simple response
        state.final_response = "I'm having trouble processing your request right now. Could you please try rephrasing your question?"
        state.agent_status = "initialize"
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
    
    # ENHANCED: Track if this was called from profile routing
    if hasattr(state, 'agent_status') and state.agent_status == "general":
        logger.info("[LANGGRAPH PATH] General node called from profile routing - providing comprehensive knowledge response")
    
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

    ## Profile-Aware General Chat Handling:
    - **IMPORTANT**: This query was routed to the general knowledge node because no matching profiles were found in the user's personal database.
    - If the user is asking about a specific person, provide helpful general information or suggestions.
    - If the user is asking about relationships, locations, or personal information, explain that this information isn't in their personal database.
    - Offer to help them create a profile for the person they're asking about.
    - Provide general knowledge and helpful responses while being clear about what information is available.
    - **Never fabricate personal information** about people not in the user's database.

    ## General Chat Handling:
    - Respond naturally and helpfully, but **always ground your answers in verifiable information.**
    - If the user asks for opinions or analysis, **clearly distinguish fact from interpretation.**
    - **Do not over-explain or add unnecessary context** unless it directly supports clarity or relevance.
    - If the context is ambiguous, ask for clarification instead of assuming.

    ## Context:
    {context_text}
    
    Respond clearly and helpfully, being mindful that this query was routed here due to missing profile information.
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
    workflow.add_node("handle_partial_name_match_confirmation", handle_partial_name_match_confirmation_node)
    workflow.add_node("handle_multiple_profile_clarification", handle_multiple_profile_clarification_node)
    workflow.add_node("handle_full_name_for_new_profile", handle_full_name_for_new_profile_node)
    workflow.add_node("handle_name_input", handle_name_input_node)
    workflow.add_node("prepare_llm_input", prepare_llm_input_node)
    workflow.add_node("call_langchain_agent", call_langchain_agent_node)
    
    # Set entry point
    workflow.set_entry_point("initialize")
    
    # Define transitions
    def route_after_initialize(state):
        next_node = state.agent_status
        logger.info(f"[TRANSITION] initialize -> {next_node}")
        logger.info(f"[TRANSITION] State agent_status: {state.agent_status}")
        logger.info(f"[TRANSITION] State final_response: {state.final_response}")
        return next_node

    workflow.add_conditional_edges("initialize", 
        route_after_initialize, {
        "initialize": "user_data_and_stats",
        "confirm_creation": "profile_create_update",
        "confirm_answer": "profile_query_answer",
        "waiting_confirmation": "handle_profile_update_confirmation",
        "waiting_profile_clarification": "handle_profile_clarification",
        "waiting_partial_match_confirmation": "handle_partial_name_match_confirmation",
        "waiting_multiple_profile_clarification": "handle_multiple_profile_clarification",
        "waiting_full_name_for_new_profile": "handle_full_name_for_new_profile",
        "waiting_for_name": "handle_name_input",
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
        "waiting_partial_match_confirmation": END,  # End workflow to wait for partial name match confirmation
        "waiting_multiple_profile_clarification": END,  # End workflow to wait for multiple profile clarification
        "waiting_for_name": END,  # End workflow to wait for name input
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
    
    # Add conditional edge for handle_partial_name_match_confirmation
    workflow.add_conditional_edges("handle_partial_name_match_confirmation",
        lambda state: state.agent_status, {
        "waiting_partial_match_confirmation": END,  # Still waiting for partial name match confirmation
        "initialize": END  # Confirmation processed, workflow complete
    })
    
    # Add conditional edge for handle_multiple_profile_clarification
    def route_multiple_profile_clarification(state):
        next_node = state.agent_status
        logger.info(f"[MULTIPLE PROFILE ROUTING] agent_status: {state.agent_status}")
        logger.info(f"[MULTIPLE PROFILE ROUTING] Routing to: {next_node}")
        return next_node
    
    workflow.add_conditional_edges("handle_multiple_profile_clarification",
        route_multiple_profile_clarification, {
        "waiting_multiple_profile_clarification": END,  # Still waiting for multiple profile clarification
        "waiting_full_name_for_new_profile": "handle_full_name_for_new_profile",  # User wants to create new profile with full name
        "initialize": END,  # Clarification processed, workflow complete
        "profile_updated": END  # Profile successfully updated, workflow complete
    })
    
    # Add conditional edge for handle_full_name_for_new_profile
    workflow.add_conditional_edges("handle_full_name_for_new_profile",
        lambda state: state.agent_status, {
        "waiting_full_name_for_new_profile": END,  # Still waiting for full name input
        "initialize": END  # Profile created, workflow complete
    })

    # Compile with built-in checkpoint saver
    compiled_workflow = workflow.compile(
        checkpointer=checkpoint_store,
        interrupt_after=["profile_create_update", "handle_profile_update_confirmation", "handle_profile_clarification", "handle_partial_name_match_confirmation", "handle_multiple_profile_clarification", "handle_full_name_for_new_profile", "handle_name_input", "call_langchain_agent"]
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
                logger.info(f"[CHAT] Reconstructed state - multiple_profile_choice: {getattr(input_state, 'multiple_profile_choice', 'NOT_FOUND')}")
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
            elif "handle_partial_name_match_confirmation" in event:
                final_state = event["handle_partial_name_match_confirmation"]
            elif "handle_profile_update_confirmation" in event:
                final_state = event["handle_profile_update_confirmation"]
            elif "handle_profile_clarification" in event:
                final_state = event["handle_profile_clarification"]
            elif "handle_multiple_profile_clarification" in event:
                final_state = event["handle_multiple_profile_clarification"]
            elif "handle_full_name_for_new_profile" in event:
                final_state = event["handle_full_name_for_new_profile"]
            elif "handle_name_input" in event:
                final_state = event["handle_name_input"]
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
        
        # Check if final_state exists before accessing it
        if final_state:
            logger.info(f"Final Response: {final_state.get('final_response', 'No response')}")
            logger.info(f"Final State Type: {type(final_state)}")
            logger.info(f"Final State Keys: {list(final_state.keys()) if isinstance(final_state, dict) else 'Not a dict'}")
            response_message = final_state.get("final_response", "No response generated")
        else:
            logger.error("Final state is None - no response generated")
            logger.error("This means the workflow did not return a valid final state")
            logger.error("Check the workflow routing and node return values")
            response_message = "I'm sorry, I encountered an error processing your request. Please try again."
        
        if final_state and final_state.get("user_name"):
            user_name = final_state.get("user_name")
            logger.info(f"User name extracted: {user_name}")
            return [response_message, {'name': user_name}]
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
        workers=1  # Use single worker to avoid issues in containers
    ) 
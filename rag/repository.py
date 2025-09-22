import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import logging

from rag.models import Repository, RepositoryResponse
from rag.database import get_documents_by_repository_id

# Configure logger
logger = logging.getLogger(__name__)

# Storage paths
REPOSITORY_STORE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "repositories")

# Create directories if they don't exist
os.makedirs(REPOSITORY_STORE_PATH, exist_ok=True)

logger.info(f"Repository store path: {REPOSITORY_STORE_PATH}")
logger.info(f"Repository store path exists: {os.path.exists(REPOSITORY_STORE_PATH)}")

# Helper functions to serialize/deserialize datetime objects
def serialize_datetime(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError("Type not serializable")

def repository_to_dict(repository: Repository) -> Dict[str, Any]:
    """Convert Repository model to dictionary for storage."""
    try:
        # Try different Pydantic methods for compatibility
        if hasattr(repository, 'model_dump'):
            repo_dict = repository.model_dump()
            logger.info(f"Using model_dump(): {repo_dict}")
        elif hasattr(repository, 'dict'):
            repo_dict = repository.dict()
            logger.info(f"Using dict(): {repo_dict}")
        else:
            repo_dict = repository.__dict__
            logger.info(f"Using __dict__: {repo_dict}")
        return repo_dict
    except Exception as e:
        logger.error(f"Error converting repository to dict: {str(e)}")
        import traceback
        traceback.print_exc()
        # Fallback to __dict__
        return repository.__dict__

def dict_to_repository(repo_dict: Dict[str, Any]) -> Repository:
    """Convert dictionary to Repository model."""
    # Convert string timestamps to datetime objects
    if isinstance(repo_dict.get("created_at"), str):
        repo_dict["created_at"] = datetime.fromisoformat(repo_dict["created_at"])
    
    if isinstance(repo_dict.get("updated_at"), str):
        repo_dict["updated_at"] = datetime.fromisoformat(repo_dict["updated_at"])
    
    return Repository(**repo_dict)

async def create_repository(name: str, user_id: str, description: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, is_global: bool = False, repository_id: Optional[str] = None) -> Repository:
    """
    Create a new repository.
    
    Args:
        name: Repository name
        user_id: ID of the user creating the repository
        description: Optional description
        metadata: Optional metadata
        is_global: Optional flag to mark if the repository is global
        repository_id: Optional, specific ID to use for the repository. If None, a new UUID will be generated.
        
    Returns:
        Created repository
    """
    logger.info(f"Creating repository: name={name}, user_id={user_id}, repository_id={repository_id}")
    
    # Use provided repository_id or generate a new one
    final_repository_id = repository_id if repository_id else str(uuid.uuid4())
    logger.info(f"Final repository ID: {final_repository_id}")
    
    # Create repository
    repository = Repository(
        id=final_repository_id,
        name=name,
        description=description,
        user_id=user_id,
        metadata=metadata or {},
        is_global=is_global,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    logger.info(f"Repository object created: {repository}")
    
    # Save repository
    save_result = await save_repository(repository)
    logger.info(f"Save result: {save_result}")
    
    # Verify the file was created
    repository_path = os.path.join(REPOSITORY_STORE_PATH, f"{final_repository_id}.json")
    if os.path.exists(repository_path):
        logger.info(f"Repository file created successfully: {repository_path}")
    else:
        logger.error(f"ERROR: Repository file not found: {repository_path}")
    
    return repository

async def save_repository(repository: Repository) -> bool:
    """
    Save a repository to the repository store.
    
    Args:
        repository: The repository to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Saving repository: {repository.id}")
        logger.info(f"Repository store path: {REPOSITORY_STORE_PATH}")
        
        # Convert repository to dictionary
        repo_dict = repository_to_dict(repository)
        logger.info(f"Repository dict: {repo_dict}")
        
        # Save repository to file
        repository_path = os.path.join(REPOSITORY_STORE_PATH, f"{repository.id}.json")
        logger.info(f"Saving to path: {repository_path}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(repository_path), exist_ok=True)
        
        with open(repository_path, "w") as f:
            json.dump(repo_dict, f, default=serialize_datetime)
        
        logger.info(f"Repository saved successfully to: {repository_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving repository: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def get_repository_by_id(repository_id: str) -> Optional[Repository]:
    """
    Get a repository by ID.
    
    Args:
        repository_id: ID of the repository to retrieve
        
    Returns:
        Repository if found, None otherwise
    """
    try:
        repository_path = os.path.join(REPOSITORY_STORE_PATH, f"{repository_id}.json")
        
        if not os.path.exists(repository_path):
            return None
        
        with open(repository_path, "r") as f:
            repo_dict = json.load(f)
        
        return dict_to_repository(repo_dict)
    except Exception as e:
        logger.error(f"Error retrieving repository: {str(e)}")
        return None

async def get_repositories_by_user_id(user_id: str) -> List[Repository]:
    """
    Get all repositories for a user.
    
    Args:
        user_id: ID of the user
        
    Returns:
        List of repositories
    """
    repositories = []
    
    try:
        logger.info(f"Getting repositories for user: {user_id}")
        logger.info(f"Repository store path: {REPOSITORY_STORE_PATH}")
        logger.info(f"Repository store path exists: {os.path.exists(REPOSITORY_STORE_PATH)}")
        
        if not os.path.exists(REPOSITORY_STORE_PATH):
            logger.error(f"ERROR: Repository store path does not exist: {REPOSITORY_STORE_PATH}")
            return repositories
        
        files = os.listdir(REPOSITORY_STORE_PATH)
        logger.info(f"Files in repository store: {files}")
        
        for filename in files:
            if filename.endswith(".json"):
                repository_path = os.path.join(REPOSITORY_STORE_PATH, filename)
                logger.info(f"Reading repository file: {repository_path}")
                
                with open(repository_path, "r") as f:
                    repo_dict = json.load(f)
                
                logger.info(f"Repository data: {repo_dict}")
                logger.info(f"Repository user_id: {repo_dict.get('user_id')}, looking for: {user_id}")
                
                if repo_dict.get("user_id") == user_id:
                    logger.info(f"Found matching repository: {filename}")
                    repositories.append(dict_to_repository(repo_dict))
                else:
                    logger.info(f"Repository {filename} does not match user_id")
    except Exception as e:
        logger.error(f"Error retrieving repositories: {str(e)}")
        import traceback
        traceback.print_exc()
    
    logger.info(f"Returning {len(repositories)} repositories for user {user_id}")
    return repositories

async def update_repository(repository_id: str, updates: Dict[str, Any]) -> Optional[Repository]:
    """
    Update a repository.
    
    Args:
        repository_id: ID of the repository to update
        updates: Dictionary of fields to update
        
    Returns:
        Updated repository if successful, None otherwise
    """
    try:
        repository = await get_repository_by_id(repository_id)
        
        if not repository:
            return None
        
        # Update fields
        for key, value in updates.items():
            if hasattr(repository, key) and key not in ["id", "user_id", "created_at"]:
                setattr(repository, key, value)
        
        # Update timestamp
        repository.updated_at = datetime.now()
        
        # Save updated repository
        await save_repository(repository)
        
        return repository
    except Exception as e:
        logger.error(f"Error updating repository: {str(e)}")
        return None

async def delete_repository(repository_id: str) -> bool:
    """
    Delete a repository.
    
    Args:
        repository_id: ID of the repository to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if repository exists
        repository_path = os.path.join(REPOSITORY_STORE_PATH, f"{repository_id}.json")
        
        if not os.path.exists(repository_path):
            return False
        
        # Delete repository file
        os.remove(repository_path)
        
        return True
    except Exception as e:
        logger.error(f"Error deleting repository: {str(e)}")
        return False

async def get_repository_response(repository: Repository) -> RepositoryResponse:
    """
    Convert a Repository to a RepositoryResponse, including document count.
    
    Args:
        repository: Repository to convert
        
    Returns:
        RepositoryResponse
    """
    # Get documents for this repository
    documents = await get_documents_by_repository_id(repository.id)
    
    return RepositoryResponse(
        repository_id=repository.id,
        name=repository.name,
        description=repository.description,
        document_count=len(documents),
        created_at=repository.created_at,
        updated_at=repository.updated_at
    ) 
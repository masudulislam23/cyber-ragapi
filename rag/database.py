import os
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid

from rag.models import Document, ProcessingStatus, Chunk
from rag.vector_store import delete_document_chunks

# Import rich profile utilities
try:
    print("Attempting to import rich profile utilities...")
    from rag.rich_profile_utils import is_rich_profile, convert_simple_to_rich_profile
    print("Successfully imported rich profile utilities")
except ImportError as e:
    print(f"Failed to import rich profile utilities: {e}")
    # Fallback if rich profile utilities are not available
    def is_rich_profile(profile):
        return False
    
    async def convert_simple_to_rich_profile(profile):
        return profile

import logging
logger = logging.getLogger(__name__)

# Storage paths
DOCUMENT_STORE_PATH = "./data/documents"
CHUNK_STORE_PATH = "./data/chunks"
USER_QUERY_COUNT_PATH = "./data/user_query_counts.json"
PROFILE_STORE_PATH = "./data/profiles"

# Create directories if they don't exist
os.makedirs(DOCUMENT_STORE_PATH, exist_ok=True)
os.makedirs(CHUNK_STORE_PATH, exist_ok=True)
os.makedirs(PROFILE_STORE_PATH, exist_ok=True)

# Ensure the user query count file exists
if not os.path.exists(USER_QUERY_COUNT_PATH):
    with open(USER_QUERY_COUNT_PATH, "w") as f:
        json.dump({}, f)

# Helper functions to serialize/deserialize datetime objects
def serialize_datetime(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError("Type not serializable")

def document_to_dict(document: Document) -> Dict[str, Any]:
    """Convert Document model to dictionary for storage."""
    doc_dict = document.model_dump()
    
    # Convert enums to strings
    doc_dict["file_type"] = document.file_type.value
    doc_dict["status"] = document.status.value
    
    # Remove chunks from dictionary
    if "chunks" in doc_dict:
        del doc_dict["chunks"]
    
    return doc_dict

def dict_to_document(doc_dict: Dict[str, Any]) -> Document:
    """Convert dictionary to Document model."""
    # Convert string timestamps to datetime objects
    if isinstance(doc_dict.get("created_at"), str):
        doc_dict["created_at"] = datetime.fromisoformat(doc_dict["created_at"])
    
    if isinstance(doc_dict.get("updated_at"), str):
        doc_dict["updated_at"] = datetime.fromisoformat(doc_dict["updated_at"])
    
    return Document(**doc_dict)

async def save_document(document: Document) -> bool:
    """
    Save a document to the document store.
    
    Args:
        document: The document to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert document to dictionary
        doc_dict = document_to_dict(document)
        
        # Save document to file
        document_path = os.path.join(DOCUMENT_STORE_PATH, f"{document.id}.json")
        
        with open(document_path, "w") as f:
            json.dump(doc_dict, f, default=serialize_datetime)
        
        return True
    except Exception as e:
        print(f"Error saving document: {str(e)}")
        return False

async def get_document_by_id(document_id: str) -> Optional[Document]:
    """
    Get a document by ID.
    
    Args:
        document_id: ID of the document to retrieve
        
    Returns:
        Document if found, None otherwise
    """
    try:
        document_path = os.path.join(DOCUMENT_STORE_PATH, f"{document_id}.json")
        
        if not os.path.exists(document_path):
            return None
        
        with open(document_path, "r") as f:
            doc_dict = json.load(f)
        
        return dict_to_document(doc_dict)
    except Exception as e:
        print(f"Error retrieving document: {str(e)}")
        return None

async def get_all_documents() -> List[Document]:
    """
    Get all documents.
    
    Returns:
        List of all documents
    """
    documents = []
    
    try:
        for filename in os.listdir(DOCUMENT_STORE_PATH):
            if filename.endswith(".json"):
                document_path = os.path.join(DOCUMENT_STORE_PATH, filename)
                
                with open(document_path, "r") as f:
                    doc_dict = json.load(f)
                
                documents.append(dict_to_document(doc_dict))
    except Exception as e:
        print(f"Error retrieving documents: {str(e)}")
    
    return documents

async def get_documents_by_repository_id(repository_id: str) -> List[Document]:
    """
    Get all documents for a repository.
    
    Args:
        repository_id: ID of the repository
        
    Returns:
        List of documents in the repository
    """
    documents = []
    
    try:
        for filename in os.listdir(DOCUMENT_STORE_PATH):
            if filename.endswith(".json"):
                document_path = os.path.join(DOCUMENT_STORE_PATH, filename)
                
                with open(document_path, "r") as f:
                    doc_dict = json.load(f)
                
                if doc_dict.get("repository_id") == repository_id:
                    documents.append(dict_to_document(doc_dict))
    except Exception as e:
        print(f"Error retrieving documents: {str(e)}")
    
    return documents

async def get_documents_by_user_id(user_id: str) -> List[Document]:
    """
    Get all documents for a user across all repositories.
    
    Args:
        user_id: ID of the user
        
    Returns:
        List of documents for the user
    """
    documents = []
    
    try:
        for filename in os.listdir(DOCUMENT_STORE_PATH):
            if filename.endswith(".json"):
                document_path = os.path.join(DOCUMENT_STORE_PATH, filename)
                
                with open(document_path, "r") as f:
                    doc_dict = json.load(f)
                
                if doc_dict.get("user_id") == user_id:
                    documents.append(dict_to_document(doc_dict))
    except Exception as e:
        print(f"Error retrieving documents: {str(e)}")
    
    return documents

async def update_document_status(document_id: str, status: ProcessingStatus, status_message: str = None) -> bool:
    """
    Update the processing status of a document.
    
    Args:
        document_id: ID of the document
        status: New processing status
        status_message: Optional status message with details
        
    Returns:
        True if successful, False otherwise
    """
    try:
        document = await get_document_by_id(document_id)
        if not document:
            return False
        
        document.status = status
        
        # Add status message to metadata if provided
        if status_message:
            if document.metadata is None:
                document.metadata = {}
            document.metadata["status_message"] = status_message
        
        document.updated_at = datetime.now()
        
        # Save document
        await save_document(document)
        
        return True
    except Exception as e:
        print(f"Error updating document status: {str(e)}")
        return False

async def delete_document(document_id: str) -> bool:
    """
    Delete a document and its associated chunks.
    
    Args:
        document_id: ID of the document to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Delete document from vector store
        await delete_document_chunks(document_id)
        
        # Delete document file
        document_path = os.path.join(DOCUMENT_STORE_PATH, f"{document_id}.json")
        
        if os.path.exists(document_path):
            os.remove(document_path)
            return True
        
        return False
    except Exception as e:
        print(f"Error deleting document: {str(e)}")
        return False

async def delete_documents_by_repository_id(repository_id: str) -> int:
    """
    Delete all documents in a repository.
    
    Args:
        repository_id: ID of the repository
        
    Returns:
        Number of documents deleted
    """
    deleted_count = 0
    
    try:
        # Get all documents in the repository
        documents = await get_documents_by_repository_id(repository_id)
        
        # Delete each document
        for document in documents:
            if await delete_document(document.id):
                deleted_count += 1
    except Exception as e:
        print(f"Error deleting documents: {str(e)}")
    
    return deleted_count 

async def get_user_query_count(user_id: str) -> int:
    """Get the persistent query count for a user."""
    try:
        with open(USER_QUERY_COUNT_PATH, "r") as f:
            counts = json.load(f)
        return counts.get(user_id, 0)
    except Exception as e:
        print(f"Error reading user query count: {str(e)}")
        return 0

async def increment_user_query_count(user_id: str) -> int:
    """Increment and return the persistent query count for a user."""
    try:
        # Read current counts
        if os.path.exists(USER_QUERY_COUNT_PATH):
            with open(USER_QUERY_COUNT_PATH, "r") as f:
                counts = json.load(f)
        else:
            counts = {}
        # Increment
        counts[user_id] = counts.get(user_id, 0) + 1
        # Write back
        with open(USER_QUERY_COUNT_PATH, "w") as f:
            json.dump(counts, f)
        return counts[user_id]
    except Exception as e:
        print(f"Error incrementing user query count: {str(e)}")
        return 0

# Profile Storage Functions
async def save_user_profiles(user_id: str, profiles: Dict[str, Dict[str, Any]]) -> bool:
    """
    Save user profiles to persistent storage in rich format.
    
    Args:
        user_id: ID of the user
        profiles: Dictionary of profiles to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        profile_path = os.path.join(PROFILE_STORE_PATH, f"{user_id}.json")
        
        print(f"Original profiles to save: {profiles}")
        
        # Convert profiles to rich format before saving
        profiles_dict = {}
        for name_lower, profile in profiles.items():
            print(f"Processing profile for key '{name_lower}': {profile}")
            
            # Check if this is already a rich profile
            if is_rich_profile(profile):
                print(f"Profile '{name_lower}' is already rich, saving as is")
                # Already rich, save as is
                profiles_dict[name_lower] = profile
            else:
                print(f"Profile '{name_lower}' is not rich, converting to rich format")
                # Convert simple profile to rich format
                try:
                    rich_profile = await convert_simple_to_rich_profile(profile)
                    print(f"Converted profile '{name_lower}' to rich format: {rich_profile}")
                    profiles_dict[name_lower] = rich_profile
                except Exception as e:
                    print(f"Failed to convert profile '{name_lower}' to rich format: {str(e)}")
                    # Keep the simple profile if conversion fails
                    profiles_dict[name_lower] = profile
        
        print(f"Saving user profiles in rich format: {profiles_dict}")
        
        with open(profile_path, "w") as f:
            json.dump(profiles_dict, f, default=serialize_datetime, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving user profiles: {str(e)}")
        return False

async def load_user_profiles(user_id: str) -> Dict[str, Dict[str, Any]]:
    """
    Load user profiles from persistent storage.
    
    Args:
        user_id: ID of the user
        
    Returns:
        Dictionary of user profiles, empty dict if none exist
    """
    try:
        profile_path = os.path.join(PROFILE_STORE_PATH, f"{user_id}.json")
        
        if not os.path.exists(profile_path):
            return {}
        
        with open(profile_path, "r") as f:
            profiles_data = json.load(f)
        
        # Handle both dictionary and array formats
        profiles = {}
        if isinstance(profiles_data, dict):
            # Already in dictionary format
            profiles = profiles_data
        elif isinstance(profiles_data, list):
            # Convert array format to dictionary
            for profile in profiles_data:
                name = profile.get('name', '')
                if name:
                    name_lower = name.lower()
                    profiles[name_lower] = profile
        
        return profiles
    except Exception as e:
        print(f"Error loading user profiles: {str(e)}")
        return {}

async def update_user_profile(user_id: str, name: str, profile_data: Dict[str, Any]) -> bool:
    """
    Update a specific profile for a user.
    
    Args:
        user_id: ID of the user
        name: Name of the person (will be converted to lowercase for storage)
        profile_data: Profile data to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load existing profiles
        profiles = await load_user_profiles(user_id)
        
        # Update the specific profile
        name_lower = name.lower()
        profiles[name_lower] = profile_data
        
        # Save back to storage
        return await save_user_profiles(user_id, profiles)
    except Exception as e:
        print(f"Error updating user profile: {str(e)}")
        return False

async def delete_user_profile(user_id: str, name: str) -> bool:
    """
    Delete a specific profile for a user.
    
    Args:
        user_id: ID of the user
        name: Name of the person to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load existing profiles
        profiles = await load_user_profiles(user_id)
        
        # Remove the specific profile
        name_lower = name.lower()
        if name_lower in profiles:
            del profiles[name_lower]
            
            # Save back to storage
            return await save_user_profiles(user_id, profiles)
        
        return True  # Profile didn't exist, so deletion is successful
    except Exception as e:
        print(f"Error deleting user profile: {str(e)}")
        return False

async def get_user_profile(user_id: str, name: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific profile for a user.
    
    Args:
        user_id: ID of the user
        name: Name of the person
        
    Returns:
        Profile data if found, None otherwise
    """
    try:
        profiles = await load_user_profiles(user_id)
        name_lower = name.lower()
        return profiles.get(name_lower)
    except Exception as e:
        print(f"Error getting user profile: {str(e)}")
        return None

async def get_all_user_profiles(user_id: str) -> Dict[str, Dict[str, Any]]:
    """
    Get all profiles for a user.
    
    Args:
        user_id: ID of the user
        
    Returns:
        Dictionary of all user profiles
    """
    return await load_user_profiles(user_id)

# User Name Storage Functions
async def save_user_name(user_id: str, name: str) -> bool:
    """
    Save the user's own name to persistent storage.
    
    Args:
        user_id: ID of the user
        name: User's name
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create a special file for user's own name
        user_name_path = os.path.join(PROFILE_STORE_PATH, f"{user_id}_name.json")
        
        user_data = {
            'user_name': name,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(user_name_path, "w") as f:
            json.dump(user_data, f, default=serialize_datetime, indent=2)
        
        print(f"Saved user name for {user_id}: {name}")
        return True
    except Exception as e:
        print(f"Error saving user name: {str(e)}")
        return False

async def load_user_name(user_id: str) -> Optional[str]:
    """
    Load the user's own name from persistent storage.
    
    Args:
        user_id: ID of the user
        
    Returns:
        User's name if found, None otherwise
    """
    try:
        user_name_path = os.path.join(PROFILE_STORE_PATH, f"{user_id}_name.json")
        
        if not os.path.exists(user_name_path):
            return None
        
        with open(user_name_path, "r") as f:
            user_data = json.load(f)
        
        return user_data.get('user_name')
    except Exception as e:
        print(f"Error loading user name: {str(e)}")
        return None 
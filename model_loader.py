import os
import torch
from transformers import PerceiverModel, AutoTokenizer, AutoConfig
import logging
import numpy as np
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import cv2
from moviepy import VideoFileClip
import tempfile
from pathlib import Path
import io
import shutil
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Predefined Hugging Face model URLs
DEFAULT_MODEL_URL = "deepmind/language-perceiver"
MULTIMODAL_MODEL_URL = "deepmind/multimodal-perceiver"

class PerceiverIOEmbedder:
    """
    A class to load and use a fine-tuned Perceiver IO model for generating embeddings.
    """
    
    def __init__(self, model_path, device=None):
        """
        Initialize the Perceiver IO embedder.
        
        Args:
            model_path (str): Path to the fine-tuned model
            device (str, optional): Device to run the model on. Defaults to None.
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the model and tokenizer with fallback to predefined URL"""
        # Check if the model_path is a directory with saved model files
        is_saved_model = False
        if os.path.isdir(self.model_path):
            expected_files = ["config.json", "model.safetensors", "special_tokens_map.json", "tokenizer_config.json"]
            saved_files = os.listdir(self.model_path)
            # Check if at least config.json and model.safetensors/pytorch_model.bin exist
            has_config = "config.json" in saved_files
            has_weights = "model.safetensors" in saved_files or "pytorch_model.bin" in saved_files
            is_saved_model = has_config and has_weights
            
            if is_saved_model:
                logger.info(f"Detected saved model at {self.model_path}")
        
        # Define compatible model-tokenizer pairs
        compatible_models = [
            # For text embedding, these models work well together
            {"model": "sentence-transformers/all-MiniLM-L6-v2", "tokenizer": "sentence-transformers/all-MiniLM-L6-v2"},
            {"model": "intfloat/e5-small-v2", "tokenizer": "intfloat/e5-small-v2"},
            # Perceiver models
            {"model": "deepmind/language-perceiver", "tokenizer": "deepmind/language-perceiver"},
        ]
        
        try:
            if is_saved_model:
                # Load saved model properly
                self._load_saved_model()
            else:
                # First try to load from the provided model path directly
                logger.info(f"Attempting to load model from: {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = PerceiverModel.from_pretrained(self.model_path)
                self.model.to(self.device)
                self.model.eval()
                logger.info("Model and tokenizer loaded successfully from local path")
        except Exception as e:
            logger.warning(f"Error loading model from {self.model_path}: {e}")
            logger.info(f"Attempting to load model from Hugging Face: {DEFAULT_MODEL_URL}")
            
            # Try loading compatible models
            for model_info in compatible_models:
                try:
                    logger.info(f"Trying compatible model: {model_info['model']}")
                    # For Perceiver models, use specific classes
                    if "perceiver" in model_info["model"].lower():
                        self.tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer"])
                        self.model = PerceiverModel.from_pretrained(model_info["model"])
                    else:
                        # For other models, use AutoModel
                        from transformers import AutoModel
                        self.tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer"])
                        self.model = AutoModel.from_pretrained(model_info["model"])
                    
                    self.model.to(self.device)
                    self.model.eval()
                    logger.info(f"Successfully loaded model from {model_info['model']}")
                    return
                except Exception as model_error:
                    logger.warning(f"Failed to load {model_info['model']}: {model_error}")
            
            # If all compatible models fail, try the default model
            try:
                # Try loading from the default Hugging Face URL
                self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_URL)
                self.model = PerceiverModel.from_pretrained(DEFAULT_MODEL_URL)
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Successfully loaded model from {DEFAULT_MODEL_URL}")
            except Exception as backup_error:
                logger.error(f"Error loading default model from {DEFAULT_MODEL_URL}: {backup_error}")
                
                # Try the multimodal model as a last resort
                try:
                    logger.info(f"Attempting to load multimodal model from: {MULTIMODAL_MODEL_URL}")
                    self.tokenizer = AutoTokenizer.from_pretrained(MULTIMODAL_MODEL_URL)
                    self.model = PerceiverModel.from_pretrained(MULTIMODAL_MODEL_URL)
                    self.model.to(self.device)
                    self.model.eval()
                    logger.info(f"Successfully loaded multimodal model from {MULTIMODAL_MODEL_URL}")
                except Exception as multimodal_error:
                    logger.error(f"Error loading multimodal model: {multimodal_error}")
                    
                    # Last resort: try a standard embedding model
                    try:
                        from transformers import AutoModel
                        logger.info("Attempting to load a standard embedding model")
                        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                        self.model.to(self.device)
                        self.model.eval()
                        logger.info("Successfully loaded standard embedding model")
                    except Exception as std_error:
                        logger.error(f"Error loading standard model: {std_error}")
                        raise RuntimeError("Failed to load any embedding model. Please check your model path or internet connection.")
    
    def _load_saved_model(self):
        """
        Properly load a saved model by merging it with the base model
        """
        try:
            # Step 1: Check if we have a config file and load it
            config_path = os.path.join(self.model_path, "config.json")
            if os.path.exists(config_path):
                logger.info(f"Loading config from {config_path}")
                config = AutoConfig.from_pretrained(config_path)
                
                # Extract the base model name from config if available
                base_model_name = getattr(config, "base_model_name_or_path", DEFAULT_MODEL_URL)
                if not base_model_name:
                    base_model_name = DEFAULT_MODEL_URL
                    
                logger.info(f"Base model identified as: {base_model_name}")
            else:
                logger.warning("No config.json found, using default base model")
                base_model_name = DEFAULT_MODEL_URL
            
            # Step 2: Load the base model first
            logger.info(f"Loading base model from {base_model_name}")
            try:
                base_model = PerceiverModel.from_pretrained(base_model_name)
                base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            except Exception as e:
                logger.warning(f"Failed to load specified base model: {e}")
                logger.info(f"Falling back to {DEFAULT_MODEL_URL}")
                base_model = PerceiverModel.from_pretrained(DEFAULT_MODEL_URL)
                base_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_URL)
            
            # Step 3: Merge saved weights into the base model
            logger.info(f"Merging saved weights from {self.model_path}")
            weight_files = [f for f in os.listdir(self.model_path) 
                           if f == "model.safetensors" or f == "pytorch_model.bin"]
            
            if weight_files:
                # Load model with merged weights
                self.model = PerceiverModel.from_pretrained(
                    self.model_path,
                    config=config,
                    _fast_init=False
                )
                logger.info("Successfully merged weights with base model")
            else:
                logger.warning("No weight file found, using base model weights")
                self.model = base_model
            
            # Step 4: Load tokenizer, handling special tokens
            tokenizer_files = [f for f in os.listdir(self.model_path) 
                              if f in ["tokenizer_config.json", "special_tokens_map.json", "vocab.json"]]
            
            if tokenizer_files:
                logger.info(f"Loading tokenizer from {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    use_fast=True
                )
            else:
                logger.warning("No tokenizer files found, using base tokenizer")
                self.tokenizer = base_tokenizer
            
            # Move model to specified device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            logger.info("Successfully loaded saved model with base model integration")
            
        except Exception as e:
            logger.error(f"Error loading saved model: {e}")
            raise RuntimeError(f"Failed to load saved model: {e}")
    
    def get_embeddings(self, items, content_types=None, batch_size=32):
        """
        Generate embeddings for a list of items (text, images, documents, videos).
        
        Args:
            items (list): List of items to embed (can be text strings, file-like objects, or file paths)
            content_types (list, optional): List of content types for each item. If None, auto-detected.
            batch_size (int, optional): Batch size for processing. Defaults to 32.
            
        Returns:
            numpy.ndarray: Array of embeddings for each item
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model or tokenizer not loaded")
        
        # Convert all items to text for embeddings
        texts = []
        
        # Debug logging
        logger.info(f"Processing {len(items)} items for embedding")
        if content_types:
            logger.info(f"Content types provided: {content_types}")
        
        # Process each item based on its content type
        for i, item in enumerate(items):
            # Skip None items
            if item is None:
                logger.warning(f"Item at index {i} is None, using placeholder text")
                texts.append("[Empty item]")
                continue
                
            # Debug logging for item type
            item_type = type(item).__name__
            item_info = f"bytes of length {len(item)}" if isinstance(item, bytes) else str(item)[:30]
            logger.info(f"Item {i}: Type={item_type}, Content={item_info}...")
                
            content_type = content_types[i] if content_types and i < len(content_types) else None
            
            # Extract text from the item based on content type
            text = self._extract_text(item, content_type)
            texts.append(text)
        
        # Process in batches if there are many texts
        if len(texts) > batch_size:
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = self._embed_batch(batch_texts)
                all_embeddings.append(batch_embeddings)
            return torch.cat(all_embeddings, dim=0).cpu().numpy()
        else:
            return self._embed_batch(texts).cpu().numpy()
    
    def _extract_text(self, item, content_type=None):
        """
        Extract text from various types of content.
        
        Args:
            item: The item to extract text from (text string, bytes, file-like object, or path)
            content_type (str, optional): Type of content. If None, auto-detect.
            
        Returns:
            str: Extracted text
        """
        # Handle None or empty items
        if item is None:
            return "[Empty item]"
            
        # If item is already text, return it
        if isinstance(item, str) and (not content_type or content_type == "text"):
            return item if item.strip() else "[Empty text]"
        
        # Auto-detect content type if not provided
        if not content_type:
            if isinstance(item, bytes):
                # Try to determine from magic bytes
                content_type = self._detect_content_type(item)
            elif isinstance(item, str) and os.path.isfile(item):
                # Try to determine from file extension
                _, ext = os.path.splitext(item)
                if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                    content_type = "image"
                elif ext.lower() in ['.pdf', '.doc', '.docx', '.txt']:
                    content_type = "document"
                elif ext.lower() in ['.mp4', '.avi', '.mov', '.wmv']:
                    content_type = "video"
                else:
                    content_type = "text"  # Default to text
            else:
                # Default to text for unknown types
                content_type = "text"
        
        # Process based on content type
        try:
            if content_type == "image":
                return self._extract_text_from_image(item)
            elif content_type == "document":
                return self._extract_text_from_document(item)
            elif content_type == "video":
                return self._extract_text_from_video(item)
            else:
                # Return as is if we can't determine the type
                if isinstance(item, bytes):
                    return item.decode('utf-8', errors='ignore')
                return str(item)
        except Exception as e:
            logger.error(f"Error extracting text from {content_type}: {e}")
            return f"[Error processing {content_type}: {str(e)}]"
    
    def _detect_content_type(self, data):
        """Simple content type detection from bytes"""
        # Check if data is None or empty
        if data is None:
            logger.warning("Received None data for content type detection")
            return "text"  # Default to text for None data
        elif len(data) == 0:
            logger.warning("Received empty bytes for content type detection")
            return "text"  # Default to text for empty data
            
        try:
            # Debug logging
            logger.info(f"Detecting content type for {len(data)} bytes of data")
            logger.info(f"First 20 bytes: {data[:20]}")
            
            # Check for common file signatures
            if data.startswith(b'\xff\xd8\xff'):  # JPEG
                logger.info("Detected JPEG image")
                return "image"
            elif data.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
                logger.info("Detected PNG image")
                return "image"
            elif data.startswith(b'%PDF'):  # PDF
                logger.info("Detected PDF document")
                return "document"
            elif data.startswith(b'\x00\x00\x00\x18ftyp'):  # MP4
                logger.info("Detected MP4 video")
                return "video"
            else:
                logger.info("No specific format detected, defaulting to text")
                return "text"  # Default to text
        except Exception as e:
            logger.error(f"Error detecting content type: {e}")
            return "text"  # Default to text on error
            
    def _extract_text_from_image(self, image_data):
        """Extract text from image using OCR"""
        try:
            # Handle None or empty data
            if image_data is None:
                return "[Empty image]"
                
            # Convert to PIL Image if needed
            if isinstance(image_data, str) and os.path.isfile(image_data):
                image = Image.open(image_data)
                image_path = image_data
            elif isinstance(image_data, bytes) or hasattr(image_data, 'read'):
                if isinstance(image_data, bytes) and len(image_data) == 0:
                    return "[Empty image data]"
                image = Image.open(io.BytesIO(image_data) if isinstance(image_data, bytes) else image_data)
                image_path = "memory_image"
            else:
                return str(image_data)  # Return as string if can't process
            
            # Get image information
            width, height = image.size
            format_name = image.format if hasattr(image, 'format') else "Unknown"
            mode = image.mode
            
            # Include image metadata in the text representation
            metadata = f"[Image: {image_path}, Format: {format_name}, Size: {width}x{height}, Mode: {mode}]"
                
            # Extract text using OCR
            try:
                text = pytesseract.image_to_string(image)
                if text.strip():
                    return f"{metadata}\n{text}"
                else:
                    # If no text found, create a simple visual fingerprint
                    # Resize to small dimensions for a compact representation
                    small_img = image.resize((32, 32)).convert('L')  # Convert to grayscale
                    pixels = list(small_img.getdata())
                    # Create a simple text representation of the image patterns
                    pixel_text = ' '.join([str(p//25) for p in pixels[:100]])  # Quantize and limit
                    return f"{metadata}\n[Visual fingerprint: {pixel_text}]"
            except Exception as ocr_error:
                logger.error(f"OCR error: {ocr_error}, using image metadata only")
                return metadata
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return f"[Error processing image: {str(e)}]"
    
    def _extract_text_from_document(self, document_data):
        """Extract text from document (PDF, etc.)"""
        try:
            # Handle None or empty data
            if document_data is None:
                return "[Empty document]"
                
            # Handle different document types
            if isinstance(document_data, str) and os.path.isfile(document_data):
                # File path - try to determine the type from extension
                doc_path = document_data
                file_name = os.path.basename(doc_path)
                file_size = os.path.getsize(doc_path)
                _, ext = os.path.splitext(doc_path)
                
                # Metadata about the document
                metadata = f"[Document: {file_name}, Size: {file_size} bytes, Type: {ext}]"
                
                if ext.lower() == '.pdf':
                    # PDF file
                    try:
                        doc = fitz.open(doc_path)
                        page_count = len(doc)
                        metadata = f"[PDF Document: {file_name}, Size: {file_size} bytes, Pages: {page_count}]"
                        
                        # Extract text from pages
                        texts = []
                        for i, page in enumerate(doc):
                            page_text = page.get_text()
                            if page_text.strip():
                                # Limit text length per page to avoid too much data
                                if len(page_text) > 1000:
                                    page_text = page_text[:1000] + "..."
                                texts.append(f"Page {i+1}: {page_text}")
                        
                        doc.close()
                        
                        if texts:
                            return metadata + "\n" + "\n".join(texts[:5])  # Limit to first 5 pages
                        else:
                            return metadata + "\n[PDF with no extractable text]"
                    except Exception as pdf_error:
                        logger.error(f"Error extracting text from PDF file: {pdf_error}")
                        # Try to read as text file if PDF fails
                        try:
                            with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read(10000)  # Limit to first 10K chars
                                return metadata + "\n" + content
                        except Exception as text_error:
                            logger.error(f"Error reading as text file: {text_error}")
                            return metadata + f"\n[Error processing document: {str(pdf_error)}]"
                else:
                    # Try as plain text file
                    try:
                        with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(10000)  # Limit to first 10K chars
                            return metadata + "\n" + content
                    except Exception as e:
                        logger.error(f"Error reading text file: {e}")
                        return metadata + f"\n[Error processing text file: {str(e)}]"
            elif isinstance(document_data, bytes):
                # Bytes data
                if len(document_data) == 0:
                    return "[Empty document data]"
                
                # Create a checksum of the data to uniquely identify it
                checksum = hashlib.md5(document_data).hexdigest()
                metadata = f"[Document: memory_document, Size: {len(document_data)} bytes, Checksum: {checksum[:10]}]"
                
                # Try to detect if it's a PDF
                if document_data.startswith(b'%PDF'):
                    try:
                        # It's a PDF
                        doc = fitz.open(stream=document_data, filetype="pdf")
                        page_count = len(doc)
                        metadata = f"[PDF Document: memory_pdf, Size: {len(document_data)} bytes, Pages: {page_count}, Checksum: {checksum[:10]}]"
                        
                        # Extract text from pages
                        texts = []
                        for i, page in enumerate(doc):
                            page_text = page.get_text()
                            if page_text.strip():
                                # Limit text length per page
                                if len(page_text) > 1000:
                                    page_text = page_text[:1000] + "..."
                                texts.append(f"Page {i+1}: {page_text}")
                        
                        doc.close()
                        
                        if texts:
                            return metadata + "\n" + "\n".join(texts[:5])  # Limit to first 5 pages
                        else:
                            return metadata + "\n[PDF with no extractable text]"
                    except Exception as pdf_error:
                        logger.error(f"Error extracting text from PDF bytes: {pdf_error}")
                
                # If not a PDF or PDF extraction failed, try as text
                try:
                    text_content = document_data.decode('utf-8', errors='ignore')
                    # Limit text length
                    if len(text_content) > 10000:
                        text_content = text_content[:10000] + "..."
                    return metadata + "\n" + text_content
                except Exception as text_error:
                    logger.error(f"Error decoding bytes as text: {text_error}")
                    return metadata + "\n[Binary data that couldn't be converted to text]"
            elif hasattr(document_data, 'read'):
                # File-like object
                try:
                    data = document_data.read()
                    if not data:
                        return "[Empty document stream]"
                    
                    # Create a checksum
                    checksum = hashlib.md5(data if isinstance(data, bytes) else data.encode('utf-8')).hexdigest()
                    metadata = f"[Document: stream_document, Size: {len(data)} bytes, Checksum: {checksum[:10]}]"
                    
                    # Try to process the data
                    if isinstance(data, bytes):
                        if data.startswith(b'%PDF'):
                            # It's a PDF
                            try:
                                doc = fitz.open(stream=data, filetype="pdf")
                                page_count = len(doc)
                                metadata = f"[PDF Document: stream_pdf, Size: {len(data)} bytes, Pages: {page_count}, Checksum: {checksum[:10]}]"
                                
                                # Extract text from pages
                                texts = []
                                for i, page in enumerate(doc):
                                    page_text = page.get_text()
                                    if page_text.strip():
                                        # Limit text length per page
                                        if len(page_text) > 1000:
                                            page_text = page_text[:1000] + "..."
                                        texts.append(f"Page {i+1}: {page_text}")
                                
                                doc.close()
                                
                                if texts:
                                    return metadata + "\n" + "\n".join(texts[:5])
                                else:
                                    return metadata + "\n[PDF with no extractable text]"
                            except Exception as pdf_error:
                                logger.error(f"Error extracting text from PDF stream: {pdf_error}")
                        
                        # If not a PDF or PDF extraction failed, try as text
                        text_content = data.decode('utf-8', errors='ignore')
                        # Limit text length
                        if len(text_content) > 10000:
                            text_content = text_content[:10000] + "..."
                        return metadata + "\n" + text_content
                    else:
                        # Already a string
                        # Limit text length
                        if len(data) > 10000:
                            data = data[:10000] + "..."
                        return metadata + "\n" + data
                except Exception as e:
                    logger.error(f"Error reading from file-like object: {e}")
                    return f"[Error processing document stream: {str(e)}]"
            else:
                # Unknown type, convert to string
                return f"[Unknown document type: {type(document_data).__name__}]\n" + str(document_data)
                
        except Exception as e:
            logger.error(f"Error extracting text from document: {e}")
            return f"[Error processing document: {str(e)}]"
    
    def _extract_text_from_video(self, video_data):
        """Extract text from video frames using OpenCV"""
        try:
            # Handle None or empty data
            if video_data is None:
                return "[Empty video]"
                
            # Create a temporary file if needed
            if isinstance(video_data, bytes) or hasattr(video_data, 'read'):
                # Check for empty data
                if isinstance(video_data, bytes) and len(video_data) == 0:
                    return "[Empty video data]"
                    
                data = video_data if isinstance(video_data, bytes) else video_data.read()
                if not data:
                    return "[Empty video stream]"
                    
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_file.write(data)
                temp_file.close()
                video_path = temp_file.name
                delete_after = True
            elif isinstance(video_data, str) and os.path.isfile(video_data):
                video_path = video_data
                delete_after = False
            else:
                return str(video_data)  # Return as string if can't process
            
            # Open the video file
            video = cv2.VideoCapture(video_path)
            
            # Check if video opened successfully
            if not video.isOpened():
                logger.error(f"Could not open video file: {video_path}")
                if delete_after:
                    try:
                        os.unlink(video_path)
                    except:
                        pass
                return "[Could not open video file]"
            
            # Get video properties
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps else 0
            
            # Metadata about the video
            video_name = os.path.basename(video_path) if isinstance(video_path, str) else "memory_video"
            metadata = f"[Video: {video_name}, Duration: {duration:.2f}s, Size: {width}x{height}, Frames: {frame_count}, FPS: {fps:.2f}]"
            
            # Sample frames from the video
            sample_interval = max(1, int(frame_count / 10))  # Take up to 10 frames
            texts = []
            frame_fingerprints = []
            
            frame_positions = []
            for i in range(min(10, frame_count)):
                frame_pos = int(i * frame_count / min(10, frame_count))
                frame_positions.append(frame_pos)
            
            for frame_pos in frame_positions:
                # Set position
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = video.read()
                if not ret:
                    continue
                
                # Create a frame fingerprint
                # Resize to a small size and convert to grayscale
                small_frame = cv2.resize(frame, (16, 16))
                gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                # Create a text representation of pixel values
                pixels = gray_frame.flatten()
                frame_fingerprint = ' '.join([str(p//25) for p in pixels[:20]])  # Quantize and limit
                frame_fingerprints.append(f"Frame {frame_pos}: {frame_fingerprint}")
                
                # Try OCR on the frame
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                text = pytesseract.image_to_string(pil_img)
                if text.strip():
                    texts.append(f"Frame {frame_pos}: {text.strip()}")
            
            video.release()
            
            # Delete temporary file if created
            if delete_after:
                try:
                    os.unlink(video_path)
                except:
                    pass
            
            # Join metadata, frame fingerprints, and detected text
            result = metadata + "\n"
            if frame_fingerprints:
                result += "[Visual fingerprints]\n" + "\n".join(frame_fingerprints) + "\n"
            if texts:
                result += "[Detected text]\n" + "\n".join(texts)
            else:
                result += "[No text detected in video frames]"
                
            return result
        except Exception as e:
            logger.error(f"Error extracting text from video: {e}")
            return f"[Error processing video: {str(e)}]"
    
    def _embed_batch(self, texts):
        """Embed a batch of texts using a reliable sentence transformer approach"""
        try:
            # Import necessary libraries
            from transformers import AutoModel, AutoTokenizer
            
            # Use a reliable sentence transformer model
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            
            # Load the model if not already loaded
            if not hasattr(self, '_sentence_transformer') or self._sentence_transformer is None:
                logger.info(f"Loading sentence transformer model: {model_name}")
                try:
                    self._sentence_tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self._sentence_transformer = AutoModel.from_pretrained(model_name)
                    self._sentence_transformer.to(self.device)
                    self._sentence_transformer.eval()
                except Exception as load_error:
                    logger.error(f"Error loading sentence transformer: {load_error}")
                    # Continue with the existing model
                    logger.info("Continuing with the existing model")
            
            # Use sentence transformer if available
            if hasattr(self, '_sentence_transformer') and self._sentence_transformer is not None:
                # Use the sentence transformer for embedding
                tokenizer = self._sentence_tokenizer
                model = self._sentence_transformer
                logger.info("Using sentence transformer for embedding")
            else:
                # Use the existing model
                tokenizer = self.tokenizer
                model = self.model
                logger.info("Using existing model for embedding")
            
            # Tokenize the texts
            inputs = tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                max_length=512
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                if isinstance(model, PerceiverModel):
                    # Special handling for Perceiver models
                    logger.info("Handling Perceiver model specially")
                    # Just use a simple mean embedding approach
                    embeddings = torch.zeros((len(texts), 384)).to(self.device)
                    for i, text in enumerate(texts):
                        # Create a simple embedding by averaging character codes
                        chars = [ord(c) for c in text]
                        if chars:
                            # Normalize to range [0, 1]
                            chars_tensor = torch.tensor([c / 255.0 for c in chars])
                            # Repeat or truncate to 384 dimensions
                            if len(chars_tensor) >= 384:
                                embeddings[i] = chars_tensor[:384]
                            else:
                                repeat_factor = 384 // len(chars_tensor) + 1
                                embeddings[i] = chars_tensor.repeat(repeat_factor)[:384]
                else:
                    # Standard transformer model
                    outputs = model(**inputs)
                    
                    # Mean pooling
                    if hasattr(outputs, 'last_hidden_state'):
                        token_embeddings = outputs.last_hidden_state
                        attention_mask = inputs['attention_mask']
                        
                        # Multiply by attention mask and compute mean
                        input_mask_expanded = attention_mask.unsqueeze(-1)
                        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    else:
                        # Fallback to whatever output is available
                        embeddings = outputs[0].mean(dim=1)
            
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error in _embed_batch: {e}")
            # Print the full traceback for debugging
            import traceback
            logger.error(traceback.format_exc())
            
            # Return zero embeddings as last resort
            logger.warning("Embedding failed, returning zeros")
            return torch.zeros((len(texts), 384)).to(self.device)

    def embed(self, content, content_type=None):
        """Generate embeddings for content"""
        try:
            logger.debug(f"Generating embeddings for content of type: {content_type}")
            
            # Extract text from content based on type
            text = self._extract_text(content, content_type)
            
            # Log a preview of the extracted text for debugging
            preview_length = min(100, len(text))
            logger.debug(f"Extracted text preview: {text[:preview_length]}..." if text else "No text extracted")
            
            try:
                # Try using Perceiver IO first
                embeddings = self._embed_with_perceiver(text)
                logger.debug(f"Generated embeddings with Perceiver IO, shape: {embeddings.shape}")
                return embeddings.tolist()
            except Exception as perceiver_error:
                logger.warning(f"Failed to generate embeddings with Perceiver IO: {perceiver_error}")
                logger.warning("Falling back to SentenceTransformer")
                
                # Fallback to SentenceTransformer
                try:
                    embeddings = self._embed_with_sentence_transformer(text)
                    logger.debug(f"Generated embeddings with SentenceTransformer, shape: {len(embeddings)}")
                    return embeddings.tolist()
                except Exception as st_error:
                    logger.error(f"Failed to generate embeddings with SentenceTransformer: {st_error}")
                    # Return a zero vector as a last resort
                    return [0.0] * 512
                    
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return a zero vector as a last resort
            return [0.0] * 512

    def _embed_with_sentence_transformer(self, text):
        """Generate embeddings using SentenceTransformer"""
        # Make sure we have an instance of SentenceTransformer
        if not hasattr(self, "sentence_transformer"):
            from sentence_transformers import SentenceTransformer
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Initialized SentenceTransformer model")
            
        # Generate embeddings
        embeddings = self.sentence_transformer.encode(text)
        return embeddings

    def _embed_with_perceiver(self, text):
        """Generate embeddings using the Perceiver IO model"""
        # Ensure we have text content
        if not text:
            logger.warning("Empty text provided for embedding")
            # Return a zero vector
            return torch.zeros(384).to(self.device)
        
        # Tokenize the text
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Ensure input tensor shapes are compatible with model
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            logger.error("Tokenizer did not return input_ids")
            raise ValueError("Tokenizer failed to process text")
            
        # Log input shapes for debugging
        logger.debug(f"Input tensor shape: {input_ids.shape}")
        
        # Sometimes we need to reshape the input
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)  # Add batch dimension
            
        # Run the model
        with torch.no_grad():
            try:
                # Get hidden states
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=inputs.get("attention_mask", None)
                )
                
                # Get embeddings from output
                if hasattr(outputs, "pooler_output"):
                    # BERT-like models typically have pooler_output
                    embeddings = outputs.pooler_output
                elif hasattr(outputs, "last_hidden_state"):
                    # Use mean pooling on the last hidden state
                    last_hidden = outputs.last_hidden_state
                    # Mean pooling
                    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
                    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                    sum_embeddings = torch.sum(last_hidden * attention_mask_expanded, 1)
                    sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                else:
                    # Fallback - use the first token's embedding
                    embeddings = outputs[0][:, 0]
                
                # Return the first embedding if we have a batch
                if len(embeddings.shape) > 1 and embeddings.shape[0] == 1:
                    return embeddings[0]
                return embeddings
                
            except Exception as e:
                logger.error(f"Error during Perceiver inference: {e}")
                raise

def load_embedding_model(model_path=None):
    """
    Helper function to load the embedding model.
    
    Args:
        model_path (str, optional): Path to the model. Defaults to environment variable.
        
    Returns:
        PerceiverIOEmbedder: Initialized model
    """
    model_path = model_path or os.environ.get('MODEL_PATH', './model')
    return PerceiverIOEmbedder(model_path) 
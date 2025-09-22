import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
import concurrent.futures
import logging
from typing import List, Tuple, Dict, Any, Optional
import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
import time

# Configure logging
logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Advanced video processing class that implements Hybrid Multimodal Video RAG.
    This class extracts frames intelligently, processes them in parallel,
    and integrates audio transcription with visual content.
    """
    
    def __init__(self, 
                 default_interval: float = 5.0,
                 max_workers: int = 4,
                 whisper_model_name: str = "base"):
        """
        Initialize the VideoProcessor.
        
        Args:
            default_interval: Default interval in seconds for regular frame sampling
            max_workers: Maximum number of parallel workers for frame processing
            whisper_model_name: Name of the Whisper model to use for audio transcription
        """
        self.default_interval = default_interval
        self.max_workers = max_workers
        
        logger.info(f"Initializing VideoProcessor with interval={default_interval}s, workers={max_workers}")
        
        # Initialize whisper model
        try:
            logger.info(f"Loading Whisper model: {whisper_model_name}")
            start_time = time.time()
            self.whisper_model = whisper.load_model(whisper_model_name)
            load_time = time.time() - start_time
            logger.info(f"Whisper model '{whisper_model_name}' loaded successfully in {load_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{whisper_model_name}': {str(e)}")
            self.whisper_model = None
    
    async def process_video(self, file_path: str, 
                     max_duration: Optional[float] = None,
                     extract_audio: bool = True) -> str:
        """
        Process a video file and extract text content from frames and audio.
        
        Args:
            file_path: Path to the video file
            max_duration: Maximum duration in seconds to process (None for full video)
            extract_audio: Whether to extract and transcribe audio
            
        Returns:
            Extracted text content with timestamps
        """
        start_time = time.time()
        logger.info(f"Starting video processing for: {file_path}")
        logger.debug(f"Process parameters: max_duration={max_duration}, extract_audio={extract_audio}")
        
        try:
            # Extract video metadata
            metadata = self._extract_video_metadata(file_path)
            if not metadata:
                logger.error(f"Failed to extract metadata from video: {file_path}")
                return "Error: Could not extract video metadata."
            
            logger.info(f"Video metadata: duration={metadata.get('duration', 'unknown')}s, " +
                       f"fps={metadata.get('fps', 'unknown')}, " +
                       f"size={metadata.get('file_size_mb', 'unknown'):.2f}MB")
            
            # Extract frames intelligently
            logger.info(f"Starting intelligent frame extraction for video: {file_path}")
            frame_start = time.time()
            frames, timestamps = self._extract_intelligent_frames(
                file_path, 
                metadata,
                max_duration=max_duration
            )
            frame_time = time.time() - frame_start
            
            if not frames:
                logger.warning(f"No frames extracted from video: {file_path}")
                return "No frames could be extracted from the video."
            
            logger.info(f"Extracted {len(frames)} frames from video in {frame_time:.2f} seconds")
            logger.debug(f"Frame timestamps: {timestamps[:5]}... (showing first 5)")
            
            # Process frames in parallel
            logger.info(f"Starting parallel frame processing with {self.max_workers} workers")
            ocr_start = time.time()
            frame_results = self._process_frames_parallel(frames, timestamps)
            ocr_time = time.time() - ocr_start
            
            logger.info(f"OCR processing completed in {ocr_time:.2f} seconds, found text in {len(frame_results)} frames")
            if frame_results:
                logger.debug(f"First frame text sample: {frame_results[0][1][:100]}...")
            
            # Extract and process audio if available
            audio_text = ""
            if extract_audio and self.whisper_model:
                logger.info(f"Starting audio extraction and transcription for: {file_path}")
                audio_start = time.time()
                audio_text = self._extract_audio_content(file_path)
                audio_time = time.time() - audio_start
                
                if audio_text:
                    logger.info(f"Audio transcription completed in {audio_time:.2f} seconds, " +
                               f"extracted {len(audio_text)} characters")
                    logger.debug(f"Audio transcription sample: {audio_text[:100]}...")
                else:
                    logger.info("No audio content extracted or no audio track found")
            else:
                logger.info("Skipping audio extraction (disabled or no Whisper model)")
            
            # Combine results
            logger.info("Formatting final results")
            result_text = self._format_results(metadata, frame_results, audio_text)
            
            total_time = time.time() - start_time
            logger.info(f"Video processing completed in {total_time:.2f} seconds")
            return result_text
            
        except Exception as e:
            logger.exception(f"Error processing video: {str(e)}")
            return f"Error processing video: {str(e)}"
    
    def _extract_video_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from the video file"""
        logger.debug(f"Extracting metadata from video: {file_path}")
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                logger.error(f"Could not open video file with OpenCV: {file_path}")
                return {}
            
            # Get video properties
            fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))  # Ensure fps is at least 1
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Get file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            
            cap.release()
            
            metadata = {
                "fps": fps,
                "total_frames": total_frames,
                "width": width,
                "height": height,
                "duration": duration,
                "file_size_mb": file_size_mb
            }
            
            logger.debug(f"Video metadata extracted: {metadata}")
            return metadata
        except Exception as e:
            logger.exception(f"Error extracting video metadata: {str(e)}")
            return {}
    
    def _extract_intelligent_frames(self, 
                                   file_path: str, 
                                   metadata: Dict[str, Any],
                                   max_duration: Optional[float] = None) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract frames using a combination of scene detection and regular sampling
        
        Args:
            file_path: Path to the video file
            metadata: Video metadata
            max_duration: Maximum duration to process
            
        Returns:
            Tuple of (frames, timestamps)
        """
        frames = []
        timestamps = []
        
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                logger.error(f"Could not open video file: {file_path}")
                return [], []
            
            fps = metadata.get("fps", 30)
            total_frames = metadata.get("total_frames", 0)
            duration = metadata.get("duration", 0)
            
            # Limit duration if specified
            if max_duration and max_duration < duration:
                duration = max_duration
                total_frames = int(duration * fps)
                logger.debug(f"Limited processing to {max_duration}s ({total_frames} frames)")
            
            # Calculate frame interval for regular sampling
            frame_interval = int(fps * self.default_interval)
            logger.debug(f"Frame interval: {frame_interval} frames ({self.default_interval}s at {fps} fps)")
            
            # Initialize variables for scene detection
            prev_frame = None
            frame_count = 0
            regular_samples = 0
            scene_changes = 0
            
            # Threshold for scene change detection
            scene_threshold = 30.0
            
            logger.debug(f"Starting frame extraction with interval {self.default_interval}s ({frame_interval} frames)")
            
            while frame_count < total_frames:
                success, frame = cap.read()
                
                if not success:
                    logger.debug(f"End of video reached at frame {frame_count}")
                    break
                
                timestamp = frame_count / fps
                
                # Check if we've reached the maximum duration
                if max_duration and timestamp >= max_duration:
                    logger.debug(f"Reached maximum duration {max_duration}s at frame {frame_count}")
                    break
                
                # Regular interval sampling
                is_regular_sample = frame_count % frame_interval == 0
                
                # Scene change detection
                is_scene_change = False
                if prev_frame is not None:
                    # Convert frames to grayscale for comparison
                    gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    
                    # Calculate difference between frames
                    diff = cv2.absdiff(gray_curr, gray_prev)
                    mean_diff = np.mean(diff)
                    
                    # If difference is above threshold, consider it a scene change
                    if mean_diff > scene_threshold:
                        is_scene_change = True
                        scene_changes += 1
                
                # Save frame if it's a regular sample or scene change
                if is_regular_sample or is_scene_change:
                    frames.append(frame.copy())
                    timestamps.append(timestamp)
                    if is_regular_sample:
                        regular_samples += 1
                    
                    logger.debug(f"Extracted frame at {timestamp:.2f}s " + 
                                f"({'scene change' if is_scene_change else 'regular interval'})")
                
                # Update previous frame
                prev_frame = frame.copy()
                frame_count += 1
            
            cap.release()
            
            logger.info(f"Extracted {len(frames)} frames from video: " +
                       f"{regular_samples} regular samples, {scene_changes} scene changes")
            return frames, timestamps
            
        except Exception as e:
            logger.exception(f"Error extracting frames: {str(e)}")
            return [], []
    
    def _process_frames_parallel(self, 
                               frames: List[np.ndarray], 
                               timestamps: List[float]) -> List[Tuple[float, str]]:
        """
        Process multiple frames in parallel using ThreadPoolExecutor
        
        Args:
            frames: List of frames to process
            timestamps: List of timestamps corresponding to frames
            
        Returns:
            List of (timestamp, text) tuples
        """
        results = []
        
        try:
            logger.debug(f"Starting parallel processing of {len(frames)} frames with {self.max_workers} workers")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all frames for processing
                future_to_frame = {
                    executor.submit(self._process_single_frame, frame): (frame, timestamp)
                    for frame, timestamp in zip(frames, timestamps)
                }
                
                # Collect results as they complete
                frames_with_text = 0
                for future in concurrent.futures.as_completed(future_to_frame):
                    _, timestamp = future_to_frame[future]
                    try:
                        frame_text = future.result()
                        if frame_text.strip():
                            results.append((timestamp, frame_text))
                            frames_with_text += 1
                            logger.debug(f"Successfully extracted text from frame at {timestamp:.2f}s: {len(frame_text)} chars")
                        else:
                            logger.debug(f"No text found in frame at {timestamp:.2f}s")
                    except Exception as e:
                        logger.warning(f"Error processing frame at {timestamp:.2f}s: {str(e)}")
            
            # Sort results by timestamp
            results.sort(key=lambda x: x[0])
            
            logger.info(f"Parallel processing complete: {frames_with_text} of {len(frames)} frames contained text")
            return results
            
        except Exception as e:
            logger.exception(f"Error in parallel frame processing: {str(e)}")
            return []
    
    def _process_single_frame(self, frame: np.ndarray) -> str:
        """
        Process a single frame with optimized OCR techniques
        
        Args:
            frame: Frame to process
            
        Returns:
            Extracted text
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply different preprocessing techniques and collect results
            ocr_results = []
            
            # 1. Original image (color)
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            original_text = pytesseract.image_to_string(pil_image)
            if original_text.strip():
                ocr_results.append(original_text)
            
            # 2. Grayscale image
            gray_pil = Image.fromarray(gray)
            gray_text = pytesseract.image_to_string(gray_pil)
            if gray_text.strip():
                ocr_results.append(gray_text)
            
            # 3. Adaptive thresholding (very effective for text)
            adaptive_thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            adaptive_pil = Image.fromarray(adaptive_thresh)
            adaptive_text = pytesseract.image_to_string(adaptive_pil)
            if adaptive_text.strip():
                ocr_results.append(adaptive_text)
            
            # 4. Otsu's thresholding
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            otsu_pil = Image.fromarray(otsu)
            otsu_text = pytesseract.image_to_string(otsu_pil)
            if otsu_text.strip():
                ocr_results.append(otsu_text)
            
            # 5. CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(gray)
            _, clahe_thresh = cv2.threshold(clahe_img, 150, 255, cv2.THRESH_BINARY)
            clahe_pil = Image.fromarray(clahe_thresh)
            clahe_text = pytesseract.image_to_string(clahe_pil)
            if clahe_text.strip():
                ocr_results.append(clahe_text)
            
            # Combine results, removing duplicates
            if not ocr_results:
                return ""
            
            # Sort by length (longest first) as they're often most complete
            ocr_results.sort(key=len, reverse=True)
            
            # Remove duplicates
            seen = set()
            unique_results = []
            for result in ocr_results:
                normalized = ''.join(result.strip().lower().split())
                if normalized and normalized not in seen and len(normalized) > 5:
                    seen.add(normalized)
                    unique_results.append(result)
            
            # Return the best result (usually the longest one)
            if unique_results:
                return unique_results[0]
            return ""
            
        except Exception as e:
            logger.warning(f"Error in frame OCR processing: {str(e)}")
            return ""
    
    def _extract_audio_content(self, file_path: str) -> str:
        """
        Extract and transcribe audio content from the video
        
        Args:
            file_path: Path to the video file
            
        Returns:
            Transcribed audio text
        """
        try:
            if not self.whisper_model:
                logger.warning("Whisper model not available, skipping audio extraction")
                return ""
            
            # Extract audio to temporary file
            audio_path = f"{file_path}.wav"
            logger.debug(f"Extracting audio to temporary file: {audio_path}")
            
            # Check if video has audio
            video = VideoFileClip(file_path)
            if not video.audio:
                logger.info(f"Video has no audio track: {file_path}")
                video.close()
                return ""
            
            # Extract audio
            logger.info(f"Extracting audio from video: {file_path}")
            audio_extract_start = time.time()
            video.audio.write_audiofile(audio_path, logger=None)
            video.close()
            audio_extract_time = time.time() - audio_extract_start
            logger.debug(f"Audio extraction completed in {audio_extract_time:.2f} seconds")
            
            # Transcribe audio
            logger.info(f"Transcribing audio: {audio_path}")
            transcribe_start = time.time()
            result = self.whisper_model.transcribe(audio_path)
            transcription = result["text"]
            transcribe_time = time.time() - transcribe_start
            logger.debug(f"Audio transcription completed in {transcribe_time:.2f} seconds")
            
            # Clean up
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.debug(f"Removed temporary audio file: {audio_path}")
            
            logger.info(f"Audio transcription complete: {len(transcription)} chars")
            return transcription
            
        except Exception as e:
            logger.exception(f"Error extracting audio content: {str(e)}")
            return ""
    
    def _format_results(self, 
                      metadata: Dict[str, Any],
                      frame_results: List[Tuple[float, str]],
                      audio_text: str) -> str:
        """
        Format the results into a coherent text output
        
        Args:
            metadata: Video metadata
            frame_results: List of (timestamp, text) tuples from frames
            audio_text: Transcribed audio text
            
        Returns:
            Formatted text output
        """
        logger.debug("Formatting results into final output")
        result_text = "--- Video Content Extraction ---\n\n"
        
        # Add metadata
        duration = metadata.get("duration", 0)
        fps = metadata.get("fps", 0)
        file_size_mb = metadata.get("file_size_mb", 0)
        
        result_text += f"Video Duration: {int(duration)} seconds\n"
        result_text += f"Frame Rate: {fps} FPS\n"
        result_text += f"File Size: {file_size_mb:.2f} MB\n\n"
        
        # Add visual content if available
        if frame_results:
            result_text += "--- Visual Content Extraction ---\n\n"
            
            # Group results by temporal proximity (within 3 seconds)
            grouped_results = self._group_by_temporal_proximity(frame_results)
            logger.debug(f"Grouped {len(frame_results)} frame results into {len(grouped_results)} temporal groups")
            
            # Format grouped results
            for group in grouped_results:
                # Get timestamp range
                start_time = group[0][0]
                end_time = group[-1][0]
                
                # Format timestamps
                minutes_start = int(start_time // 60)
                seconds_start = int(start_time % 60)
                
                # For single frame groups, just show the start time
                if len(group) == 1 or abs(end_time - start_time) < 1.0:
                    time_marker = f"[{minutes_start:02d}:{seconds_start:02d}]"
                else:
                    # For multi-frame groups, show the time range
                    minutes_end = int(end_time // 60)
                    seconds_end = int(end_time % 60)
                    time_marker = f"[{minutes_start:02d}:{seconds_start:02d}-{minutes_end:02d}:{seconds_end:02d}]"
                
                # Combine text from all frames in the group
                combined_text = " ".join([text for _, text in group])
                
                # Add to result
                result_text += f"{time_marker} {combined_text}\n\n"
        
        # Add audio transcription if available
        if audio_text:
            result_text += "--- Audio Transcription ---\n\n"
            result_text += audio_text + "\n\n"
        
        # Add summary
        frames_with_text = len(frame_results)
        result_text += f"Processed video with {frames_with_text} frames containing text.\n"
        if audio_text:
            result_text += f"Audio transcription: {len(audio_text)} characters.\n"
        
        return result_text
    
    def _group_by_temporal_proximity(self, 
                                   results: List[Tuple[float, str]],
                                   threshold: float = 3.0) -> List[List[Tuple[float, str]]]:
        """
        Group results by temporal proximity
        
        Args:
            results: List of (timestamp, text) tuples
            threshold: Time threshold in seconds for grouping
            
        Returns:
            List of groups, where each group is a list of (timestamp, text) tuples
        """
        if not results:
            return []
        
        # Sort by timestamp
        sorted_results = sorted(results, key=lambda x: x[0])
        
        # Group results
        grouped_results = []
        current_group = [sorted_results[0]]
        
        for i in range(1, len(sorted_results)):
            current_time = sorted_results[i][0]
            prev_time = current_group[-1][0]
            
            if current_time - prev_time <= threshold:
                # Add to current group
                current_group.append(sorted_results[i])
            else:
                # Start a new group
                grouped_results.append(current_group)
                current_group = [sorted_results[i]]
        
        # Add the last group
        if current_group:
            grouped_results.append(current_group)
        
        logger.debug(f"Grouped {len(results)} results into {len(grouped_results)} temporal groups " +
                    f"with threshold {threshold}s")
        return grouped_results 
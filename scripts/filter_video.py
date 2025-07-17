#!/usr/bin/env python3
"""
Video Transcript Length Filter
Downloads videos, transcribes them using Medical Whisper, and filters for videos with sufficient transcript length.
"""

import os
import sys
import re
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Set, Optional, Dict, Union, Any
import subprocess
import time
import json
import platform

# Import required libraries
try:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    from transformers.pipelines import pipeline
    import yt_dlp
    import librosa
    import numpy as np
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Please install with: pip install torch transformers yt-dlp librosa numpy tqdm")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medical_video_filter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VideoTranscriptFilter:
    def __init__(self, video_links_file: str, output_file: Optional[str] = None):
        """
        Initialize the Video Transcript Length Filter
        
        Args:
            video_links_file: Path to file containing video links (one per line)
            output_file: Path to output filtered links (defaults to filtered_links.txt)
        """
        self.video_links_file = video_links_file
        self.output_file = output_file or "filtered_links.txt"
        self.temp_dir = tempfile.mkdtemp()
        self.processed_count = 0
        self.kept_count = 0
        
        # Medical keywords for content filtering
        self.medical_keywords = {
            'anatomy', 'physiology', 'pathology', 'diagnosis', 'treatment', 'therapy', 
            'medicine', 'medical', 'clinical', 'patient', 'disease', 'syndrome', 
            'symptom', 'pharmaceutical', 'drug', 'medication', 'prescription',
            'hospital', 'clinic', 'doctor', 'physician', 'nurse', 'healthcare',
            'surgery', 'surgical', 'operation', 'procedure', 'biopsy', 'examination',
            'blood', 'pressure', 'heart', 'cardiac', 'respiratory', 'pulmonary',
            'neurological', 'psychiatric', 'psychological', 'oncology', 'cancer',
            'infection', 'bacteria', 'virus', 'antibiotic', 'vaccine', 'immunization',
            'radiology', 'ultrasound', 'mri', 'ct scan', 'x-ray', 'imaging',
            'laboratory', 'test', 'specimen', 'sample', 'results', 'abnormal',
            'chronic', 'acute', 'inflammation', 'pain', 'relief', 'recovery',
            'rehabilitation', 'therapy', 'counseling', 'mental health'
        }
        
        # Initialize Medical Whisper model
        self.setup_whisper()
        
        # Cookie refresh settings
        self.cookies_refreshed = False
        self.max_cookie_refreshes = 3
        
    def setup_whisper(self):
        """Initialize the Medical Whisper model"""
        try:
            logger.info("Loading Medical Whisper model...")
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            model_id = "Crystalcareai/Whisper-Medicalv1"
            
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, 
                torch_dtype=self.torch_dtype, 
                low_cpu_mem_usage=True, 
                use_safetensors=True
            )
            self.model.to(self.device)
            
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
                max_new_tokens=400,
                chunk_length_s=30,
                batch_size=16
            )
            
            logger.info(f"Medical Whisper model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Medical Whisper model: {e}")
            raise
    
    def refresh_cookies_from_browser(self) -> bool:
        """
        Refresh cookies from browser using yt-dlp's built-in cookie extraction
        
        Returns:
            True if cookies were successfully refreshed
        """
        try:
            logger.info("Attempting to refresh cookies from browser...")
            
            # Determine the system and browser
            system = platform.system().lower()
            
            # Try different browsers in order of preference
            browsers = ['chrome', 'firefox', 'safari', 'edge']
            
            for browser in browsers:
                try:
                    logger.info(f"Trying to extract cookies from {browser}...")
                    
                    # Use yt-dlp's built-in cookie extraction
                    ydl_opts = {
                        'cookiesfrombrowser': (browser,),
                        'cookiefile': os.path.join(os.path.dirname(self.video_links_file), 'cookies.txt'),
                        'quiet': True,
                        'no_warnings': True,
                    }
                    
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        # Test with a simple YouTube URL to extract cookies
                        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll
                        ydl.extract_info(test_url, download=False)
                    
                    logger.info(f"Successfully refreshed cookies from {browser}")
                    return True
                    
                except Exception as e:
                    logger.debug(f"Failed to extract cookies from {browser}: {e}")
                    continue
            
            logger.warning("Failed to refresh cookies from any browser")
            return False
            
        except Exception as e:
            logger.error(f"Error refreshing cookies: {e}")
            return False
    
    def load_video_links(self) -> List[str]:
        """Load video links from file"""
        try:
            with open(self.video_links_file, 'r', encoding='utf-8') as f:
                links = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(links)} video links")
            return links
        except Exception as e:
            logger.error(f"Failed to load video links: {e}")
            raise
    
    def get_resume_index(self, video_links: List[str]) -> int:
        """
        Find the index to resume processing from based on the last video in filtered_links.txt
        
        Args:
            video_links: List of all video links
            
        Returns:
            Index to resume from (0 if no filtered links exist)
        """
        try:
            if not os.path.exists(self.output_file):
                logger.info("No existing filtered links file found, starting from beginning")
                return 0
            
            # Read the last line from filtered_links.txt
            with open(self.output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if not lines:
                    logger.info("Filtered links file is empty, starting from beginning")
                    return 0
                
                last_processed_link = lines[-1].strip()
                if not last_processed_link:
                    logger.info("Last line in filtered links file is empty, starting from beginning")
                    return 0
            
            # Find the index of this link in the original video_links.txt
            try:
                resume_index = video_links.index(last_processed_link)
                logger.info(f"Found last processed video at index {resume_index}")
                
                # Resume from the next video after the last processed one
                next_index = resume_index + 1
                if next_index >= len(video_links):
                    logger.info("All videos have been processed")
                    return len(video_links)
                
                logger.info(f"Resuming from video {next_index + 1}/{len(video_links)}: {video_links[next_index]}")
                return next_index
                
            except ValueError:
                logger.warning(f"Last processed link '{last_processed_link}' not found in original video list, starting from beginning")
                return 0
                
        except Exception as e:
            logger.error(f"Error determining resume index: {e}")
            return 0
    
    def save_filtered_links(self, filtered_links: List[str], append: bool = False):
        """
        Save filtered links to output file
        
        Args:
            filtered_links: List of links to save
            append: If True, append to existing file. If False, overwrite file.
        """
        try:
            mode = 'a' if append else 'w'
            with open(self.output_file, mode, encoding='utf-8') as f:
                for link in filtered_links:
                    f.write(f"{link}\n")
            
            if append:
                logger.info(f"Appended {len(filtered_links)} new filtered links to {self.output_file}")
            else:
                logger.info(f"Saved {len(filtered_links)} filtered links to {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to save filtered links: {e}")
            raise
    
    def download_video_audio(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Download video and extract audio
        
        Args:
            url: Video URL
            
        Returns:
            Dict with audio_path and duration, or None if failed
        """
        try:
            # Configure yt-dlp options
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(self.temp_dir, 'temp_audio.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
            }

            # Add cookies file if it exists in the same directory as video_links_file
            cookies_path = os.path.join(os.path.dirname(self.video_links_file), 'cookies.txt')
            if os.path.exists(cookies_path):
                ydl_opts['cookiefile'] = cookies_path
                logger.info(f"Using cookies file: {cookies_path}")

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info first
                info = ydl.extract_info(url, download=False)
                if info is None:
                    logger.warning(f"Could not extract info for: {url}")
                    return None
                
                duration = info.get('duration', 0)
                logger.info(f"Video duration: {duration//60} minutes {duration%60} seconds")
                
                # Try to download with WAV conversion
                ydl.download([url])
                
                # Find the downloaded audio file
                for file in os.listdir(self.temp_dir):
                    if file.startswith('temp_audio'):
                        audio_path = os.path.join(self.temp_dir, file)
                        logger.debug(f"Downloaded audio: {audio_path}")
                        return {'audio_path': audio_path, 'duration': duration}
                
                return None
                
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if this is an age verification error
            if "sign in to confirm your age" in error_str or "inappropriate for some users" in error_str:
                logger.warning(f"Age verification required for: {url}")
                
                # Try to refresh cookies if we haven't exceeded the limit
                if not self.cookies_refreshed and self.max_cookie_refreshes > 0:
                    logger.info("Attempting to refresh cookies...")
                    if self.refresh_cookies_from_browser():
                        self.cookies_refreshed = True
                        self.max_cookie_refreshes -= 1
                        logger.info("Cookies refreshed, retrying download...")
                        
                        # Retry the download with fresh cookies
                        return self.download_video_audio(url)
                    else:
                        logger.warning("Failed to refresh cookies")
                else:
                    logger.warning("Cookie refresh limit reached or already attempted")
            
            # If it's not an age verification error, try alternative audio formats
            try:
                logger.warning(f"WAV conversion failed, trying MP3: {e}")
                
                # Try with MP3 instead
                ydl_opts_mp3 = {
                    'format': 'bestaudio/best',
                    'outtmpl': os.path.join(self.temp_dir, 'temp_audio.%(ext)s'),
                    'quiet': True,
                    'no_warnings': True,
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '192',
                    }],
                }
                
                # Add cookies if available
                cookies_path = os.path.join(os.path.dirname(self.video_links_file), 'cookies.txt')
                if os.path.exists(cookies_path):
                    ydl_opts_mp3['cookiefile'] = cookies_path
                
                with yt_dlp.YoutubeDL(ydl_opts_mp3) as ydl2:
                    ydl2.download([url])
                    
                    # Find the downloaded audio file
                    for file in os.listdir(self.temp_dir):
                        if file.startswith('temp_audio'):
                            audio_path = os.path.join(self.temp_dir, file)
                            logger.debug(f"Downloaded audio (MP3): {audio_path}")
                            return {'audio_path': audio_path, 'duration': duration}
                            
            except Exception as e2:
                try:
                    logger.warning(f"MP3 conversion also failed, trying raw audio: {e2}")
                    
                    # Try without any conversion - just get the best audio format
                    ydl_opts_raw = {
                        'format': 'bestaudio/best',
                        'outtmpl': os.path.join(self.temp_dir, 'temp_audio.%(ext)s'),
                        'quiet': True,
                        'no_warnings': True,
                    }
                    
                    # Add cookies if available
                    cookies_path = os.path.join(os.path.dirname(self.video_links_file), 'cookies.txt')
                    if os.path.exists(cookies_path):
                        ydl_opts_raw['cookiefile'] = cookies_path
                    
                    with yt_dlp.YoutubeDL(ydl_opts_raw) as ydl3:
                        ydl3.download([url])
                        
                        # Find the downloaded audio file
                        for file in os.listdir(self.temp_dir):
                            if file.startswith('temp_audio'):
                                audio_path = os.path.join(self.temp_dir, file)
                                logger.debug(f"Downloaded raw audio: {audio_path}")
                                return {'audio_path': audio_path, 'duration': duration}
                                
                except Exception as e3:
                    logger.warning(f"All download methods failed: {e3}")
            
            logger.warning(f"Failed to download {url}: {e}")
            return None
    
    def transcribe_audio_chunked(self, audio_path: str, duration: int, min_words: int = 30) -> str:
        """
        Transcribe audio in chunks until minimum word count is reached
        
        Args:
            audio_path: Path to audio file
            duration: Video duration in seconds
            min_words: Minimum number of words required
            
        Returns:
            Transcribed text (accumulated from chunks)
        """
        try:
            # Load audio with librosa
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Skip very short audio (< 5 seconds)
            if len(audio) < 5 * sr:
                logger.debug("Audio too short, skipping")
                return ""
            
            accumulated_text = ""
            chunk_duration = 30  # 30 seconds per chunk
            chunk_samples = chunk_duration * sr
            
            logger.info(f"Processing audio in {chunk_duration}-second chunks")
            
            for start_sample in range(0, len(audio), int(chunk_samples)):
                end_sample = min(start_sample + chunk_samples, len(audio))
                chunk_audio = audio[start_sample:end_sample]
                
                # Skip chunks that are too short
                if len(chunk_audio) < 2 * sr:  # Less than 2 seconds
                    continue
                
                # Transcribe chunk
                result = self.pipe(chunk_audio)
                
                # Handle different result formats
                if isinstance(result, dict):
                    result_dict = result
                    text_value = result_dict.get('text')
                    if text_value is not None:
                        chunk_text = str(text_value).strip()
                    else:
                        logger.warning(f"No 'text' key in result: {result_dict}")
                        continue
                elif isinstance(result, str):
                    chunk_text = result.strip()
                else:
                    logger.warning(f"Unexpected result format: {type(result)}")
                    continue
                
                if chunk_text:
                    accumulated_text += " " + chunk_text
                    word_count = len(accumulated_text.split())
                    logger.debug(f"Chunk transcribed: {len(chunk_text)} chars, Total words: {word_count}")
                    
                    # Check if we've reached the minimum word count
                    if word_count >= min_words:
                        logger.info(f"Reached minimum word count ({min_words}) after processing chunk - stopping transcription")
                        break
            
            logger.info(f"Final transcription: {len(accumulated_text)} characters, {len(accumulated_text.split())} words")
            return accumulated_text.strip()
            
        except Exception as e:
            logger.warning(f"Failed to transcribe audio: {e}")
            return ""
    
    def has_sufficient_transcript(self, text: str, min_words: int = 30) -> bool:
        """
        Check if text has sufficient length for meaningful content
        
        Args:
            text: Transcribed text
            min_words: Minimum number of words required (default: 30)
            
        Returns:
            True if text has sufficient length
        """
        if not text:
            return False
        
        word_count = len(text.split())
        has_sufficient = word_count >= min_words
        
        logger.debug(f"Word count: {word_count}, Minimum required: {min_words}, "
                    f"Has sufficient transcript: {has_sufficient}")
        
        return has_sufficient
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            for file in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")
    
    def process_video(self, url: str) -> bool:
        """
        Process a single video
        
        Args:
            url: Video URL
            
        Returns:
            True if video has sufficient transcript length
        """
        logger.info(f"Processing: {url}")
        
        try:
            # Download video audio
            download_result = self.download_video_audio(url)
            if not download_result:
                logger.warning(f"Failed to download audio for: {url}")
                return False
            
            audio_path = download_result['audio_path']
            duration = download_result['duration']
            
            # Transcribe audio in chunks
            transcription = self.transcribe_audio_chunked(audio_path, duration)
            if not transcription:
                logger.warning(f"Failed to transcribe: {url}")
                return False
            
            # Check for sufficient transcript length
            has_sufficient_transcript = self.has_sufficient_transcript(transcription)
            
            # Log sample of transcription for debugging
            sample_text = transcription[:200] + "..." if len(transcription) > 200 else transcription
            logger.info(f"Sample transcription: {sample_text}")
            logger.info(f"Has sufficient transcript: {has_sufficient_transcript}")
            
            return has_sufficient_transcript
            
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return False
        
        finally:
            # Clean up temp files
            self.cleanup_temp_files()
    
    def run(self):
        """Main processing loop"""
        logger.info("Starting Video Transcript Length Filter")
        logger.info(f"Output file: {self.output_file}")
        
        try:
            # Load video links
            video_links = self.load_video_links()
            total_videos = len(video_links)
            
            if total_videos == 0:
                logger.warning("No video links found")
                return
            
            # Determine where to resume from
            resume_index = self.get_resume_index(video_links)
            
            if resume_index >= total_videos:
                logger.info("All videos have already been processed")
                return
            
            # Load existing filtered links if resuming
            existing_filtered_links = []
            if resume_index > 0 and os.path.exists(self.output_file):
                try:
                    with open(self.output_file, 'r', encoding='utf-8') as f:
                        existing_filtered_links = [line.strip() for line in f if line.strip()]
                    logger.info(f"Loaded {len(existing_filtered_links)} existing filtered links")
                except Exception as e:
                    logger.warning(f"Failed to load existing filtered links: {e}")
            
            filtered_links = existing_filtered_links.copy()
            
            # Update counters for resume
            self.kept_count = len(existing_filtered_links)
            self.processed_count = resume_index
            
            # Process videos starting from resume_index
            videos_to_process = video_links[resume_index:]
            logger.info(f"Processing {len(videos_to_process)} videos starting from index {resume_index}")
            
            for i, url in enumerate(tqdm(videos_to_process, desc="Processing videos"), resume_index + 1):
                logger.info(f"Processing video {i}/{total_videos}")
                
                try:
                    has_sufficient_transcript = self.process_video(url)
                    
                    if has_sufficient_transcript:
                        filtered_links.append(url)
                        self.kept_count += 1
                        logger.info(f"✓ KEPT: {url}")
                    else:
                        logger.info(f"✗ REMOVED: {url}")
                    
                    self.processed_count += 1
                    
                    # Save progress every 10 videos
                    if i % 10 == 0:
                        # Save all filtered links (overwrite to ensure consistency)
                        self.save_filtered_links(filtered_links)
                        logger.info(f"Progress saved: {self.kept_count}/{self.processed_count} videos kept")
                    
                    # Add small delay to avoid overwhelming servers
                    time.sleep(1)
                    
                except KeyboardInterrupt:
                    logger.info("Processing interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Error processing video {i}: {e}")
                    continue
            
            # Save final results
            self.save_filtered_links(filtered_links)
            
            # Print summary
            logger.info(f"\n=== PROCESSING COMPLETE ===")
            logger.info(f"Total videos processed: {self.processed_count}")
            logger.info(f"Videos with sufficient transcript: {self.kept_count}")
            logger.info(f"Videos removed: {self.processed_count - self.kept_count}")
            logger.info(f"Sufficient transcript ratio: {self.kept_count/max(self.processed_count, 1)*100:.1f}%")
            logger.info(f"Filtered links saved to: {self.output_file}")
            
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise
        
        finally:
            # Cleanup
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python filter_video.py <video_links_file> [output_file]")
        print("Example: python filter_video.py my_videos.txt filtered_videos.txt")
        sys.exit(1)
    
    video_links_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(video_links_file):
        print(f"Error: File {video_links_file} not found")
        sys.exit(1)
    
    try:
        filter_app = VideoTranscriptFilter(video_links_file, output_file)
        filter_app.run()
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
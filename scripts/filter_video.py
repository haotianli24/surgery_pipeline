#!/usr/bin/env python3
"""
Medical Video Content Filter
Downloads videos, transcribes them using Medical Whisper, and filters for medical content.
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

class MedicalVideoFilter:
    def __init__(self, video_links_file: str, output_file: Optional[str] = None):
        """
        Initialize the Medical Video Filter
        
        Args:
            video_links_file: Path to file containing video links (one per line)
            output_file: Path to output filtered links (defaults to input file)
        """
        self.video_links_file = video_links_file
        self.output_file = output_file or video_links_file
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
    
    def save_filtered_links(self, filtered_links: List[str]):
        """Save filtered links back to file"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for link in filtered_links:
                    f.write(f"{link}\n")
            logger.info(f"Saved {len(filtered_links)} filtered links to {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to save filtered links: {e}")
            raise
    
    def download_video_audio(self, url: str) -> Optional[str]:
        """
        Download video and extract audio
        
        Args:
            url: Video URL
            
        Returns:
            Path to audio file or None if failed
        """
        try:
            # Configure yt-dlp options
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(self.temp_dir, 'temp_audio.%(ext)s'),
                'extractaudio': True,
                'audioformat': 'wav',
                'audioquality': 5,
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
                
                # Skip very long videos (>30 minutes) to save time
                if duration > 1800:
                    logger.info(f"Skipping long video ({duration//60} minutes): {url}")
                    return None
                
                # Download audio
                ydl.download([url])
                
                # Find the downloaded audio file
                for file in os.listdir(self.temp_dir):
                    if file.startswith('temp_audio'):
                        audio_path = os.path.join(self.temp_dir, file)
                        logger.debug(f"Downloaded audio: {audio_path}")
                        return audio_path
                
                return None
                
        except Exception as e:
            logger.warning(f"Failed to download {url}: {e}")
            return None
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio using Medical Whisper
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            # Load audio with librosa
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Skip very short audio (< 5 seconds)
            if len(audio) < 5 * sr:
                logger.debug("Audio too short, skipping")
                return ""
            
            # Transcribe using the pipeline
            result = self.pipe(audio)
            
            # Handle different result formats
            if isinstance(result, dict):
                result_dict = result
                text_value = result_dict.get('text')
                if text_value is not None:
                    transcription = str(text_value).strip()
                else:
                    logger.warning(f"No 'text' key in result: {result_dict}")
                    return ""
            elif isinstance(result, str):
                transcription = result.strip()
            else:
                logger.warning(f"Unexpected result format: {type(result)}")
                return ""
            
            logger.debug(f"Transcribed {len(transcription)} characters")
            return transcription
            
        except Exception as e:
            logger.warning(f"Failed to transcribe audio: {e}")
            return ""
    
    def contains_medical_content(self, text: str) -> bool:
        """
        Check if text contains medical content
        
        Args:
            text: Transcribed text
            
        Returns:
            True if contains medical content
        """
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Count medical keywords
        medical_count = 0
        total_words = len(text.split())
        
        for keyword in self.medical_keywords:
            if keyword in text_lower:
                medical_count += text_lower.count(keyword)
        
        # Consider medical if:
        # 1. At least 3 different medical terms found, OR
        # 2. Medical terms make up at least 2% of total words
        medical_density = medical_count / max(total_words, 1)
        
        is_medical = medical_count >= 3 or medical_density >= 0.02
        
        logger.debug(f"Medical keywords found: {medical_count}, Density: {medical_density:.3f}, "
                    f"Is medical: {is_medical}")
        
        return is_medical
    
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
            True if video contains medical content
        """
        logger.info(f"Processing: {url}")
        
        try:
            # Download video audio
            audio_path = self.download_video_audio(url)
            if not audio_path:
                logger.warning(f"Failed to download audio for: {url}")
                return False
            
            # Transcribe audio
            transcription = self.transcribe_audio(audio_path)
            if not transcription:
                logger.warning(f"Failed to transcribe: {url}")
                return False
            
            # Check for medical content
            has_medical_content = self.contains_medical_content(transcription)
            
            # Log sample of transcription for debugging
            sample_text = transcription[:200] + "..." if len(transcription) > 200 else transcription
            logger.info(f"Sample transcription: {sample_text}")
            logger.info(f"Has medical content: {has_medical_content}")
            
            return has_medical_content
            
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return False
        
        finally:
            # Clean up temp files
            self.cleanup_temp_files()
    
    def run(self):
        """Main processing loop"""
        logger.info("Starting Medical Video Filter")
        
        try:
            # Load video links
            video_links = self.load_video_links()
            total_videos = len(video_links)
            
            if total_videos == 0:
                logger.warning("No video links found")
                return
            
            filtered_links = []
            
            # Process each video
            for i, url in enumerate(tqdm(video_links, desc="Processing videos"), 1):
                logger.info(f"Processing video {i}/{total_videos}")
                
                try:
                    has_medical_content = self.process_video(url)
                    
                    if has_medical_content:
                        filtered_links.append(url)
                        self.kept_count += 1
                        logger.info(f"✓ KEPT: {url}")
                    else:
                        logger.info(f"✗ REMOVED: {url}")
                    
                    self.processed_count += 1
                    
                    # Save progress every 10 videos
                    if i % 10 == 0:
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
            logger.info(f"Videos with medical content: {self.kept_count}")
            logger.info(f"Videos removed: {self.processed_count - self.kept_count}")
            logger.info(f"Medical content ratio: {self.kept_count/max(self.processed_count, 1)*100:.1f}%")
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
        print("Usage: python medical_video_filter.py <video_links_file> [output_file]")
        print("Example: python medical_video_filter.py my_videos.txt filtered_videos.txt")
        sys.exit(1)
    
    video_links_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else video_links_file
    
    if not os.path.exists(video_links_file):
        print(f"Error: File {video_links_file} not found")
        sys.exit(1)
    
    try:
        filter_app = MedicalVideoFilter(video_links_file, output_file)
        filter_app.run()
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
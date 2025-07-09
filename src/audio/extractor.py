import subprocess
from pathlib import Path
from typing import Optional
import logging
import math

class AudioExtractor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def extract_audio(self, video_path: Path, output_path: Optional[Path] = None) -> Path:
        """Extract audio from video using ffmpeg
        Args:
            video_path: Path to the video file
            output_path: Path to save the extracted audio file
        Returns:
            Path to the extracted audio file
        """
        if output_path is None:
            # Generate default output path
            output_path = Path(self.config.temp_dir) / f"{video_path.stem}.{self.config.audio_format}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        command = [
            'ffmpeg',
            '-i', str(video_path),
            '-q:a', '2',  # Lower quality to reduce file size
            '-ar', '16000',  # Reduce sample rate to 16kHz
            '-ac', '1',  # Convert to mono
            '-map', 'a',  # Select audio stream
            str(output_path)
        ]
        try: 
            subprocess.run(command, check=True)
            print(f"Audio extracted to {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to extract audio: {e}")
            raise e
        
    def get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds using ffprobe"""
        command = [
            'ffprobe',
            '-v', 'quiet',
            '-show_entries', 'format=duration',
            '-of', 'csv=p=0',
            str(audio_path)
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError) as e:
            self.logger.error(f"Failed to get audio duration: {e}")
            return 0.0
        
    def chunk_audio_if_needed(self, audio_path: Path) -> list[Path]:
        """Split large audio files to meet API limits"""
        file_size = audio_path.stat().st_size
        max_size = self.config.max_file_size_mb * 1024 * 1024  # Convert MB to bytes
        
        if file_size <= max_size:
            return [audio_path]
        
        # File is too large, need to chunk it
        self.logger.info(f"Audio file {file_size / (1024*1024):.1f}MB exceeds {self.config.max_file_size_mb}MB limit, chunking...")
        
        # Get audio duration
        duration = self.get_audio_duration(audio_path)
        if duration == 0:
            self.logger.error("Could not determine audio duration, cannot chunk")
            return [audio_path]
        
        # Use a more conservative chunk size (20MB instead of 25MB to be safe)
        safe_max_size = 20 * 1024 * 1024  # 20MB in bytes
        
        # Calculate chunk duration based on file size ratio
        # Assuming linear relationship between duration and file size
        chunk_duration = (safe_max_size / file_size) * duration * 0.8  # 80% to be extra safe
        
        # Create chunks directory
        chunks_dir = Path(self.config.temp_dir) / f"{audio_path.stem}_chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        
        chunks = []
        num_chunks = math.ceil(duration / chunk_duration)
        
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = min((i + 1) * chunk_duration, duration)
            
            chunk_path = chunks_dir / f"chunk_{i:03d}.{self.config.audio_format}"
            
            command = [
                'ffmpeg',
                '-i', str(audio_path),
                '-ss', str(start_time),
                '-to', str(end_time),
                '-c', 'copy',  # Copy without re-encoding for speed
                str(chunk_path)
            ]
            
            try:
                subprocess.run(command, check=True, capture_output=True)
                chunks.append(chunk_path)
                self.logger.info(f"Created chunk {i+1}/{num_chunks}: {chunk_path}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to create chunk {i}: {e}")
                # If chunking fails, return original file
                return [audio_path]
        
        return chunks
import openai
from pathlib import Path
from typing import Dict, Any
import logging
import time

class WhisperClient:
    def __init__(self, config):
        self.config = config
        self.client = openai.OpenAI(api_key=config.openai_api_key)
        self.logger = logging.getLogger(__name__)
    
    def transcribe_audio(self, audio_path: Path, **kwargs) -> Dict[str, Any]:
        """Transcribe audio file using Whisper API"""
        start_time = time.time()
        try:
            with open(audio_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                    **kwargs
                )
            duration = time.time() - start_time
            self.logger.info(f"Transcription succeeded for {audio_path} in {duration:.2f} seconds.")
            
            # Convert response objects to dictionaries for JSON serialization
            segments = []
            if hasattr(response, 'segments') and response.segments:
                for segment in response.segments:
                    segments.append({
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text
                    })
            
            words = []
            if hasattr(response, 'words') and response.words:
                for word in response.words:
                    words.append({
                        'start': word.start,
                        'end': word.end,
                        'word': word.word
                    })
            
            return {
                "text": response.text,
                "segments": segments,
                "words": words
            }
        except TimeoutError as e:
            duration = time.time() - start_time
            self.logger.error(f"API TIMEOUT: Transcription timed out for {audio_path} after {duration:.2f} seconds: {e}")
            raise
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Transcription failed for {audio_path} after {duration:.2f} seconds: {e}")
            raise
    
    def transcribe_chunks(self, audio_chunks: list[Path]) -> Dict[str, Any]:
        """Transcribe multiple audio chunks and merge results"""
        if not audio_chunks:
            return {"text": "", "segments": [], "words": []}
        
        if len(audio_chunks) == 1:
            return self.transcribe_audio(audio_chunks[0])
        
        self.logger.info(f"Transcribing {len(audio_chunks)} audio chunks...")
        
        all_texts = []
        all_segments = []
        all_words = []
        chunk_duration = 0  # Track cumulative duration for timestamp offsets
        
        for i, chunk_path in enumerate(audio_chunks):
            try:
                self.logger.info(f"Transcribing chunk {i+1}/{len(audio_chunks)}: {chunk_path.name}")
                result = self.transcribe_audio(chunk_path)
                all_texts.append(result["text"])
                
                # Adjust timestamps for segments and words based on chunk position
                if result.get("segments"):
                    for segment in result["segments"]:
                        segment["start"] += chunk_duration
                        segment["end"] += chunk_duration
                    all_segments.extend(result["segments"])
                
                if result.get("words"):
                    for word in result["words"]:
                        word["start"] += chunk_duration
                        word["end"] += chunk_duration
                    all_words.extend(result["words"])
                
                # Estimate chunk duration (assuming 10 minutes per chunk if not specified)
                chunk_duration += 600  # 10 minutes in seconds
                
            except TimeoutError as e:
                self.logger.error(f"API TIMEOUT: Failed to transcribe chunk {i+1} ({chunk_path.name}) due to timeout: {e}")
                all_texts.append(f"[ERROR: Timeout on chunk {i+1}]")
                chunk_duration += 600  # Still increment for timestamp consistency
            except Exception as e:
                self.logger.error(f"Failed to transcribe chunk {i+1}: {e}")
                all_texts.append(f"[ERROR: Failed to transcribe chunk {i+1}]")
                chunk_duration += 600  # Still increment for timestamp consistency
        
        # Combine all transcripts with proper spacing
        combined_text = " ".join(all_texts)
        
        return {
            "text": combined_text,
            "segments": all_segments,
            "words": all_words,
            "chunk_count": len(audio_chunks),
            "chunks_transcribed": len([t for t in all_texts if not t.startswith("[ERROR")])
        }
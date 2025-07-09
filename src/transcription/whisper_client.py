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
                    **kwargs
                )
            duration = time.time() - start_time
            self.logger.info(f"Transcription succeeded for {audio_path} in {duration:.2f} seconds.")
            return {"text": response.text}
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
            return {"text": ""}
        
        if len(audio_chunks) == 1:
            return self.transcribe_audio(audio_chunks[0])
        
        self.logger.info(f"Transcribing {len(audio_chunks)} audio chunks...")
        
        all_texts = []
        for i, chunk_path in enumerate(audio_chunks):
            try:
                self.logger.info(f"Transcribing chunk {i+1}/{len(audio_chunks)}: {chunk_path.name}")
                result = self.transcribe_audio(chunk_path)
                all_texts.append(result["text"])
            except TimeoutError as e:
                self.logger.error(f"API TIMEOUT: Failed to transcribe chunk {i+1} ({chunk_path.name}) due to timeout: {e}")
                all_texts.append(f"[ERROR: Timeout on chunk {i+1}]")
            except Exception as e:
                self.logger.error(f"Failed to transcribe chunk {i+1}: {e}")
                all_texts.append(f"[ERROR: Failed to transcribe chunk {i+1}]")
        
        # Combine all transcripts with proper spacing
        combined_text = " ".join(all_texts)
        
        return {
            "text": combined_text,
            "chunk_count": len(audio_chunks),
            "chunks_transcribed": len([t for t in all_texts if not t.startswith("[ERROR")])
        }
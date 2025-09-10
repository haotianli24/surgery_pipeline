from dotenv import load_dotenv
load_dotenv()
import os
print("OPENAI_API_KEY loaded:", os.getenv("OPENAI_API_KEY"))
import sys
from pathlib import Path
import logging
from typing import Dict, Any
import time
from src.settings.config import Config
from src.audio.extractor import AudioExtractor
from src.transcription.whisper_client import WhisperClient
from src.analysis.quality_checker import TranscriptQualityChecker
import json



class SurgeryTranscriptionPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.audio_extractor = AudioExtractor(config)
        self.whisper_client = WhisperClient(config)
        self.quality_checker = TranscriptQualityChecker(config)
        self.logger = logging.getLogger(__name__)
    
    def process_video(self, video_path: Path) -> Dict[str, Any]:
        """Main pipeline method to process a single video"""
        start_time = time.time()
        timing_info = {}
        
        # 1. Extract audio from video
        audio_start = time.time()
        audio_path = self.audio_extractor.extract_audio(video_path)
        audio_time = time.time() - audio_start
        timing_info['audio_extraction'] = audio_time
        print(f"Audio extraction completed in {audio_time:.2f} seconds")
        
        # 2. Check if audio needs chunking
        chunk_start = time.time()
        audio_chunks = self.audio_extractor.chunk_audio_if_needed(audio_path)
        chunk_time = time.time() - chunk_start
        timing_info['audio_chunking'] = chunk_time
        print(f"Audio chunking completed in {chunk_time:.2f} seconds ({len(audio_chunks)} chunks)")
        
        # 3. Transcribe audio (handle chunks if needed)
        transcription_start = time.time()
        if len(audio_chunks) == 1:
            transcript = self.whisper_client.transcribe_audio(audio_chunks[0])
        else:
            transcript = self.whisper_client.transcribe_chunks(audio_chunks)
        transcription_time = time.time() - transcription_start
        timing_info['transcription'] = transcription_time
        print(f"Transcription completed in {transcription_time:.2f} seconds")
        
        # 4. Quality assessment
        quality_start = time.time()
        quality_report = self.quality_checker.assess_quality(transcript)
        quality_time = time.time() - quality_start
        timing_info['quality_assessment'] = quality_time
        print(f"Quality assessment completed in {quality_time:.2f} seconds")
        
        # Save transcript text
        save_start = time.time()
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save plain text transcript
        transcript_path = output_dir / f"{video_path.stem}_transcript.txt"
        with open(transcript_path, "w") as f:
            f.write(transcript["text"] if isinstance(transcript, dict) and "text" in transcript else str(transcript))

        # Save transcript with timestamps for each sentence
        try:
            import re
            # Get video duration (from audio extractor or ffprobe)
            duration = None
            try:
                duration = self.audio_extractor.get_audio_duration(audio_path)
            except Exception:
                pass
            if not duration:
                # fallback: try to get from transcript segments
                if isinstance(transcript, dict) and transcript.get("segments") and len(transcript["segments"]):
                    duration = transcript["segments"][-1]["end"]
            if not duration:
                duration = 60  # fallback default
            
            # Get transcript text
            text = transcript["text"] if isinstance(transcript, dict) and "text" in transcript else str(transcript)
            
            # Split into sentences
            sentences = re.split(r'(?<=[.!?]) +', text)
            sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty sentences
            
            # Assign timestamps to each sentence
            if sentences:
                interval = duration / len(sentences)
                timestamped_lines = []
                for i, sentence in enumerate(sentences):
                    ts_sec = int(i * interval)
                    ts_min = ts_sec // 60
                    ts_rem = ts_sec % 60
                    timestamped_lines.append(f"[{ts_min}:{ts_rem:02d}] {sentence}")
                
                timestamped_path = output_dir / f"{video_path.stem}_transcript_timestamped.txt"
                with open(timestamped_path, "w") as f:
                    f.write("\n".join(timestamped_lines))
            else:
                print("Warning: No sentences found in transcript")
                
        except Exception as e:
            print(f"Warning: Failed to save timestamped transcript: {e}")
        
        # Save detailed transcript with timestamps (JSON)
        if isinstance(transcript, dict) and "segments" in transcript:
            detailed_transcript_path = output_dir / f"{video_path.stem}_transcript_detailed.json"
            with open(detailed_transcript_path, "w") as f:
                json.dump(transcript, f, indent=2)
            
            # Save VTT format for video players
            vtt_path = output_dir / f"{video_path.stem}_transcript.vtt"
            with open(vtt_path, "w") as f:
                f.write("WEBVTT\n\n")
                for i, segment in enumerate(transcript.get("segments", [])):
                    start_time_str = self._format_timestamp(segment.get("start", 0))
                    end_time_str = self._format_timestamp(segment.get("end", 0))
                    text = segment.get("text", "").strip()
                    f.write(f"{i+1}\n")
                    f.write(f"{start_time_str} --> {end_time_str}\n")
                    f.write(f"{text}\n\n")
        
        # Save quality report as JSON
        quality_report_path = output_dir / f"{video_path.stem}_quality_report.json"
        with open(quality_report_path, "w") as f:
            json.dump(quality_report, f, indent=2)
        
        # Save timing report
        timing_report_path = output_dir / f"{video_path.stem}_timing_report.json"
        total_time = time.time() - start_time
        timing_info['total_time'] = total_time
        timing_info['video_file'] = str(video_path)
        timing_info['video_size_mb'] = video_path.stat().st_size / (1024 * 1024)
        with open(timing_report_path, "w") as f:
            json.dump(timing_info, f, indent=2)
        
        save_time = time.time() - save_start
        timing_info['file_saving'] = save_time
        
        report = self.generate_report(quality_report, audio_path, transcript_path, quality_report_path, timing_info)

        # --- Cleanup: Delete audio, and chunk files (keep original video) ---
        # Note: Original video file is preserved
        print(f"Preserving original video file: {video_path}")
        
        # try:
        #     # Delete original video file
        #     if video_path.exists():
        #         video_path.unlink()
        #         print(f"Deleted video file: {video_path}")
        # except Exception as e:
        #     print(f"Warning: Failed to delete video file {video_path}: {e}")

        try:
            # Delete extracted audio file (if not chunked, or if chunked, will be deleted below)
            if audio_path.exists():
                audio_path.unlink()
                print(f"Deleted audio file: {audio_path}")
        except Exception as e:
            print(f"Warning: Failed to delete audio file {audio_path}: {e}")

        # Delete audio chunks and chunk directory if chunking was used
        if len(audio_chunks) > 1:
            chunk_dir = audio_chunks[0].parent
            for chunk in audio_chunks:
                try:
                    if chunk.exists():
                        chunk.unlink()
                        print(f"Deleted audio chunk: {chunk}")
                except Exception as e:
                    print(f"Warning: Failed to delete audio chunk {chunk}: {e}")
            # Try to remove the chunk directory if empty
            try:
                if chunk_dir.exists() and not any(chunk_dir.iterdir()):
                    chunk_dir.rmdir()
                    print(f"Deleted chunk directory: {chunk_dir}")
            except Exception as e:
                print(f"Warning: Failed to delete chunk directory {chunk_dir}: {e}")

        return report
        
    def generate_report(self, results: Dict[str, Any], output_path: Path, transcript_path: Path, quality_report_path: Path, timing_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive processing report"""
        return {
            "quality_report": results,
            "timing_info": timing_info,
            "status": "completed",
            "output_path": str(output_path),
            "transcript_path": str(transcript_path),
            "quality_report_path": str(quality_report_path)
        }

    def _format_timestamp(self, seconds: float) -> str:
        """Helper to format seconds into HH:MM:SS.ms"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_part = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}.{milliseconds:03d}"

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_pipeline.py <video_path>")
        sys.exit(1)


    config = Config.from_env()
    logging.basicConfig(level=logging.INFO)


    
    pipeline = SurgeryTranscriptionPipeline(config)
    video_path = Path(sys.argv[1])
    
    print(f"\n{'='*60}")
    print(f"Processing video: {video_path.name}")
    print(f"Video size: {video_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    results = pipeline.process_video(video_path)
    total_time = time.time() - start_time


    #send transcript to LLM to reformat into steps

    #step 1: grab transcript from folder
    #step 2: send text to LLM with a prompt like this
    # "Break this transcript into numbered procedural steps,
    # using provided timestamps and section headings.
    # For each step, include relevant transcript excerpts and timestamps."

    # step 3: get the feedback transcript from llm and format it into a folder called results
    # it should have: youtube link as first line, then steps with transcript for the following lines

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Results saved to: {results['output_path']}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
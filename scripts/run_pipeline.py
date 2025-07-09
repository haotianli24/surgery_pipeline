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
        transcript_path = output_dir / f"{video_path.stem}_transcript.txt"
        with open(transcript_path, "w") as f:
            f.write(transcript["text"] if isinstance(transcript, dict) and "text" in transcript else str(transcript))
        
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
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Results saved to: {results['output_path']}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
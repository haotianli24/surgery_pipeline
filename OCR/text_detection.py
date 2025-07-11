import cv2
import pytesseract
import os
import pandas as pd
from collections import defaultdict
import numpy as np

# Configure Tesseract path (adjust for your system)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Linux/Mac

def preprocess_frame(frame):
    """Preprocess frame to improve OCR accuracy"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Optional: Apply morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def detect_text_in_frame(frame, min_confidence=30):
    """Detect text in a single frame"""
    # Preprocess frame
    processed_frame = preprocess_frame(frame)
    
    # Use Tesseract to detect text with confidence scores
    data = pytesseract.image_to_data(processed_frame, output_type=pytesseract.Output.DICT)
    
    # Filter out low-confidence detections
    confidences = [int(conf) for conf in data['conf'] if int(conf) > min_confidence]
    texts = [data['text'][i] for i, conf in enumerate(data['conf']) if int(conf) > min_confidence and data['text'][i].strip()]
    
    return texts, confidences

def analyze_video_for_text(video_path, sample_interval=30, min_confidence=30):
    """
    Analyze video for text overlays
    
    Args:
        video_path: Path to video file
        sample_interval: Seconds between frame samples
        min_confidence: Minimum OCR confidence threshold
    
    Returns:
        dict with analysis results
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {"error": f"Could not open video: {video_path}"}
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Calculate frame sampling
    frame_interval = int(fps * sample_interval)
    
    results = {
        "video_path": video_path,
        "duration": duration,
        "has_text": False,
        "text_frames": 0,
        "total_sampled_frames": 0,
        "detected_texts": [],
        "avg_confidence": 0,
        "text_density": 0
    }
    
    frame_count = 0
    text_frame_count = 0
    all_confidences = []
    all_texts = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Sample frames at specified interval
        if frame_count % frame_interval == 0:
            texts, confidences = detect_text_in_frame(frame, min_confidence)
            
            results["total_sampled_frames"] += 1
            
            if texts:
                text_frame_count += 1
                all_texts.extend(texts)
                all_confidences.extend(confidences)
                
                # Store some sample texts (limit to avoid memory issues)
                if len(results["detected_texts"]) < 20:
                    results["detected_texts"].extend(texts[:5])  # Max 5 texts per frame
        
        frame_count += 1
    
    cap.release()
    
    # Calculate final metrics
    if results["total_sampled_frames"] > 0:
        results["text_frames"] = text_frame_count
        results["text_density"] = text_frame_count / results["total_sampled_frames"]
        results["has_text"] = text_frame_count > 0
        
        if all_confidences:
            results["avg_confidence"] = sum(all_confidences) / len(all_confidences)
    
    return results

def batch_analyze_videos(video_directory, output_csv="video_text_analysis.csv", 
                        text_threshold=0.1, sample_interval=30):
    """
    Analyze multiple videos for text content
    
    Args:
        video_directory: Directory containing video files
        output_csv: Output CSV file path
        text_threshold: Minimum text density to classify as "has text"
        sample_interval: Seconds between frame samples
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_files = []
    
    # Find all video files
    for ext in video_extensions:
        video_files.extend([f for f in os.listdir(video_directory) if f.lower().endswith(ext)])
    
    print(f"Found {len(video_files)} video files")
    
    results = []
    
    for i, video_file in enumerate(video_files):
        video_path = os.path.join(video_directory, video_file)
        print(f"Processing {i+1}/{len(video_files)}: {video_file}")
        
        try:
            result = analyze_video_for_text(video_path, sample_interval)
            
            # Ensure text_density is a float for comparison
            try:
                density = float(result.get("text_density", 0))
            except (ValueError, TypeError):
                density = 0.0
            result["has_significant_text"] = str(bool(density > text_threshold))
            result["filename"] = video_file
            
            results.append(result)
            
            # Print progress
            if result["has_significant_text"]:
                print(f"  ✓ Text detected (density: {result['text_density']:.2f})")
            else:
                print(f"  - No significant text (density: {result['text_density']:.2f})")
                
        except Exception as e:
            print(f"  ✗ Error processing {video_file}: {str(e)}")
            results.append({
                "filename": video_file,
                "video_path": video_path,
                "error": str(e),
                "has_significant_text": False
            })
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    
    # Print summary
    text_videos = df[df['has_significant_text'] == True]
    print(f"\nSummary:")
    print(f"Total videos analyzed: {len(df)}")
    print(f"Videos with text overlays: {len(text_videos)}")
    print(f"Percentage with text: {len(text_videos)/len(df)*100:.1f}%")
    
    return df

def separate_text_videos(analysis_csv, output_dir="separated_lists"):
    """
    Separate videos with text from those without
    
    Args:
        analysis_csv: CSV file from batch_analyze_videos
        output_dir: Directory to save separated lists
    """
    df = pd.read_csv(analysis_csv)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Videos with text
    text_videos = df[df['has_significant_text'] == True]
    if not text_videos.empty:
        text_videos.loc[:, ['filename', 'video_path', 'text_density', 'avg_confidence']].to_csv(
            os.path.join(output_dir, 'videos_with_text.csv'), index=False
        )
    # Videos without text
    no_text_videos = df[df['has_significant_text'] == False]
    if not no_text_videos.empty:
        no_text_videos.loc[:, ['filename', 'video_path']].to_csv(
            os.path.join(output_dir, 'videos_without_text.csv'), index=False
        )
    
    print(f"Separated lists saved to {output_dir}/")
    print(f"Videos with text: {len(text_videos)}")
    print(f"Videos without text: {len(no_text_videos)}")

# Example usage:
if __name__ == "__main__":
    # Single video analysis
    # result = analyze_video_for_text("path/to/your/video.mp4")
    # print(result)
    
    # Batch analysis
    video_directory = "../data/videos"  # Update this path
    df = batch_analyze_videos(
        video_directory=video_directory,
        output_csv="surgery_video_analysis.csv",
        text_threshold=0.1,  # Adjust based on your needs
        sample_interval=30   # Sample every 30 seconds
    )
    
    # Separate videos
    separate_text_videos("surgery_video_analysis.csv")
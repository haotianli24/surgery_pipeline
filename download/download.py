# Download videos from a list of youtube video IDs to data/videos

import os
import yt_dlp
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def load_video_ids(filename="videoid.txt"):
    """Load video IDs from a text file, one per line"""
    try:
        with open(filename, 'r') as f:
            video_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(video_ids)} video IDs from {filename}")
        return video_ids
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return []
    except Exception as e:
        print(f"Error loading video IDs: {e}")
        return []

def save_remaining_video_ids(video_ids, filename="videoid.txt"):
    """Save remaining video IDs back to file"""
    try:
        with open(filename, 'w') as f:
            for video_id in video_ids:
                f.write(f"{video_id}\n")
        print(f"Saved {len(video_ids)} remaining video IDs to {filename}")
    except Exception as e:
        print(f"Error saving remaining video IDs: {e}")

def download_single_video(video_id, output_dir, ydl_opts, progress_lock, remaining_video_ids, original_filename):
    """Download a single video (thread-safe)"""
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        
        # Thread-safe operations
        with progress_lock:
            remaining_video_ids.remove(video_id)
            print(f"✓ Successfully downloaded: {video_id}")
            return True, video_id, None
            
    except Exception as e:
        print(f"✗ Failed to download {video_id}: {e}")
        return False, video_id, str(e)

def download_videos(video_ids, output_dir="../data/videos", original_filename="videoid.txt", max_workers=4):
    """Download videos from YouTube video IDs using multithreading"""
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure yt-dlp options
    ydl_opts = {
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'cookiefile': 'cookies.txt' if os.path.exists('cookies.txt') else None,
        'ffmpeg_location': '/usr/bin',  # Use system ffmpeg
        'writeinfojson': False,  # Don't save video metadata
        'writesubtitles': False,  # Don't download subtitles
        'ignoreerrors': True,  # Continue on errors
    }
    
    # Remove cookiefile if it doesn't exist
    if not ydl_opts['cookiefile'] or not os.path.exists(ydl_opts['cookiefile']):
        ydl_opts.pop('cookiefile', None)
        print("No cookies.txt found, downloading without authentication")
    
    success_count = 0
    error_count = 0
    remaining_video_ids = video_ids.copy()  # Work with a copy
    progress_lock = threading.Lock()  # Thread safety for shared variables
    
    print(f"Starting download of {len(video_ids)} videos with {max_workers} threads...")
    start_time = time.time()
    
    # Use ThreadPoolExecutor for concurrent downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_video = {
            executor.submit(
                download_single_video, 
                video_id, 
                output_dir, 
                ydl_opts, 
                progress_lock, 
                remaining_video_ids, 
                original_filename
            ): video_id for video_id in video_ids
        }
        
        # Process completed downloads
        for future in as_completed(future_to_video):
            video_id = future_to_video[future]
            try:
                success, vid_id, error = future.result()
                if success:
                    success_count += 1
                    # Save progress every 10 successful downloads
                    if success_count % 10 == 0:
                        with progress_lock:
                            save_remaining_video_ids(remaining_video_ids, original_filename)
                else:
                    error_count += 1
                    
            except Exception as e:
                error_count += 1
                print(f"✗ Exception in thread for {video_id}: {e}")
    
    # Save final remaining video IDs
    save_remaining_video_ids(remaining_video_ids, original_filename)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nDownload Summary:")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed downloads: {error_count}")
    print(f"Remaining to download: {len(remaining_video_ids)}")
    print(f"Total processed: {len(video_ids)}")
    print(f"Total time: {duration:.2f} seconds ({duration/60:.1f} minutes)")
    print(f"Average time per video: {duration/len(video_ids):.2f} seconds")
    if success_count > 0:
        print(f"Download rate: {success_count/duration:.2f} videos/second")

if __name__ == "__main__":
    # Load video IDs from file
    video_ids = load_video_ids("videoid.txt")
    
    if video_ids:
        # Configure number of threads (adjust based on your system and internet speed)
        # Recommended: 2-8 threads (more threads = faster but more resource intensive)
        max_workers = 4  # You can change this number
        
        print(f"Starting download of {len(video_ids)} videos with {max_workers} threads...")
        print("Note: More threads = faster downloads but more CPU/network usage")
        download_videos(video_ids, max_workers=max_workers)
    else:
        print("No video IDs loaded. Exiting.")
import webbrowser
import time

def manual_review_interface(video_ids):
    surgery_videos = []
    non_surgery_videos = []
    skipped_videos = []
    
    print("Manual Video Review")
    print("Commands: 's' = surgery, 'n' = not surgery, 'skip' = skip, 'quit' = exit")
    print("-" * 50)
    
    for i, video_id in enumerate(video_ids):
        url = f"https://www.youtube.com/watch?v={video_id}"
        print(f"\nVideo {i+1}/{len(video_ids)}: {video_id}")
        print(f"URL: {url}")
        
        # Open in browser
        webbrowser.open(url)
        
        while True:
            decision = input("Decision (s/n/skip/quit): ").strip().lower()
            
            if decision == 's':
                surgery_videos.append(video_id)
                print("✓ Added to surgery dataset")
                break
            elif decision == 'n':
                non_surgery_videos.append(video_id)
                print("✗ Not surgery")
                break
            elif decision == 'skip':
                skipped_videos.append(video_id)
                print("⏭ Skipped")
                break
            elif decision == 'quit':
                print(f"\nStopping review. Progress saved.")
                return surgery_videos, non_surgery_videos, skipped_videos
            else:
                print("Invalid input. Use: s/n/skip/quit")
    
    return surgery_videos, non_surgery_videos, skipped_videos

# Load video IDs from file
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

# Usage
video_ids = load_video_ids("videoid.txt")
if video_ids:
    surgery, non_surgery, skipped = manual_review_interface(video_ids)
    
    # Save results
    print(f"\nReview complete!")
    print(f"Surgery videos: {len(surgery)}")
    print(f"Non-surgery videos: {len(non_surgery)}")
    print(f"Skipped videos: {len(skipped)}")
    
    # Save surgery video IDs to file
    with open("surgery_videos.txt", "w") as f:
        for vid in surgery:
            f.write(f"{vid}\n")
    print(f"Surgery video IDs saved to surgery_videos.txt")
else:
    print("No video IDs loaded. Exiting.")
# Download videos from a list of youtube links to data/videos

import os
import yt_dlp

def download_videos(video_links):
    for link in video_links:
        ydl_opts = {
            'outtmpl': os.path.join('data/videos', '%(title)s.%(ext)s'),
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'cookiefile': 'cookies.txt'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])

if __name__ == "__main__":
    video_links = [
        "https://www.youtube.com/watch?v=E_wWpRKBy4w&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=279&pp=iAQB",
        "https://www.youtube.com/watch?v=E8qWV0GvUDM&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=321&pp=iAQB",
        "https://www.youtube.com/watch?v=pQBch2Fxnvc&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=322&pp=iAQB",
        "https://www.youtube.com/watch?v=fPQtYiWhaJw&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=323&pp=iAQB",
        "https://www.youtube.com/watch?v=uDtK3Be6P84&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=324&pp=iAQB",
        "https://www.youtube.com/watch?v=SKsrj-76n30&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=325&pp=iAQB",
        "https://www.youtube.com/watch?v=OLJvLCq1w4w&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=327&pp=iAQB",
        "https://www.youtube.com/watch?v=9QkuxaYE5oU&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=329&pp=iAQB",
        "https://www.youtube.com/watch?v=sOcnbHB8S3E&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=330&pp=iAQB",
        "https://www.youtube.com/watch?v=QeRU3oo91YQ&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=331&pp=iAQB",
        "https://www.youtube.com/watch?v=YikJOoFz06U&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=332&pp=iAQB",
        "https://www.youtube.com/watch?v=a34VC_hJJEo",
            "https://youtu.be/XHonLTLI4Ns",
            "https://youtu.be/1M3LVCNJnV8",
            "https://youtu.be/Xno9UJMwRJs",
            "https://youtu.be/XNtN91f_riE",
            "https://youtu.be/oOilwb3YQBs",
            "https://youtu.be/taygFcTAFiw",
            "https://youtu.be/XjRF1odKHrE",
            "https://youtu.be/dE9ItW9STGk",
            "https://youtu.be/fQy-WwKzwcY",
            "https://youtu.be/LucszLj9oIc",
            "https://youtu.be/4QRW9O61H4k",
            "https://youtu.be/7OHQ4LQ3WHc",
            "https://youtu.be/7leGeXpJapU",
    ]
    
    download_videos(video_links)
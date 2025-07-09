# Download videos from a list of youtube links to data/videos

import os
import yt_dlp

def download_videos(video_links):
    for link in video_links:
        ydl_opts = {
            'outtmpl': os.path.join('..', 'data', 'videos', '%(title)s.%(ext)s'),
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'cookiefile': 'cookies.txt',
            'ffmpeg_location': '/opt/homebrew/bin',  # Explicitly set ffmpeg/ffprobe location
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])

if __name__ == "__main__":
    video_links = [
        "https://www.youtube.com/watch?v=SZmjBSDZhTo&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=100&pp=iAQB",
        "https://www.youtube.com/watch?v=YkgupI0wJ_g&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=101&pp=iAQB0gcJCcEJAYcqIYzv",
        "https://www.youtube.com/watch?v=y6za2Ho2YKI&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=102&pp=iAQB",
        "https://www.youtube.com/watch?v=GehRI1pWv_Q&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=107&pp=iAQB",
        "https://www.youtube.com/watch?v=K9S9BazUF4I&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=108&pp=iAQB",
        "https://www.youtube.com/watch?v=Zv0o5AxHV_U&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=110&pp=iAQB",
        "https://www.youtube.com/watch?v=v0QShlD4pg8&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=111&pp=iAQB",
        "https://www.youtube.com/watch?v=fg5aSeV_wVM&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=115&pp=iAQB",
        "https://www.youtube.com/watch?v=IUhSOHZj5qg&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=116&t=124s&pp=iAQB",
        "https://www.youtube.com/watch?v=oFuDSx6OBLI&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=119&pp=iAQB",
        "https://www.youtube.com/watch?v=9-UI2UIrvRQ&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=120&pp=iAQB",
        "https://www.youtube.com/watch?v=SS6r0d_RTw4&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=121&pp=iAQB",
        "https://www.youtube.com/watch?v=bMuQYXIsvJk&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=122&pp=iAQB",
        "https://www.youtube.com/watch?v=w-2iuAFrFH4&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=124&pp=iAQB",
        "https://www.youtube.com/watch?v=lyTqwvQ4Tho&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=126&pp=iAQB",
        "https://www.youtube.com/watch?v=kjVyBbDZPnc&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=127&pp=iAQB",
        "https://www.youtube.com/watch?v=0OEi2r-DCMQ&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=130&pp=iAQB",
        "https://www.youtube.com/watch?v=fWK2sPzrmnA&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=132&pp=iAQB",
        "https://www.youtube.com/watch?v=fftwINkBnXs&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=133&pp=iAQB",
        "https://www.youtube.com/watch?v=kCA-qeC4T84&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=134&pp=iAQB",
        "https://www.youtube.com/watch?v=a1A4hTfP-cU&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=135&pp=iAQB",
        "https://www.youtube.com/watch?v=1yDOXO4dc4I&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=138&pp=iAQB",
        "https://www.youtube.com/watch?v=6rMsmSFSkuY&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=146&pp=iAQB0gcJCcEJAYcqIYzv",
        "https://www.youtube.com/watch?v=VJGAJ2GFrPE&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=147&pp=iAQB",
        "https://www.youtube.com/watch?v=hl6q6PDwuBU&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=148&pp=iAQB",
        "https://www.youtube.com/watch?v=ISxTyNpoNG8&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=149&pp=iAQB",
        "https://www.youtube.com/watch?v=8QfLMoczT6g&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=152&pp=iAQB",
        "https://www.youtube.com/watch?v=x1gBV-UeL8c&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=153&pp=iAQB",
        "https://www.youtube.com/watch?v=6WpCkTL35JY&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=154&pp=iAQB",
        "https://www.youtube.com/watch?v=WAtAr1Bu5Yw&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=158&pp=iAQB",
        "https://www.youtube.com/watch?v=y5CYB_gDKdA&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=159&pp=iAQB",
        "https://www.youtube.com/watch?v=YnsofXmPpH8&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=160&pp=iAQB",
        "https://www.youtube.com/watch?v=rmc6HQ-TRs4&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=161&pp=iAQB",
        "https://www.youtube.com/watch?v=t-R_blSBngY&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=162&pp=iAQB",
        "https://www.youtube.com/watch?v=n_Zh6b0tXAY&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=164&pp=iAQB",
        "https://www.youtube.com/watch?v=446QvEz7miY&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=167&pp=iAQB",
        "https://www.youtube.com/watch?v=eW_1EiLGFm0&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=168&pp=iAQB",
        "https://www.youtube.com/watch?v=l1xFv6fWSgI&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=169&pp=iAQB",
        "https://www.youtube.com/watch?v=v2zyTD3e2KM&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=172&pp=iAQB0gcJCcEJAYcqIYzv",
        "https://www.youtube.com/watch?v=7TtcCZwBLwU&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=174&pp=iAQB",
        "https://www.youtube.com/watch?v=evikvqlp8ao&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=176&pp=iAQB",
        "https://www.youtube.com/watch?v=bs3iOWE7JkM&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=178&pp=iAQB",
        "https://www.youtube.com/watch?v=O7X9oQum7pI&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=182&pp=iAQB",
        "https://www.youtube.com/watch?v=Ngw_Zuh6GDU&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=185&pp=iAQB",
        "https://www.youtube.com/watch?v=PBHj6AM2Kuo&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=191&pp=iAQB",
        "https://www.youtube.com/watch?v=6T-KRgf5HD8&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=197&pp=iAQB",
        "https://www.youtube.com/watch?v=SrAHr7mm-co&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=202&pp=iAQB",
        "https://www.youtube.com/watch?v=5o--0l9glF8&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=203&pp=iAQB",
        "https://www.youtube.com/watch?v=A2IH9HwBZtE&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=204&pp=iAQB",
        "https://www.youtube.com/watch?v=4vyx9LmJCkA&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=210&pp=iAQB",
        "https://www.youtube.com/watch?v=2uSJYa8l9VE&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=211&pp=iAQB",
        "https://www.youtube.com/watch?v=JGJzKRkrzYg&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=212&pp=iAQB",
        "https://www.youtube.com/watch?v=ql409_cBJ7c&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=217&pp=iAQB",
        "https://www.youtube.com/watch?v=OuK4hgofDjQ&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=219&pp=iAQB",
        "https://www.youtube.com/watch?v=6FRwSMG4ri8&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=223&pp=iAQB",
        "https://www.youtube.com/watch?v=rF8BOL7wNfs&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=225&pp=",
        "https://www.youtube.com/watch?v=GCz2L-hgHp0&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=226&pp=iAQB",
        "https://www.youtube.com/watch?v=-4PXsMu8d9M&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=227&pp=iAQB",
        "https://www.youtube.com/watch?v=sRcWtywbnCo&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=229&pp=iAQB",
        "https://www.youtube.com/watch?v=0SXQwJG4LDI&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=236&pp=iAQB",
        "https://www.youtube.com/watch?v=cNTfAEi1Im0&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=247&pp=iAQB",
        "https://www.youtube.com/watch?v=PF5RYg6V4Gk&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=251&pp=iAQB",
        "https://www.youtube.com/watch?v=g9F0BIXvFeU&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=252&pp=iAQB0gcJCcEJAYcqIYzv",
        "https://www.youtube.com/watch?v=Rn5dO8_-8eo&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=254&pp=iAQB",
        "https://www.youtube.com/watch?v=PdcTK8xtvik&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=263&pp=iAQB",
        "https://www.youtube.com/watch?v=G4FF-9JOV90&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=264&pp=iAQB",
        "https://www.youtube.com/watch?v=wzHmnIuVgDw&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=268&pp=iAQB",
        "https://www.youtube.com/watch?v=5Bv_8xPTgPU&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=269&pp=iAQB",
        "https://www.youtube.com/watch?v=U4mpl1W7w4M&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=270&pp=iAQB",
        "https://www.youtube.com/watch?v=tQ5eDZbXC6Q&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=271&pp=iAQB",
        "https://www.youtube.com/watch?v=AWXCpRJAN8M&list=PLZpDzANLjPtUQAWLis2Ayiln1wicQKdjw&index=275&pp=iAQB",
    ]
    
    download_videos(video_links)
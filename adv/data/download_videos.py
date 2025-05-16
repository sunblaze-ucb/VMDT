"""Use yt-dlp to download videos from YouTube. Write a function that takes in a list of Youtube video ids and downloads the videos to a specified directory. The function should return a list of paths to the downloaded videos. Make sure to include a warning about the use of this script."""

# Now write the function to download videos
import os
import subprocess
import json
from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func
from youtube_transcript_api import YouTubeTranscriptApi

def download_video(url, output_dir, start_time=None, end_time=None):
    """Download a YouTube video using yt-dlp."""
    try:
        options = {
            # save all as mp4 for consistency, video only no audio needed
            "format": "bestvideo[ext=mp4]",
            # save as id_start_end.ext
            "outtmpl": os.path.join(output_dir, f"%(id)s_{start_time}_{end_time}.%(ext)s") if start_time is not None and end_time is not None else os.path.join(output_dir, "%(id)s.%(ext)s"),
            "quiet": False,
        }
        if start_time is not None and end_time is not None:
            options = {'download_ranges': download_range_func(None, [(int(start_time), int(end_time))]), **options}
        with YoutubeDL(options) as ydl:
            ydl.download([url])
        print(f"Downloaded: {url}")
        return True
    except Exception as e:
        print(f"An error occurred while downloading '{url}': {e}")
        return False

def get_video_duration(file_path):
    """Get the duration of a video file in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f"Duration of '{file_path}': {result.stdout.strip()}")
        return float(result.stdout.strip())
    except Exception as e:
        print(f"An error occurred while retrieving video duration for '{file_path}': {e}")
        return 0

def convert_to_mp4(file_path):
    """Convert a video file to mp4 format using ffmpeg from webm to mp4."""
    try:
        output_path = file_path.replace(".webm", ".mp4")
        if os.path.exists(output_path):
            print(f"File '{output_path}' already exists")
            return output_path
        subprocess.run(["ffmpeg", "-i", file_path, output_path])
        print(f"Converted '{file_path}' to '{output_path}'")
        return output_path
    except Exception as e:
        print(f"An error occurred while converting '{file_path}' to mp4: {e}")
        return ""

def download_captions(video_id, output_dir):
    """Download captions for a YouTube video using yt-dlp."""
    # youtube-dlp -- easy but in vtt so harder to parse
    # try:
    #     options = {
    #         "writesubtitles": True,
    #         "writeautomaticsub": True,
    #         "subtitleslangs": ["en"],
    #         "skip_download": True,
    #         "outtmpl": os.path.join(output_dir, f"%(id)s.%(ext)s"),
    #         "quiet": False,
    #     }
    #     with YoutubeDL(options) as ydl:
    #         ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    #     print(f"Downloaded captions for: {video_id}")
    # except Exception as e:
    #     print(f"An error occurred while downloading captions for '{video_id}': {e}")

    # youtube-transcript-api -- nicer format
    try:
        captions = YouTubeTranscriptApi.get_transcript(video_id)
        with open(os.path.join(output_dir, f"{video_id}.json"), "w") as f:
            json.dump(captions, f)
        print(f"Downloaded captions for: {video_id} into '{output_dir}'")
    except Exception as e:
        print(f"An error occurred while downloading captions for '{video_id}': {e}")

def main():
    # first get vatex video ids from vatex/T2V_non_temporal_prompts_6000_6000s.json. This files is a list of json objects with keys including "video_id", which is of the form youtube-id_start_end, e.g., MhVg5YenAps_000001_000011. The url is then https://www.youtube.com/watch?v=MhVg5YenAps
    missing_videos_vatex = []
    with open("vatex/extracted_information_6000_cleaned.json") as f:
        vatex_data = json.load(f)
    video_ids = [("_".join(d["video_id"].split("_")[:-2]), d["video_id"].split("_")[-2], d["video_id"].split("_")[-1]) for d in vatex_data]
    output_dir = "vatex/videos"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading {len(video_ids)} videos to '{output_dir}'...")
    for video_id, start_time, end_time in video_ids:
        if os.path.exists(os.path.join(output_dir, f"{video_id}_{start_time}_{end_time}.mp4")):
            continue
        url = f"https://www.youtube.com/watch?v={video_id}"
        downloaded = download_video(url, output_dir, start_time, end_time)
        if not downloaded:
            missing_videos_vatex.append(video_id)
    with open("vatex/missing_videos.json", "w") as f:
        json.dump(missing_videos_vatex, f)

    # next get neptune video ids from neptune/neptune_mmh.json. This file is a list of json objects with keys including "video_id", which is a string, e.g., "-9QGyHPv1v0". The url is then https://www.youtube.com/watch?v=-9QGyHPv1v0. There is no start or end time so we will download the entire video.
    missing_videos_neptune = []
    long_videos_neptune = []
    video_length_neptune = {}
    max_length = 3 * 60  # set max length to 3 minutes as that's what Internvideo2 can do and ~70% of the videos in Neptune are less than 3 minutes
    with open("neptune/neptune_mmh.json") as f:
        neptune_data = json.load(f)
    video_ids = [d["video_id"] for d in neptune_data]
    output_dir = "neptune/videos"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading {len(video_ids)} videos to '{output_dir}'...")
    for video_id in video_ids:
        video_path = os.path.join(output_dir, f"{video_id}.mp4")
        if os.path.exists(video_path):
            duration = get_video_duration(video_path)
            video_length_neptune[video_id] = duration
            if duration > max_length:
                long_videos_neptune.append(video_id)
            continue
        url = f"https://www.youtube.com/watch?v={video_id}"
        downloaded = download_video(url, output_dir)
        if not downloaded:
            missing_videos_neptune.append(video_id)
        else:
            duration = get_video_duration(video_path)
            if duration > max_length:
                long_videos_neptune.append(video_id)
    with open("neptune/missing_videos.json", "w") as f:
        json.dump(missing_videos_neptune, f)
    with open("neptune/long_videos.json", "w") as f:
        json.dump(long_videos_neptune, f)
    with open("neptune/video_lengths.json", "w") as f:
        json.dump(video_length_neptune, f)

    # convert webm videos to mp4 -- iterate over all webm videos in the directory and convert them to mp4 if the mp4 version does not exist.
    for video_dir in ["vatex/videos", "neptune/videos"]:
        for file in os.listdir(video_dir):
            if file.endswith(".webm"):
                convert_to_mp4(os.path.join(video_dir, file))

def download_neptune_captions():
    with open("neptune/V2T_questions_6000_alls.json") as f:
        neptune_data = json.load(f)
    for instance in neptune_data:
        video_id = instance["video_id"]
        output_dir = f"neptune/videos/{video_id}"
        os.makedirs(output_dir, exist_ok=True)
        download_captions(video_id, output_dir)

if __name__ == "__main__":
    main()
    # video_id = "-C7DmQLUN84"
    # download_neptune_captions()
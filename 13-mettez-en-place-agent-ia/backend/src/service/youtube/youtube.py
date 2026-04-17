import os
from typing import List, Dict
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class YouTubeServiceError(Exception):
    pass


class YouTubeService:

    def __init__(self):
        self.api_key = os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            raise YouTubeServiceError("YOUTUBE_API_KEY non définie")

        self.youtube = build(
            "youtube",
            "v3",
            developerKey=self.api_key
        )

    def search_videos(self, opening: str, max_results: int = 5) -> List[Dict]:
        try:
            query = f"{opening} chess opening tutorial explanation"

            request = self.youtube.search().list(
                q=query,
                part="snippet",
                type="video",
                maxResults=max_results,
                relevanceLanguage="en"
            )

            response = request.execute()

            videos = []
            for item in response.get("items", []):
                video_id = item["id"]["videoId"]
                snippet = item["snippet"]

                videos.append({
                    "title": snippet["title"],
                    "description": snippet["description"],
                    "channel": snippet["channelTitle"],
                    "video_id": video_id,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "thumbnail": snippet["thumbnails"]["high"]["url"]
                })

            return videos

        except HttpError as e:
            raise YouTubeServiceError(f"Erreur API YouTube: {str(e)}")
from fastapi import APIRouter, HTTPException
from service.youtube.youtube import (
    YouTubeService,
    YouTubeServiceError
)

router = APIRouter(tags=["chess"])


@router.get("/videos/{opening}")
def get_videos(opening: str):

    try:
        yt = YouTubeService()
        videos = yt.search_videos(opening)

        if not videos:
            return {
                "opening": opening,
                "videos": [],
                "message": "Aucune vidéo trouvée"
            }

        return {
            "opening": opening,
            "videos": videos
        }

    except YouTubeServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))
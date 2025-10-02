from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List
from pipeline.prediction_pipeline import Recommender
from pipeline.get_anime_lists import get_anime_list
import os

app = FastAPI(title="Anime Recommendation System")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models
class RatingInput(BaseModel):
    anime_name: str
    rating: int = Field(..., ge=1, le=10)

class RecommendationRequest(BaseModel):
    ratings: List[RatingInput] = Field(..., min_items=5, max_items=20)
    n: int = Field(default=10, ge=5, le=50)

class AnimeRecommendation(BaseModel):
    anime_name: str
    genres: str
    synopsis: str
    hybrid_score: float
    probability: float

class RecommendationResponse(BaseModel):
    recommendations: List[AnimeRecommendation]

# Global variable to store anime list
anime_list_cache = None

@app.on_event("startup")
async def startup_event():
    """Load anime list on startup"""
    global anime_list_cache
    anime_list_cache = get_anime_list()
    print(f"Loaded {len(anime_list_cache)} animes")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/animes", response_model=List[str])
async def get_animes():
    if anime_list_cache is None:
        raise HTTPException(status_code=500, detail="Anime list not loaded")
    return anime_list_cache

@app.post("/api/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get anime recommendations based on user ratings"""
    try:
        user_ratings = {rating.anime_name: rating.rating for rating in request.ratings}
        recommender = Recommender(user_ratings=user_ratings, n=request.n)
        recommendations_df = recommender.recommend()
        if isinstance(recommendations_df,str) and  recommendations_df == "404":
            raise HTTPException(status_code=500, detail="Couldn't find any appropriate anime at this time")
        elif isinstance(recommendations_df,str) and recommendations_df == "403":
            raise HTTPException(status_code=500, detail="There was a problem in finding good animes at this time")
        recommendations = []
        for _, row in recommendations_df.iterrows():
            recommendations.append(AnimeRecommendation(
                anime_name=row['anime_name'],
                genres=row['genres'],
                synopsis=row.get('synopsis', 'No synopsis available.'),
                hybrid_score=float(row['hybrid_score']),
                probability=float(row['probability'])
            ))
        
        return RecommendationResponse(recommendations=recommendations)
    
    except Exception as e:
        print("Problem is here",e)
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    os.makedirs("static", exist_ok=True)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
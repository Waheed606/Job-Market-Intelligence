from fastapi import FastAPI
from pydantic import BaseModel

# pyrefly: ignore [missing-import]
from src.pipelines.pred_pipeline import CustomData, Pred_Pipelines
from src.pipelines.recomend_responsiblilties import RecommendResponsibilities

app = FastAPI(title="AI Job Market Intelligence API")

# loaded once at startup, reused across requests
predict_pipeline = Pred_Pipelines()
responsibilities_engine = RecommendResponsibilities()


class PredictionRequest(BaseModel):
    keywords: str
    experience_level: str


class PredictionResponse(BaseModel):
    predicted_role: str
    responsibilities: str


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    data = CustomData(
        keyword=payload.keywords,
        experiencelevel=payload.experience_level
    )
    pred_df = data.get_data_frame()

    result = predict_pipeline.predict(pred_df)
    predicted_role = str(result[0])

    responsibilities = responsibilities_engine.recommend(
        keyword=payload.keywords,
        experiencelevel=payload.experience_level,
        predicted_title=predicted_role
    )

    return PredictionResponse(
        predicted_role=predicted_role,
        responsibilities=str(responsibilities)
    )
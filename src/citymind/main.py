from fastapi import FastAPI

app = FastAPI(
    title="CityMind API",
    description="Autonomous Smart City Intelligence Agent",
    version="0.1.0",
)

@app.get("/")
def home():
    return {
        "project": "CityMind",
        "message": "CityMind API is running",
        "version": "0.1.0",
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "citymind-api",
    }
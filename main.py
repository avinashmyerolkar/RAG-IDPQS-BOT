from fastapi import FastAPI
from app.api import api as api_router
from app.utilities.settings import settings

app = FastAPI()

# Include API routes
app.include_router(api_router.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)

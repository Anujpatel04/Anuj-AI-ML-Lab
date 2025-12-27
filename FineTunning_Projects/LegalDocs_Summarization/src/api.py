from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import pipeline  # Import the RAG pipeline function
import uvicorns

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/summarize/")
async def summarize(request: QueryRequest):
    try:
        summaries = rag_pipeline(request.query, top_k=request.top_k)
        return {"query": request.query, "summaries": summaries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.calculator import router as calculator_router, CompensationRequest
from backend.ocr import router as ocr_router
from backend.vector_db import semantic_search, get_qdrant_client, VECTOR_DB_INITIALIZED
from backend.evaluator import evaluate_compensation_precedents

app = FastAPI(
    title="Compensation Calculator & Centralized Qdrant Vector DB",
    description="Enterprise motor claims compensation dashboard with local Qdrant database indexing, batch PDF processing, and AI Legal Precedents Assistant.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# PYDANTIC SCHEMAS FOR CHAT & EVALUATIONS
# ======================================================

class ChatRequest(BaseModel):
    message: str
    case_type: str = "all"  # 'injury', 'death', or 'all'

class EvaluateRequest(BaseModel):
    params: dict
    calculated_amount: float

# ======================================================
# API ENDPOINTS
# ======================================================

app.include_router(calculator_router, prefix="/api/calculate", tags=["Calculation"])
app.include_router(ocr_router, prefix="/api/ocr", tags=["OCR"])

@app.get("/api/health")
async def health_check():
    """Returns server connection stats, indicating if Qdrant and FastEmbed are online!"""
    client = get_qdrant_client()
    return {
        "status": "healthy",
        "message": "Compensation Calculator Server is running!",
        "vector_db": "online" if VECTOR_DB_INITIALIZED else "offline_simulation_mode",
        "local_storage_path": "./qdrant_db/"
    }

@app.post("/api/search/chat")
async def legal_ai_chat(request: ChatRequest):
    """
    Receives user query, runs a semantic vector search across 
    100+ indexed PDF documents in Qdrant, and returns matching precedents 
    and summaries.
    """
    try:
        case_filter = None if request.case_type == "all" else request.case_type
        
        # 1. Perform semantic search
        logger_results = semantic_search(request.message, limit=3, case_type_filter=case_filter)
        
        if not logger_results:
            # Fallback chat response if Qdrant is empty
            return {
                "response": "Hello! I am your AI Legal Assistant. The Qdrant centralized database is currently running in local-disk mode and is awaiting PDF document uploads to learn from precedents.\n\nOnce you drop legal petitions, judgments, or prayers in the **PDF Library**, they will be automatically indexed, and I can semantically answer specific profile questions (e.g. searching by age, income, and disability) and retrieve precedents!",
                "precedents": []
            }
            
        # 2. Compile matches into a highly professional response
        response_text = f"Based on your query **\"{request.message}\"**, I searched the centralized Qdrant vector database and retrieved the most relevant precedent cases:\n\n"
        
        for idx, match in enumerate(logger_results):
            meta = match["metadata"]
            name = meta.get("name", "Unnamed Claimant")
            filename = match["filename"]
            score = match["score"]
            award_amount = meta.get("award_amount", "")
            
            response_text += f"{idx+1}. **{name}** (Precedent file: *{filename}*, Semantic Match: {score*100:.1f}%)\n"
            response_text += f"   - *Case Parameters:* Age {meta.get('age', 'N/A')} | Income Rs. {meta.get('monthly_income', 'N/A')}/pm"
            if meta.get("case_type") == "injury":
                response_text += f" | Disability {meta.get('disability', 'N/A')}%"
            if award_amount:
                response_text += f" | **Award Amount: Rs. {int(float(award_amount)):,}**"
            response_text += f"\n   - *Key Extract:* \"...{match['text'].strip()}...\"\n\n"
            
        response_text += "\nThese matching judgments can be applied immediately as defensible courtroom precedents. Let me know if you would like me to compile the formal claim brief or run a comparative mathematical benchmarking evaluation!"
        
        return {
            "response": response_text,
            "precedents": logger_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Legal chat error: {str(e)}")

@app.post("/api/search/evaluate")
async def evaluate_precedents(request: EvaluateRequest):
    """
    Benchmarks the math engine's calculated sum against semantic matched Qdrant precedents.
    """
    try:
        evaluation = evaluate_compensation_precedents(request.params, request.calculated_amount)
        return {
            "success": True,
            "evaluation": evaluation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparative evaluation failed: {str(e)}")

# ======================================================
# STATIC FILES SERVING (MAPPED TO THE TABBED SPA)
# ======================================================

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    print(f"Warning: Frontend directory '{frontend_dir}' not found.")

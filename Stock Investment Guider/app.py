from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse,FileResponse
from fastapi.staticfiles import StaticFiles
import os
from pydantic import BaseModel
import asyncio
import json
import re
from typing import Dict, Any

# Import your agent pipeline
from agents import run_agent_pipeline

app = FastAPI(title="Stock Investment Guider API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InvestmentRequest(BaseModel):
    user_interests: str
    budget: float
    risk_tolerance: str

def parse_agent_output(output: str) -> Dict[str, str]:
    """Parse the agent output to extract strategy, allocation, and returns"""
    
    # Extract strategy
    strategy_match = re.search(r'Strategy:\s*(.*?)(?=\n|Allocations:|$)', output, re.DOTALL)
    strategy = strategy_match.group(1).strip() if strategy_match else "Strategy not found"
    
    # Extract allocations
    allocations_match = re.search(r'Allocations:(.*?)(?=Estimated Total Return:|$)', output, re.DOTALL)
    allocations = allocations_match.group(1).strip() if allocations_match else "Allocations not found"
    
    # Extract expected returns
    returns_match = re.search(r'Estimated Total Return:\s*(.*?)(?=\n|$)', output, re.DOTALL)
    expected_return = returns_match.group(1).strip() if returns_match else "Returns not found"
    
    return {
        "strategy": strategy,
        "allocation": allocations,
        "expected_return": expected_return
    }

async def stream_text_simulation(text: str, chunk_size: int = 5, delay: float = 0.1):
    """Simulate streaming text by yielding chunks"""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if i + chunk_size < len(words):
            chunk += " "
        yield chunk
        await asyncio.sleep(delay)

async def generate_investment_stream(user_interests: str, budget: float, risk_tolerance: str):
    """Generator function that streams the investment analysis"""
    
    try:
        # Send initial status
        yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing your investment preferences...'})}\n\n"
        await asyncio.sleep(1)
        
        # Run the agent pipeline
        yield f"data: {json.dumps({'type': 'status', 'message': 'Searching for suitable stocks...'})}\n\n"
        await asyncio.sleep(1)
        
        # Execute the pipeline (this might take a while)
        results = await asyncio.get_event_loop().run_in_executor(
            None, 
            run_agent_pipeline, 
            user_interests, 
            budget, 
            risk_tolerance
        )
        
        # Parse the final output
        parsed_results = parse_agent_output(results['final_strategy'])
        
        yield f"data: {json.dumps({'type': 'status', 'message': 'Generating investment strategy...'})}\n\n"
        await asyncio.sleep(1)
        
        # Stream strategy
        yield f"data: {json.dumps({'type': 'strategy_start'})}\n\n"
        await asyncio.sleep(0.5)
        
        async for chunk in stream_text_simulation(parsed_results['strategy'], chunk_size=3, delay=0.05):
            yield f"data: {json.dumps({'type': 'strategy', 'content': chunk})}\n\n"
        
        yield f"data: {json.dumps({'type': 'strategy_end'})}\n\n"
        await asyncio.sleep(1)
        
        # Stream allocation
        yield f"data: {json.dumps({'type': 'allocation_start'})}\n\n"
        await asyncio.sleep(0.5)
        
        async for chunk in stream_text_simulation(parsed_results['allocation'], chunk_size=3, delay=0.05):
            yield f"data: {json.dumps({'type': 'allocation', 'content': chunk})}\n\n"
        
        yield f"data: {json.dumps({'type': 'allocation_end'})}\n\n"
        await asyncio.sleep(1)
        
        # Stream expected returns
        yield f"data: {json.dumps({'type': 'return_start'})}\n\n"
        await asyncio.sleep(0.5)
        
        async for chunk in stream_text_simulation(parsed_results['expected_return'], chunk_size=3, delay=0.05):
            yield f"data: {json.dumps({'type': 'expected_return', 'content': chunk})}\n\n"
        
        yield f"data: {json.dumps({'type': 'return_end'})}\n\n"
        
        # Send completion signal
        yield f"data: {json.dumps({'type': 'complete', 'message': 'Analysis complete!'})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': f'Error: {str(e)}'})}\n\n"


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_html():
    return FileResponse("static/index.html")

@app.post("/analyze")
async def analyze_investment(request: InvestmentRequest):
    """Endpoint to get complete investment analysis (non-streaming)"""
    try:
        results = run_agent_pipeline(
            request.user_interests,
            request.budget,
            request.risk_tolerance
        )
        
        parsed_results = parse_agent_output(results['final_strategy'])
        
        return {
            "success": True,
            "data": {
                "strategy": parsed_results['strategy'],
                "allocation": parsed_results['allocation'],
                "expected_return": parsed_results['expected_return'],
                "location": results['location']
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream-analyze")
async def stream_analyze_investment(request: InvestmentRequest):
    """Endpoint for streaming investment analysis"""
    try:
        return StreamingResponse(
            generate_investment_stream(
                request.user_interests,
                request.budget,
                request.risk_tolerance
            ),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
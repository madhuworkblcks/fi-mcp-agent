import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai

# --- Configuration ---
# Initialize the FastAPI app
app = FastAPI(
    title="Unstructured Finance Analysis Agent",
    description="An agent that analyzes unstructured text about finances and provides recommendations.",
    version="1.0.0"
)

# Configure the Gemini API Key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=api_key)

# --- Pydantic Models for Request and Response ---
# This defines the expected structure of our input data
class AnalysisRequest(BaseModel):
    unstructured_text: str
    context: str = "rural loans and government schemes in India"

# --- Gemini Model Interaction ---
# Initialize the Gemini Pro model
model = genai.GenerativeModel('gemini-1.5-flash-001')

# Define the structured output we want from the model
JSON_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "A brief, one-paragraph summary of the provided text."},
        "key_entities": {
            "type": "array",
            "description": "A list of key financial entities or terms mentioned.",
            "items": {"type": "string"}
        },
        "sentiment": {"type": "string", "description": "Overall sentiment (Positive, Negative, Neutral)."},
        "recommendations": {
            "type": "array",
            "description": "A list of 2-3 actionable financial recommendations based on the text.",
            "items": {"type": "string"}
        },
        "potential_risks": {
            "type": "array",
            "description": "A list of potential risks or downsides identified from the text.",
            "items": {"type": "string"}
        }
    },
    "required": ["summary", "key_entities", "sentiment", "recommendations", "potential_risks"]
}

# --- API Endpoint ---
@app.post("/analyze", response_class=JSONResponse)
async def analyze_unstructured_data(request: AnalysisRequest):
    """
    This endpoint receives unstructured text and returns a structured analysis.
    """
    try:
        # Construct a detailed prompt for the Gemini model
        prompt = f"""
        Analyze the following unstructured text regarding financial matters, specifically in the context of '{request.context}'.
        Your task is to act as an expert financial analyst. Extract key information, identify risks, and provide actionable recommendations.
        
        Please provide your analysis strictly in the following JSON format:
        {json.dumps(JSON_OUTPUT_SCHEMA)}

        Here is the text to analyze:
        ---
        {request.unstructured_text}
        ---
        """

        # Generate content using the model with JSON output forced
        response = await model.generate_content_async(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )

        # Parse the JSON response from the model
        parsed_response = json.loads(response.text)

        return JSONResponse(content=parsed_response, status_code=200)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# Health check endpoint
@app.get("/")
def read_root():
    return {"status": "Agent is running"}
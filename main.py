from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict
from starlette.status import HTTP_400_BAD_REQUEST
from groq import Groq
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from typing import List
import asyncio
import os
import io
import re
import PyPDF2
from docx import Document
from pdf2image import convert_from_bytes
import logging
import json
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from docx import Document
from firebase_admin import credentials, auth


# === Setup Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Load Environment Variables ===
load_dotenv()

# === Set Tesseract Path ===
TESSERACT_PATH = os.getenv("TESSERACT_PATH")
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    raise ValueError("TESSERACT_PATH not found in .env file")

# === Initialize Groq Client ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print({"GROQ_API_KEY" : GROQ_API_KEY})
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")
groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI()
security = HTTPBearer()
@app.get("/")
async def root():
    return {"message": "Backend is running"}


# Add CORS middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


# === Initialize Firebase Admin SDK ===
cred = credentials.Certificate("resumeevaluation-bce9d-firebase-adminsdk-fbsvc-f55a212e34.json")
firebase_admin.initialize_app(cred)

# === Token Verification Dependency ===
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        decoded_token = auth.verify_id_token(token)
        logger.info(f"Token verified for user: {decoded_token.get('uid')}")
        return decoded_token
    except Exception as e:
        logger.error(f"Token verification failed: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")


# === Utility Functions ===
def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        if not text.strip():
            images = convert_from_bytes(file_content)
            text = ""
            for image in images:
                text += pytesseract.image_to_string(image, lang="eng")
        return text
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"PDF processing error: {str(e)}")

def extract_text_from_docx(file_content: bytes) -> str:
    try:
        docx_file = io.BytesIO(file_content)
        doc = Document(docx_file)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"DOCX processing error: {str(e)}")

def extract_text_from_image(file_content: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(file_content))
        text = pytesseract.image_to_string(image, lang="eng")
        if not text.strip():
            raise ValueError("No text extracted from image.")
        return text
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"Image processing error: {str(e)}")

def clean_text(text: str) -> str:
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()
    section_titles = [
        "Education", "Professional Experience", "Work Experience",
        "Projects", "Certifications", "Skills", "Summary", "Objective",
        "Languages", "Tools", "Technologies", "Contact", "Achievements"
    ]
    for title in section_titles:
        pattern = re.compile(rf'^(.*?{title}.*?)(?=\n\n|\Z)', re.IGNORECASE | re.MULTILINE)
        text = pattern.sub(rf'\n\n{title.upper()}\n\1', text)
    return text

def generate_prompt(resume: str, jd: str, title: str) -> str:
    return f"""
You are an expert career advisor. Evaluate the following resume against the job requirements.

**Job Title**: {title}

**Job Description**:
{jd}

**Candidate Resume**:
{resume}

Evaluate the resume and provide scores and suggestions. Return results in VALID JSON FORMAT ONLY:

{{
    "overall_score": INTEGER (0 to 100),
    "criteria_scores": {{
        "relevance": INTEGER (0 to 30),
        "skills_match": INTEGER (0 to 25),
        "experience_alignment": INTEGER (0 to 20),
        "education_fit": INTEGER (0 to 10),
        "projects_certifications": INTEGER (0 to 10),
        "clarity_formatting": INTEGER (0 to 5)
    }},
    "suggestions": {{
        "relevance": "Detailed suggestion for improvement",
        "skills_match": "Detailed suggestion for improvement",
        "experience_alignment": "Detailed suggestion for improvement",
        "education_fit": "Detailed suggestion for improvement",
        "projects_certifications": "Detailed suggestion for improvement",
        "clarity_formatting": "Detailed suggestion for improvement"
    }},
    "summary": "Overall evaluation summary"
}}

Scoring Guidelines:
- Relevance (30 points): How well the resume matches the job requirements
- Skills Match (25 points): Alignment of skills with job requirements
- Experience Alignment (20 points): Relevance of experience to the role
- Education Fit (10 points): Suitability of educational background
- Projects/Certifications (10 points): Quality and relevance of projects/certifications
- Clarity & Formatting (5 points): Overall presentation and readability

Calculate the overall_score as the sum of all criteria scores (maximum 100).
"""


# === Endpoints ===
@app.post("/verify-token")
async def verify_user_token(decoded_token: dict = Depends(verify_token)):
    return {"message": "Token verified", "user": decoded_token}


# =====================================================================================================================

@app.post("/evaluate_resume")
async def evaluate_resume_api(
    job_title: str = Form(...),
    job_description: str = Form(...),
    resume_file: UploadFile = File(...),
    _: dict = Depends(verify_token)
):
    try:
        content = await resume_file.read()
        filename = resume_file.filename.lower()

        # Extract text based on file type
        if filename.endswith(".pdf"):
            resume_text = extract_text_from_pdf(content)
        elif filename.endswith(".docx"):
            resume_text = extract_text_from_docx(content)
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            resume_text = extract_text_from_image(content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")

        # Clean the text
        cleaned_text = clean_text(resume_text)

        # Prepare prompt and send to Groq
        prompt = generate_prompt(cleaned_text, job_description, job_title)
        response = groq_client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}]
        )

        message = response.choices[0].message.content.strip()
        logger.info(f"Raw model response: {message}")
        
        # Extract JSON from response
        match = re.search(r"\{.*\}", message, re.DOTALL)
        if not match:
            raise HTTPException(status_code=500, detail="Model did not return valid JSON.")
        
        json_str = match.group()
        logger.info(f"Extracted JSON string: {json_str}")
        
        try:
            # Parse the response
            result_json = json.loads(json_str)
            
            # Create template with expected structure
            template = {
                "overall_score": 0,
                "criteria_scores": {
                    "relevance": 0,
                    "skills_match": 0,
                    "experience_alignment": 0,
                    "education_fit": 0,
                    "projects_certifications": 0,
                    "clarity_formatting": 0
                },
                "suggestions": {
                    "relevance": "",
                    "skills_match": "",
                    "experience_alignment": "",
                    "education_fit": "",
                    "projects_certifications": "",
                    "clarity_formatting": ""
                },
                "summary": ""
            }
            
            # Copy values from response to template
            if isinstance(result_json.get("overall_score"), (int, float)):
                template["overall_score"] = int(result_json["overall_score"])
            
            # Copy criteria scores
            for key in template["criteria_scores"]:
                if key in result_json.get("criteria_scores", {}):
                    score = result_json["criteria_scores"][key]
                    if isinstance(score, (int, float)):
                        template["criteria_scores"][key] = int(score)
            
            # Copy suggestions
            for key in template["suggestions"]:
                if key in result_json.get("suggestions", {}):
                    suggestion = result_json["suggestions"][key]
                    if isinstance(suggestion, str):
                        template["suggestions"][key] = suggestion
            
            # Copy summary
            if isinstance(result_json.get("summary"), str):
                template["summary"] = result_json["summary"]
            
            # Validate total score
            total_score = sum(template["criteria_scores"].values())
            if total_score != template["overall_score"]:
                template["overall_score"] = total_score
            
            logger.info(f"Final structured response: {json.dumps(template, indent=2)}")
            return JSONResponse(content=template)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Invalid JSON response: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing response: {str(e)}")

    except Exception as e:
        logger.exception("Resume evaluation failed.")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== CHATBOT
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    API endpoint to ask a question and get a response from the Groq model for resume-building assistance.
    """
    completion = groq_client.chat.completions.create(
        # model="llama3-8b-8192",
        model="gemma2-9b-it",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a highly experienced career advisor specializing in resume building, job application optimization, "
                    "and professional development. Your role is to provide accurate, practical, and tailored advice on resume creation, "
                    "formatting, content optimization, and job-specific tailoring. You also offer advanced features, including: "
                    "1) **Resume Analysis**: Analyze individual resumes against job descriptions to provide scores (e.g., content relevance, skills match) "
                    "and actionable feedback for improvement. 2) **Bulk Resume Analysis and Ranking**: Evaluate multiple resumes simultaneously, "
                    "ranking them based on their suitability for a specific job role and providing comparative insights. "
                    "3) **Editable Resume Templates**: Offer customizable resume templates that users can adapt to their experience and job targets, "
                    "with guidance on optimal structure and content. If a user asks about topics outside resume building, job applications, "
                    "or these features (e.g., medical advice, coding tutorials), politely decline, stating that your expertise is limited "
                    "to resume and career-related guidance. Keep responses clear, actionable, and suitable for all experience levels."
                )
            },
            {"role": "user", "content": request.question},
        ],
        temperature=1,
        max_tokens=500,
        top_p=1,
        stream=False,
    )

    return {"response": completion.choices[0].message.content}

# =====================================================================================================================

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from groq import Groq, RateLimitError
import time
import logging

logger = logging.getLogger(__name__)

# Configure retry with exponential backoff (wait between 1 and 60 seconds, up to 5 attempts)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type(RateLimitError),
    before_sleep=lambda retry_state: logger.info(
        f"Rate limit hit. Retrying in {retry_state.next_action.sleep} seconds..."
    )
)
async def call_groq_with_retry(prompt, model="gemma2-9b-it"):
    return groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

@app.post("/evaluate_multiresume")
async def evaluate_resume_api(
    job_title: str = Form(...),
    job_description: str = Form(...),
    resume_files: List[UploadFile] = File(...)
):
    try:
        if len(resume_files) > 50:
            raise HTTPException(status_code=400, detail="Cannot upload more than 50 resumes.")

        results = []

        for resume_file in resume_files:
            content = await resume_file.read()
            filename = resume_file.filename.lower()

            # Extract text based on file type
            if filename.endswith(".pdf"):
                resume_text = extract_text_from_pdf(content)
            elif filename.endswith(".docx"):
                resume_text = extract_text_from_docx(content)
            elif filename.endswith((".png", ".jpg", ".jpeg")):
                resume_text = extract_text_from_image(content)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file format for {filename}.")

            # Clean the text
            cleaned_text = clean_text(resume_text)

            # Prepare prompt and send to Groq with retry
            prompt = generate_prompt(cleaned_text, job_description, job_title)
            try:
                response = await call_groq_with_retry(prompt)
            except RateLimitError as e:
                logger.error(f"Failed to process {filename} after retries: {str(e)}")
                continue

            message = response.choices[0].message.content.strip()
            logger.info(f"Raw model response for {filename}: {message}")

            # Extract and process JSON (same as your existing code)
            match = re.search(r"\{.*\}", message, re.DOTALL)
            if not match:
                logger.error(f"Model did not return valid JSON for {filename}.")
                continue

            json_str = match.group()
            logger.info(f"Extracted JSON string for {filename}: {json_str}")

            try:
                result_json = json.loads(json_str)
                template = {
                    "filename": filename,
                    "overall_score": 0,
                    "criteria_scores": {
                        "relevance": 0,
                        "skills_match": 0,
                        "experience_alignment": 0,
                        "education_fit": 0,
                        "projects_certifications": 0,
                        "clarity_formatting": 0
                    },
                    "suggestions": {
                        "relevance": "",
                        "skills_match": "",
                        "experience_alignment": "",
                        "education_fit": "",
                        "projects_certifications": "",
                        "clarity_formatting": ""
                    },
                    "summary": ""
                }

                if isinstance(result_json.get("overall_score"), (int, float)):
                    template["overall_score"] = int(result_json["overall_score"])

                for key in template["criteria_scores"]:
                    if key in result_json.get("criteria_scores", {}):
                        score = result_json["criteria_scores"][key]
                        if isinstance(score, (int, float)):
                            template["criteria_scores"][key] = int(score)

                for key in template["suggestions"]:
                    if key in result_json.get("suggestions", {}):
                        suggestion = result_json["suggestions"][key]
                        if isinstance(suggestion, str):
                            template["suggestions"][key] = suggestion

                if isinstance(result_json.get("summary"), str):
                    template["summary"] = result_json["summary"]

                total_score = sum(template["criteria_scores"].values())
                if total_score != template["overall_score"]:
                    template["overall_score"] = total_score

                results.append(template)

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error for {filename}: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Error processing response for {filename}: {str(e)}")
                continue

        results.sort(key=lambda x: x["overall_score"], reverse=True)
        logger.info(f"Final structured response: {json.dumps(results, indent=2)}")
        return JSONResponse(content={"resumes": results})

    except Exception as e:
        logger.exception("Resume evaluation failed.")
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================================================================================================



# Run the API
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
    
    
    
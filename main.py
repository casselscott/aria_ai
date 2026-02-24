# main.py (integrated fixes: OpenAI v1 client, /api/ai/image endpoint, + optional TTS/STT stubs)
import faiss
import os
import base64
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_engine import generate_with_rag, refine_with_edit, regenerate_section

# ✅ OpenAI v1+ client
from openai import OpenAI

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(
    title="ARIA - Abuja Real Estate AI",
    description="AI assistant for Abuja luxury properties",
    version="1.0",
)

# Enable CORS (dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# OpenAI Client (v1+)
# -----------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # Don't crash at import-time in dev; but warn clearly
    print("⚠️ OPENAI_API_KEY not set. Image/TTS/STT endpoints will fail until set.")

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str


class RefineRequest(BaseModel):
    original_text: str
    edited_text: str


class PartialRequest(BaseModel):
    full_text: str
    selected_text: str
    instruction: str


# ✅ Models your UI /chat expects
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


# ✅ Image generation
class ImageRequest(BaseModel):
    prompt: str
    size: Optional[str] = "1024x1024"  # "1024x1024", "1024x1536", etc. depending on model support


class ImageResponse(BaseModel):
    b64_json: str  # base64 image content


# ✅ Text-to-Speech
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "alloy"  # common default voice name
    format: Optional[str] = "mp3"   # "mp3", "wav" depending on API support


class TTSResponse(BaseModel):
    b64_audio: str
    mime_type: str


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "ARIA Abuja Real Estate AI API", "status": "running"}


@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "ARIA AI"}


@app.post("/api/ai/generate")
async def generate(req: GenerateRequest):
    try:
        response = generate_with_rag(req.prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/refine")
async def refine(req: RefineRequest):
    try:
        response = refine_with_edit(req.original_text, req.edited_text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/partial")
async def partial(req: PartialRequest):
    try:
        response = regenerate_section(req.full_text, req.selected_text, req.instruction)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ✅ /chat wired to your real RAG
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        response = generate_with_rag(req.message)
        return {"reply": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# ✅ Image Generation Endpoint (OpenAI v1+)
# -----------------------------------------------------------------------------
@app.post("/api/ai/image", response_model=ImageResponse)
async def create_image(req: ImageRequest):
    """
    Returns a base64-encoded image in JSON:
      { "b64_json": "...." }

    Frontend can render with:
      img.src = `data:image/png;base64,${b64_json}`
    """
    try:
        # NOTE:
        # - If "gpt-image-1" isn't enabled on your account, switch to:
        #   "gpt-image-1.5" or "chatgpt-image-latest" (depending on availability).
        result = client.images.generate(
            model="gpt-image-1",
            prompt=req.prompt,
            size=req.size,
        )

        b64 = result.data[0].b64_json
        if not b64:
            raise RuntimeError("No image data returned by the image API.")

        return {"b64_json": b64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")


# -----------------------------------------------------------------------------
# ✅ Text-to-Speech Endpoint (OpenAI v1+)
# -----------------------------------------------------------------------------
@app.post("/api/ai/tts", response_model=TTSResponse)
async def tts(req: TTSRequest):
    """
    Returns base64 audio:
      { "b64_audio": "...", "mime_type": "audio/mpeg" }

    Frontend can play with:
      audio.src = `data:${mime_type};base64,${b64_audio}`
    """
    try:
        # TTS in OpenAI v1+:
        audio = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=req.voice,
            input=req.text,
            format=req.format,
        )

        # The SDK returns bytes-like content; convert to base64 for transport
        audio_bytes = audio.read() if hasattr(audio, "read") else bytes(audio)
        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")

        mime = "audio/mpeg" if req.format == "mp3" else "audio/wav"
        return {"b64_audio": b64_audio, "mime_type": mime}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


# -----------------------------------------------------------------------------
# ✅ Speech-to-Text Endpoint (OpenAI v1+)
# -----------------------------------------------------------------------------
@app.post("/api/ai/stt")
async def stt(file: UploadFile = File(...)):
    """
    Accepts multipart/form-data with 'file' (audio)
    Returns:
      { "text": "transcript..." }
    """
    try:
        # Save to temp (UploadFile is a SpooledTemporaryFile under the hood, but we can pass bytes)
        audio_bytes = await file.read()

        # OpenAI expects a file-like object; easiest is to write to a temp file
        # (keeps it very compatible across environments)
        import tempfile

        suffix = os.path.splitext(file.filename or "")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()

            transcript = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=open(tmp.name, "rb"),
            )

        # transcript.text is typical in v1 SDK
        return {"text": getattr(transcript, "text", str(transcript))}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {e}")


# -----------------------------------------------------------------------------
# Local run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",  # ✅ import string needed for reload/workers
        host="127.0.0.1",
        port=8000,
        reload=True,
    )

import base64
import json
import os
import logging
import httpx
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Header, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("aetheris")

load_dotenv()

app = FastAPI(title="Aetheris – Medical & Behavioural Vision AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ───────────────────────────────────────────────────────────
EMOTIONS    = ["Happy","Sad","Crying","Pain","Angry","Scared","Surprised",
               "Neutral","Tired / Sleepy","Disgusted","Anxious / Nervous","Confused"]
GESTURES    = ["Holding Stomach","Raising Index Finger","Holding Head",
               "Waving Hand (Hello)","Waving Goodbye","Holding Throat / Coughing",
               "Rubbing Eyes","Thumbs Up","Thumbs Down",
               "Shushing Finger on Lips","Covering Ears","None"]
STATUSES    = ["Hungry or Pain","Needs Bathroom","Headache or Tired","Normal",
               "Crying","Greeting","Leaving","Distressed",
               "Choking or Throat Pain","Approval","Disapproval",
               "Request Quiet","Sensory Overload","Tired or Sleepy"]
CONFIDENCE  = ["High","Medium","Low"]

class AnalysisResponse(BaseModel):
    Detected_Emotion:        str
    Physical_Gesture:        str
    Predicted_Need_or_Status:str
    Confidence_Score:        str
    Detailed_Explanation:    str
    Restlessness_Level:      str
    Actionable_Recommendation: str
    Face_Bounding_Box:       Optional[List[int]] = Field(default=[0,0,0,0])
    Hand_Bounding_Box:       Optional[List[int]] = Field(default=[0,0,0,0])

class AnalysisRequest(BaseModel):
    image: str
    model: Optional[str] = None

# ── Config endpoint ───────────────────────────────────────────────────────────
@app.get("/api/config")
async def get_config():
    api_key     = os.getenv("OPENROUTER_API_KEY") or os.getenv("GEMINI_API_KEY", "")
    is_openrouter = api_key.startswith("sk-or-")
    default_model = (
        os.getenv("DEFAULT_MODEL")
        or ("google/gemini-2.5-flash" if is_openrouter else "gemini-2.5-flash")
    )
    return {
        "has_server_key": bool(api_key),
        "default_model":  default_model,
        "is_openrouter":  is_openrouter,
    }

# ── Main analysis endpoint ────────────────────────────────────────────────────
@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_frame(
    request: AnalysisRequest,
    response: Response,
    x_gemini_api_key: Optional[str] = Header(None, alias="X-Gemini-API-Key"),
):
    # Payload size guard (~5 MB base64)
    if len(request.image) > 7_000_000:
        raise HTTPException(status_code=413, detail="Payload too large. Max ~5 MB.")

    api_key = (
        x_gemini_api_key
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="API key missing. Add it in Settings or configure the server .env file.",
        )

    # Strip base64 header if present
    img_str = request.image
    if "base64," in img_str:
        img_str = img_str.split("base64,", 1)[1]

    try:
        base64.b64decode(img_str, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image encoding.")

    # Highly detailed prompt – specific visual cues for every emotion & gesture
    prompt = (
        "You are a precise medical-behavioural AI vision analyst.\n"
        "Carefully examine every visible detail in this camera frame and produce a structured analysis.\n\n"

        "=== FACIAL EMOTION (pick EXACTLY one) ===\n"
        "Happy: corners of mouth turned up, cheeks raised, eyes crinkling, smiling or laughing.\n"
        "Sad: mouth corners pulled down, droopy eyelids, furrowed brow, downcast gaze, frowning.\n"
        "Crying: tears visible, squinted wet eyes, scrunched nose, open quivering mouth, wiping eyes.\n"
        "Pain: tightly closed or squinting eyes, deep forehead furrows, clenched jaw, grimace, wincing.\n"
        "Angry: lowered/knitted brows, narrowed eyes, flared nostrils, pressed-tight lips, scowl.\n"
        "Scared: wide-open eyes with visible whites, raised brows pulled together, mouth open in panic.\n"
        "Surprised: brows raised high, eyes fully open and wide, mouth dropped open in shock.\n"
        "Neutral: relaxed face, no muscular tension, calm resting expression, no emotion shown.\n"
        "Tired / Sleepy: heavy drooping eyelids, yawning, glazed unfocused eyes, slow blinking.\n"
        "Disgusted: nose wrinkled, upper lip curled, head slightly recoiling, expression of aversion.\n"
        "Anxious / Nervous: biting or pressing lips, tense eyebrows, darting or wide eyes, worried look.\n"
        "Confused: head tilted to side, one brow raised, squinted eyes, puzzled uncertain expression.\n\n"

        "=== PHYSICAL GESTURE (pick EXACTLY one) ===\n"
        "Holding Stomach: one or both hands pressed flat over the abdomen/stomach area.\n"
        "Raising Index Finger: single index finger raised straight up, all other fingers folded down.\n"
        "Holding Head: hands pressing temples, forehead, or sides of head with evident discomfort.\n"
        "Waving Hand (Hello): open palm raised facing camera, waving side to side as a greeting.\n"
        "Waving Goodbye: hand waving sideways away from camera or arm pointing outward to indicate leaving.\n"
        "Holding Throat / Coughing: hand touching or gripping the neck/throat, or fist at mouth coughing.\n"
        "Rubbing Eyes: one or both hands actively rubbing the eyes or eye area.\n"
        "Thumbs Up: thumb pointing clearly upward with other fingers closed in a fist.\n"
        "Thumbs Down: thumb pointing clearly downward with other fingers closed in a fist.\n"
        "Shushing Finger on Lips: index finger placed vertically over closed lips (shhh gesture).\n"
        "Covering Ears: both hands pressed flat over both ears simultaneously.\n"
        "None: no active hand or body gesture visible, hands resting naturally.\n\n"

        "=== PREDICTED NEED / STATUS (pick EXACTLY one) ===\n"
        "Map gesture→status as follows:\n"
        "Holding Stomach → Hungry or Pain\n"
        "Raising Index Finger → Needs Bathroom\n"
        "Holding Head OR Rubbing Eyes → Headache or Tired\n"
        "Waving Hand (Hello) → Greeting\n"
        "Waving Goodbye → Leaving\n"
        "Holding Throat / Coughing → Choking or Throat Pain\n"
        "Thumbs Up → Approval\n"
        "Thumbs Down → Disapproval\n"
        "Shushing Finger on Lips → Request Quiet\n"
        "Covering Ears → Sensory Overload\n"
        "Crying emotion → Crying\n"
        "Angry OR Scared OR Anxious / Nervous emotion (with no gesture) → Distressed\n"
        "Tired / Sleepy emotion (with no gesture) → Tired or Sleepy\n"
        "No significant gesture or distress → Normal\n\n"

        "=== BOUNDING BOXES ===\n"
        "Face_Bounding_Box: [ymin, xmin, ymax, xmax] in 0-1000 scale. Always provide if face visible.\n"
        "Hand_Bounding_Box: [ymin, xmin, ymax, xmax] in 0-1000 scale. Provide when hand/gesture is active. [0,0,0,0] otherwise.\n\n"

        "Output ONLY a raw JSON object with keys: Detected_Emotion, Physical_Gesture, "
        "Predicted_Need_or_Status, Confidence_Score, Detailed_Explanation, Restlessness_Level, "
        "Actionable_Recommendation, Face_Bounding_Box, Hand_Bounding_Box. No markdown."
    )

    # JSON schema for Gemini native structured output
    schema = {
        "type": "object",
        "properties": {
            "Detected_Emotion":         {"type": "string", "enum": EMOTIONS},
            "Physical_Gesture":         {"type": "string", "enum": GESTURES},
            "Predicted_Need_or_Status": {"type": "string", "enum": STATUSES},
            "Confidence_Score":         {"type": "string", "enum": CONFIDENCE},
            "Detailed_Explanation":     {"type": "string"},
            "Restlessness_Level":       {"type": "string", "enum": CONFIDENCE},
            "Actionable_Recommendation":{"type": "string"},
            "Face_Bounding_Box":        {"type": "array", "items": {"type": "integer"}},
            "Hand_Bounding_Box":        {"type": "array", "items": {"type": "integer"}},
        },
        "required": [
            "Detected_Emotion","Physical_Gesture","Predicted_Need_or_Status",
            "Confidence_Score","Detailed_Explanation","Restlessness_Level",
            "Actionable_Recommendation",
        ],
    }

    is_openrouter = api_key.startswith("sk-or-")

    try:
        if is_openrouter:
            model = _or_model(request.model or os.getenv("DEFAULT_MODEL") or "google/gemini-2.5-flash")
            return await _call_openrouter(api_key, model, request.image, prompt)
        else:
            model = _gg_model(request.model or os.getenv("DEFAULT_MODEL") or "gemini-2.5-flash")
            return await _call_gemini(api_key, model, img_str, prompt, schema)

    except HTTPException as http_err:
        # Transparent fallback: if client key fails with 401/403/429, try server key
        server_key = os.getenv("OPENROUTER_API_KEY")
        if http_err.status_code in (401, 403, 429) and x_gemini_api_key and server_key:
            logger.warning(
                f"Client key failed ({http_err.status_code}). Falling back to server key."
            )
            fb_model = _or_model(os.getenv("DEFAULT_MODEL") or "google/gemini-2.5-flash")
            try:
                return await _call_openrouter(server_key, fb_model, request.image, prompt)
            except Exception as fb_err:
                logger.error(f"Fallback also failed: {fb_err}")
                raise http_err
        raise

# ── Helpers ───────────────────────────────────────────────────────────────────
def _or_model(m: str) -> str:
    """Normalise model name for OpenRouter (must have vendor prefix)."""
    alias = {"gemini-2.5-flash": "google/gemini-2.5-flash",
             "gemini-2.5-pro":   "google/gemini-2.5-pro"}
    return alias.get(m, m)

def _gg_model(m: str) -> str:
    """Normalise model name for Google Gemini (strip vendor prefix)."""
    return m.split("/")[-1] if "/" in m else m

def _level_from_value(val) -> str:
    """Convert numeric (0-1 or 0-100) or string confidence/restlessness to High/Medium/Low."""
    if val is None:
        return "Low"
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("high", "h"):   return "High"
        if v in ("medium", "med", "m", "moderate"): return "Medium"
        if v in ("low", "l"):    return "Low"
        # Try parsing as a number string
        try:
            num = float(v)
            if num > 1:  num = num / 100.0   # 0-100 scale
            if num >= 0.65: return "High"
            if num >= 0.35: return "Medium"
            return "Low"
        except ValueError:
            return "Low"
    if isinstance(val, (int, float)):
        num = float(val)
        if num > 1: num = num / 100.0         # handle 0-100 scale
        if num >= 0.65: return "High"
        if num >= 0.35: return "Medium"
        return "Low"
    return "Low"

def _safe_bbox(val) -> list:
    """Ensure bounding box is a list of 4 ints 0-1000. Returns [0,0,0,0] on any problem."""
    try:
        if not val or not isinstance(val, (list, tuple)) or len(val) < 4:
            return [0, 0, 0, 0]
        ints = [int(float(v)) for v in val[:4]]
        # Validate all values in range
        if all(0 <= x <= 1000 for x in ints):
            return ints
        # If values > 1 and <= 1 scale (normalised 0-1 from AI), scale up
        if all(0.0 <= float(v) <= 1.0 for v in val[:4]):
            return [int(float(v) * 1000) for v in val[:4]]
        return [0, 0, 0, 0]
    except Exception:
        return [0, 0, 0, 0]

def _sanitize(raw: dict) -> dict:
    """
    Normalise the AI JSON response to our canonical key names and types.
    Handles numeric confidence scores, wrong types, missing keys, and type mismatches.
    """
    KEY_ALIASES = {
        "Detected_Emotion":          ["Detected_Emotion","detected_emotion","emotion","Emotion","facial_emotion","facial_expression"],
        "Physical_Gesture":          ["Physical_Gesture","physical_gesture","gesture","Gesture","hand_gesture","body_gesture"],
        "Predicted_Need_or_Status":  ["Predicted_Need_or_Status","predicted_need_or_status","predicted_status","need","status","need_or_status","predicted_need"],
        "Confidence_Score":          ["Confidence_Score","confidence_score","confidence","Confidence","confidence_level","confidence_value"],
        "Detailed_Explanation":      ["Detailed_Explanation","detailed_explanation","explanation","Explanation","reasoning","analysis"],
        "Restlessness_Level":        ["Restlessness_Level","restlessness_level","restlessness","Restlessness","agitation_level","agitation"],
        "Actionable_Recommendation": ["Actionable_Recommendation","actionable_recommendation","recommendation","Recommendation","suggested_action","action"],
        "Face_Bounding_Box":         ["Face_Bounding_Box","face_bounding_box","face_box","FaceBox","face_bbox","face_coordinates","face_location"],
        "Hand_Bounding_Box":         ["Hand_Bounding_Box","hand_bounding_box","hand_box","HandBox","hand_bbox","hand_coordinates","hand_location","gesture_bounding_box"],
    }
    DEFAULTS = {
        "Detected_Emotion":          "Neutral",
        "Physical_Gesture":          "None",
        "Predicted_Need_or_Status":  "Normal",
        "Confidence_Score":          "Low",
        "Detailed_Explanation":      "No details provided.",
        "Restlessness_Level":        "Low",
        "Actionable_Recommendation": "Continue standard observation.",
        "Face_Bounding_Box":         [0, 0, 0, 0],
        "Hand_Bounding_Box":         [0, 0, 0, 0],
    }

    out = {}
    for key, aliases in KEY_ALIASES.items():
        val = None
        for alias in aliases:
            if alias in raw:
                val = raw[alias]
                break

        if val is None:
            out[key] = DEFAULTS[key]
            continue

        # Type-safe coercion per field
        if key in ("Confidence_Score", "Restlessness_Level"):
            out[key] = _level_from_value(val)
        elif key in ("Face_Bounding_Box", "Hand_Bounding_Box"):
            out[key] = _safe_bbox(val)
        elif key in ("Detected_Emotion", "Physical_Gesture", "Predicted_Need_or_Status"):
            out[key] = str(val).strip() if val else DEFAULTS[key]
        else:
            out[key] = str(val).strip() if val else DEFAULTS[key]

    return out

def _parse_json_text(text: str) -> dict:
    """Strip markdown fences and parse JSON. Raises ValueError on failure."""
    s = text.strip()
    if s.startswith("```"):
        # Remove opening fence (with optional language tag)
        s = s.split("\n", 1)[-1] if "\n" in s else s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return json.loads(s.strip())

# ── OpenRouter call ───────────────────────────────────────────────────────────
async def _call_openrouter(api_key: str, model: str, image_b64: str, prompt: str) -> AnalysisResponse:
    image_url = image_b64 if image_b64.startswith("data:") else f"data:image/jpeg;base64,{image_b64}"

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text",      "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }],
        "response_format": {"type": "json_object"},
        "temperature": 0.05,
        "max_tokens": 350,
    }
    headers = {
        "Authorization":  f"Bearer {api_key}",
        "Content-Type":   "application/json",
        "HTTP-Referer":   "http://localhost:8000",
        "X-Title":        "Aetheris Medical Vision AI",
    }

    async with httpx.AsyncClient(timeout=28.0) as client:
        r = await client.post("https://openrouter.ai/api/v1/chat/completions",
                              headers=headers, json=payload)

    if r.status_code == 429:
        raise HTTPException(status_code=429, detail="OpenRouter rate limit exceeded (429).")
    if r.status_code != 200:
        detail = r.text
        try:
            detail = r.json().get("error", {}).get("message", detail)
        except Exception:
            pass
        raise HTTPException(status_code=r.status_code, detail=f"OpenRouter error: {detail}")

    choices = r.json().get("choices", [])
    if not choices:
        raise HTTPException(status_code=502, detail="OpenRouter returned no choices.")

    raw_text = (choices[0].get("message") or {}).get("content") or ""
    if not raw_text.strip():
        raise HTTPException(status_code=502, detail="OpenRouter returned empty content.")

    try:
        parsed = _parse_json_text(raw_text)
        return AnalysisResponse(**_sanitize(parsed))
    except Exception as e:
        logger.error(f"OpenRouter JSON parse error: {e} | raw: {raw_text[:300]}")
        raise HTTPException(status_code=502, detail=f"Failed to parse OpenRouter JSON: {e}")

# ── Google Gemini call ────────────────────────────────────────────────────────
async def _call_gemini(api_key: str, model: str, img_str: str, prompt: str, schema: dict) -> AnalysisResponse:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{
            "parts": [
                {"inlineData": {"mimeType": "image/jpeg", "data": img_str}},
                {"text": prompt},
            ],
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema":   schema,
            "temperature":      0.05,
            "maxOutputTokens":  350,
        },
    }

    async with httpx.AsyncClient(timeout=28.0) as client:
        r = await client.post(url, json=payload)

    if r.status_code == 429:
        try:
            msg = r.json().get("error", {}).get("message", "Rate limit exceeded.")
        except Exception:
            msg = "Rate limit exceeded."
        raise HTTPException(status_code=429, detail=f"Gemini rate limit: {msg}")
    if r.status_code != 200:
        detail = r.text
        try:
            detail = r.json().get("error", {}).get("message", detail)
        except Exception:
            pass
        raise HTTPException(status_code=r.status_code, detail=f"Gemini error: {detail}")

    candidates = r.json().get("candidates", [])
    if not candidates:
        raise HTTPException(status_code=502, detail="Gemini returned no candidates.")

    parts = (candidates[0].get("content") or {}).get("parts") or [{}]
    raw_text = (parts[0].get("text") or "")
    if not raw_text.strip():
        raise HTTPException(status_code=502, detail="Gemini returned empty content.")

    try:
        parsed = _parse_json_text(raw_text)
        return AnalysisResponse(**_sanitize(parsed))
    except Exception as e:
        logger.error(f"Gemini JSON parse error: {e} | raw: {raw_text[:300]}")
        raise HTTPException(status_code=502, detail=f"Failed to parse Gemini JSON: {e}")

# ── Static files ──────────────────────────────────────────────────────────────
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

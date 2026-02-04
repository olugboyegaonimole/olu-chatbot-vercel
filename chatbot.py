from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (Allow GitHub Pages to call the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model and Tokenizer at Startup (for faster responses)
print("Loading BlenderBot model into memory...")
MODEL_NAME = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)
model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)
print("Model loaded successfully!")

# Define the request structure
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        user_input = request.message.strip()
        if not user_input:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        # Tokenize input and generate response
        inputs = tokenizer(user_input, return_tensors="pt")
        response_ids = model.generate(**inputs)
        bot_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

        return {"response": bot_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root Endpoint for API Status Check
@app.get("/", include_in_schema=False)
@app.head("/")
def root():
    return {"message": "Welcome to the Chatbot API!"}

# Run the FastAPI app with optimized settings
if __name__ == "__main__":
    uvicorn.run("chatbot:app", host="0.0.0.0", port=8000, workers=1)

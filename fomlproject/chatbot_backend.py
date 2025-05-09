import requests
from transformers import pipeline, BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch
from fuzzywuzzy import fuzz
import re
import sys
import os
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Set UTF-8 encoding for console
if os.name == "nt":
    os.system("chcp 65001 > nul")
sys.stdout.reconfigure(encoding="utf-8")

# Print startup message
print("Starting chatbot initialization...")

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Warning: GPU not detected. Running on CPU. Check CUDA setup.")

# Initialize models
print("Loading dialogue model (this may take 5-20 seconds on GPU)...")
try:
    dialogue_tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    dialogue_model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill").to(device)
    print("Dialogue model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}. Check disk space or internet.")
    exit(1)

print("Loading sentiment analyzer...")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if device.type == "cuda" else -1)
print("Sentiment analyzer loaded.")

# API configurations (load from environment variables for security)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "46618fb78a244d31a162e64a028460cf")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "944e2d9a86c328075eb70972cb8c85e3")
NEWS_API_URL = "https://newsapi.org/v2/everything"
WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"

# Conversation history
conversation_history = []

# Simple joke database
jokes = [
    "Why did the scarecrow become a motivational speaker? Because he was outstanding in his field!",
    "What do you call a bear with no socks on? Barefoot!",
    "Why can’t basketball players go on vacation? Because they would get called for traveling!"
]

# Dictionary to map sentiments to food recommendations and YouTube recipe links
FOOD_RECOMMENDATIONS = {
    "positive": {
        "food": "chocolate cake",
        "youtube_link": "https://www.youtube.com/watch?v=9ukEM1e5uVU" 
    },
    "negative": {
        "food": "chicken soup",
        "youtube_link": "https://www.youtube.com/watch?v=Z9tCFqwT1wY&list=PL-NHxGqGGGk8gerVuoUJzJUMeRfcVD8zX"  
    },
    "neutral": {
        "food": "vegetable stir-fry",
        "youtube_link": "https://www.youtube.com/watch?v=YDw2TkCmXgQ&list=PL-NHxGqGGGk96lkKvjvpBKGfIdKE2Izwk"  
    }
}

# Function to fetch news
def fetch_news(query="general", max_articles=3):
    try:
        params = {"q": query, "apiKey": NEWS_API_KEY, "language": "en", "sortBy": "publishedAt", "pageSize": max_articles}
        response = requests.get(NEWS_API_URL, params=params)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        if not articles:
            return "No recent news found on that topic."
        summary = "\n".join([f"- {a['title']}: {a['description'] or 'No description.'}" for a in articles])
        return f"Here’s the latest news on {query}:\n{summary}"
    except Exception as e:
        return f"Sorry, I couldn’t fetch news: {str(e)}"

# Function to fetch weather
def fetch_weather(city="Chennai"):
    try:
        params = {"q": city, "appid": WEATHER_API_KEY, "units": "metric"}
        response = requests.get(WEATHER_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get("cod") != 200:
            return f"Couldn’t find weather for {city}: {data.get('message', 'Unknown error')}"
        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        return f"The weather in {city} is {weather} with a temperature of {temp}°C."
    except Exception as e:
        return f"Sorry, I couldn’t fetch the weather: {str(e)}"

# Function to fetch Wikipedia
def fetch_wikipedia(query):
    try:
        query = query.strip()
        response = requests.get(f"https://api.wikimedia.org/core/v1/wikipedia/en/search/page?q={query}")
        response.raise_for_status()
        results = response.json().get("pages", [])
        if not results:
            return f"No information found on {query}."
        top_result = results[0]
        title = top_result.get("title", query)
        excerpt = top_result.get("excerpt", "No details available.")
        description = top_result.get("description", "")
        clean_text = re.sub(r"<[^>]+>", "", excerpt).replace("  ", " ").strip()
        response_text = f"Here’s what I found on {title}: {clean_text}"
        if description:
            response_text += f" ({description})"
        return response_text
    except Exception as e:
        return f"Sorry, I couldn’t fetch info on {query}: {str(e)}"

# Function to generate response
def generate_response(user_input):
    global conversation_history
    conversation_history.append({"role": "user", "text": user_input})
    input_lower = user_input.lower().strip()

    # Check for end conversation
    if fuzz.ratio("thank you", input_lower) > 70 or "thanks" in input_lower or input_lower == "thank":
        user_texts = [msg["text"] for msg in conversation_history if msg["role"] == "user"]
        if user_texts:
            sentiments = sentiment_analyzer(user_texts)
            # Compute a weighted sentiment score
            total_score = 0
            for sentiment in sentiments:
                score = sentiment["score"]  # Confidence score (0 to 1)
                if sentiment["label"] == "POSITIVE":
                    total_score += score  # Positive score
                else:  # NEGATIVE
                    total_score -= score  # Negative score
            average_score = total_score / len(sentiments)
            # Classify sentiment based on average score
            if average_score > 0.1:
                overall_sentiment = "positive"
            elif average_score < -0.1:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"
        else:
            overall_sentiment = "neutral"
        # Get food recommendation based on sentiment
        recommendation = FOOD_RECOMMENDATIONS[overall_sentiment]
        food = recommendation["food"]
        youtube_link = recommendation["youtube_link"]
        goodbye_message = (
            f"Goodbye! The sentiment of our conversation was {overall_sentiment}. "
            f"Based on that, I recommend trying some {food}. Here's a recipe video: {youtube_link}\n"
            # f"If the video is unavailable, you can search for '{food} recipe' on YouTube to find another tutorial."
        )
        conversation_history = []
        return goodbye_message

    # Check for weather query
    weather_keywords = ["weather", "temperature", "forecast"]
    if any(keyword in input_lower for keyword in weather_keywords):
        city = input_lower
        for keyword in weather_keywords + ["today", "now", "is", "what", "the"]:
            city = city.replace(keyword, "").strip()
        if not city or city in ["in", "at", ""]:
            city = "Chennai"
        weather_response = fetch_weather(city)
        conversation_history.append({"role": "bot", "text": weather_response})
        return weather_response

    # Check for news query
    news_keywords = ["news", "latest", "what's happening", "current events"]
    if any(keyword in input_lower for keyword in news_keywords):
        topic = input_lower
        for keyword in news_keywords:
            topic = topic.replace(keyword, "").strip()
        if not topic or topic in ["on", "about", "the"]:
            topic = "general"
        news_response = fetch_news(topic)
        conversation_history.append({"role": "bot", "text": news_response})
        return news_response

    # Check for Wikipedia query
    wiki_keywords = ["tell me about", "what is", "who is", "about", "what does mean", "mean by", "famous", "known for"]
    wiki_triggered = False
    wiki_query = ""
    for keyword in wiki_keywords:
        if keyword in input_lower:
            wiki_triggered = True
            wiki_query = input_lower.replace(keyword, "").strip()
            break
    if wiki_triggered and wiki_query:
        wifi_response = fetch_wikipedia(wiki_query)
        conversation_history.append({"role": "bot", "text": wifi_response})
        return wifi_response

    # Handle specific intents
    if "hear me" in input_lower or "can you hear" in input_lower:
        response = "Yup, I hear you loud and clear! What’s up?"
    elif "who are you" in input_lower or "hu r u" in input_lower:
        response = "I’m Cibus,I'm here to chat with and find your meal today!"
    elif any(word in input_lower for word in ["bad", "ugly", "stupid", "shut"]):
        response = "Ouch, sounds like you’re not in a great mood! Want to tell me what’s up or maybe ask about something fun?"
    elif "joke" in input_lower:
        response = random.choice(jokes)
    elif "how" in input_lower and "life" in input_lower:
        response = "My life’s pretty simple—just chatting with folks like you all day! What’s your life like?"
    elif "how" in input_lower and "you" in input_lower:
        response = "I’m doing awesome, thanks for asking! How about you?"
    elif len(input_lower.split()) <= 2 and "thank" not in input_lower:
        response = f"{user_input.capitalize()}? That’s interesting—care to share more?"
    else:
        # General conversation with BlenderBot
        context = []
        for msg in conversation_history[-3:]:
            prefix = "__USER__" if msg["role"] == "user" else "__BOT__"
            context.append(f"{prefix} {msg['text']}")
        context = "\n".join(context).strip() or user_input

        inputs = dialogue_tokenizer(context, return_tensors="pt", truncation=True, max_length=128).to(device)
        outputs = dialogue_model.generate(
            **inputs,
            max_length=150,
            num_beams=5,
            length_penalty=0.8,
            no_repeat_ngram_size=3,
            early_stopping=True,
            pad_token_id=dialogue_tokenizer.eos_token_id,
            bos_token_id=dialogue_tokenizer.bos_token_id,
            eos_token_id=dialogue_tokenizer.eos_token_id
        )
        response = dialogue_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Validate response
        if not response or len(response.split()) < 3 or response.lower() == input_lower:
            response = "I might’ve missed that one—could you give me a bit more?"
        elif any(word in response.lower() for word in ["work", "job", "accountant", "cashier"]):
            response = "I don’t have a job—I’m just here to chat! What’s on your mind?"

    conversation_history.append({"role": "bot", "text": response})
    return response

# FastAPI setup
app = FastAPI(title="AI Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request validation
class ChatRequest(BaseModel):
    message: str

# API endpoints
@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        response = generate_response(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

# Run the server
if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
from typing import Union

from core.settings import settings
import openai
from fastapi import FastAPI

app = FastAPI()
openai.api_key = settings.OPENAI_API_KEY

from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Messages:
    def __init__(self):
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

    def add_message(self, message):
        self.messages.append(message)

    def get_messages(self):
        return self.messages

    def get_last_message(self):
        return self.messages[-1]

    def add_assistant_message(self, response):
        message = response['choices'][0]['message']['content']
        self.messages.append({"role": "assistant", "content": message})


msgs = Messages()

@app.post("/chat")
async def chat(q: str):
    msgs.add_message({"role": "user", "content": q})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=msgs.get_messages(),
    )
    msgs.add_assistant_message(response)
    return {"response": response}

@app.post("/train")
async def train(q: str):
    msgs.add_message({"role": "system", "content": q})

    return {"response": "ok"}


@app.get("/show-input")
def show_input():
    return {"messages": msgs.get_messages()}  

# Planner
# -------------------------------------------------

@app.post("/plan")
async def plan(q: str):
    msgs.add_message({"role": "user", "content": q}) 

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=msgs.get_messages(),
    )
    msgs.add_assistant_message(response)
    return {"response": response}


# Image Generation
# -------------------------------------------------

@app.post("/image")
async def image(q: str):
    response = openai.Image.create(
        prompt=q,
        n=1,
        size="1024x1024"
    )

    image_url = response['data'][0]['url']

    return {"response": image_url}

# Fine Tuning
# -------------------------------------------------

@app.post("/fine-tune")
async def fine_tune(path: str):
    openai.FineTunes.create(
        training_file=open(path, "rb"),
        purpose="fine-tune",
    )
    
    try:
        openai.FineTuningJob.create(training_file="training_ai", model="gpt-3.5-turbo")
    except Exception:
        return {"response": "error"}
    
    return {
        "response": openai.FineTuningJob.list(), 
        "state": openai.FineTuningJob.retrieve("training_ai"),
        "events": openai.FineTuningJob.list_events(id="training_ai"),
    } 


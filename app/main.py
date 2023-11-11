from core.settings import settings
import openai
from fastapi import FastAPI

app = FastAPI()
openai.api_key = settings.OPENAI_API_KEY

from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

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

# Utilities
# -------------------------------

class Messages:
    def __init__(self):
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

    def add_message(self, message):
        self.messages.append(message)

    def add_user_message(self, message):
        self.add_message({"role": "user", "content": message})

    def add_system_message(self, message):
        self.add_message({"role": "system", "content": message})

    def get_messages(self):
        return self.messages

    def get_last_message(self):
        return self.messages[-1]

    def add_assistant_message(self, response):
        message = response["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": message})


class PlannerMessages(Messages):
    def __init__(self):
        super().__init__()
        self.add_system_message("You are a helpful planner.")
        self.add_system_message("You help me plan my day.")
        self.add_system_message("Ok, I will do the following activities:")

    def format_time(self, time: str):
        # time format '07/11/2023 15:30:00'
        try:
            datetime_object = datetime.strptime(time, "%d/%m/%Y %H:%M:%S")
            iso_datetime_string = datetime_object.isoformat()
            print(f"The ISO format datetime is: {iso_datetime_string}")
        except ValueError:
            print(f"The datetime {time} is not in the correct format.")

        return iso_datetime_string

    def make_activity(self, activity: str, time: str):
        time = self.format_time(time)
        activity = activity.lower()
        self.add_user_message(f"Ok, I will {activity} at {time}.")

    def get_daily_routine(self):
        pass

    def make_daily_routine(seld):
        pass


msgs = Messages()
planner_msgs = PlannerMessages()

# API
# -----------------------------------------------------
# _____________________________________________________


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


@app.get("/show-planner-input")
def show_planner_input():
    return {"messages": planner_msgs.get_messages()}


# Planner
# -------------------------------------------------


@app.post("/plan")
async def plan(q: str):
    planner_msgs.add_message({"role": "user", "content": q})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=planner_msgs.get_messages(),
    )
    planner_msgs.add_assistant_message(response)
    return {"response": response}


@app.post("/add/activity")
async def activity(q: str, t: str):
    planner_msgs.make_activity(activity=q, time=t)
    return {"response": planner_msgs.get_messages()}


# Image Generation
# -------------------------------------------------


@app.post("/image")
async def image(q: str):
    response = openai.Image.create(prompt=q, n=1, size="1024x1024")

    image_url = response["data"][0]["url"]

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

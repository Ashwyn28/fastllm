
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    OPENAI_API_KEY: str = ""

    class Config:
        env_file = "../.env"
        env_file_encoding = "utf-8"

settings = Settings()

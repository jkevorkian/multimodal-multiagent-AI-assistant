from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "multimodal-multiagent-ai-assistant"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_prefix="MMAA_", extra="ignore")


settings = Settings()


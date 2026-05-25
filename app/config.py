from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite+aiosqlite:///./smartroute.db"
    SECRET_KEY: str = "change-me-in-production-use-a-real-secret"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Circuit breaker settings
    CB_FAILURE_THRESHOLD: int = 5
    CB_RECOVERY_TIMEOUT: int = 30
    CB_HALF_OPEN_MAX_CALLS: int = 2

    # Model call settings
    MODEL_CALL_TIMEOUT: float = 60.0
    MODEL_CALL_MAX_RETRIES: int = 3
    MODEL_CALL_RETRY_WAIT_MIN: float = 1.0
    MODEL_CALL_RETRY_WAIT_MAX: float = 10.0

    class Config:
        env_file = ".env"


settings = Settings()

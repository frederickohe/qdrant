from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    qdrant_url: str = "http://localhost:6333"
    embeddings_base_url: str = "http://localhost:80"
    rag_collection_name: str = "autobus_conversations"
    vector_size: int = 384
    rag_api_key: str = ""
    embed_timeout_s: float = 60.0
    qdrant_timeout_s: float = 30.0


settings = Settings()

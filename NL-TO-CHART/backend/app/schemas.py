from pydantic import BaseModel


class ConnectionRequest(BaseModel):
    host: str
    port: int = 5432
    user: str
    password: str
    dbname: str
    dialect: str = "postgresql"


class ChartPromptRequest(BaseModel):
    prompt: str


class ChartConfig(BaseModel):
    chart_type: str
    x_field: str
    y_field: str
    title: str
    data: list[dict]

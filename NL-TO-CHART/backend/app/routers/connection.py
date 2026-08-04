from fastapi import APIRouter, HTTPException
from app.schemas import ConnectionRequest
from app.db.connection import connect, DBState

router = APIRouter(prefix="/connection", tags=["connection"])


@router.post("")
def create_connection(body: ConnectionRequest):
    try:
        connect(
            host=body.host,
            port=body.port,
            user=body.user,
            password=body.password,
            dbname=body.dbname,
            dialect=body.dialect,
        )
        return {"status": "connected"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/schema")
def get_schema():
    if DBState.schema_text is None:
        raise HTTPException(status_code=400, detail="No database connected")
    return {"schema": DBState.schema_text}

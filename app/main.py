from fastapi import FastAPI

from backend.services.logger_config import logger
from backend.services.init.init_db import init_db
from backend.services.init.init_ch import init_ch
from backend.services.postgresql import connect_pg
from backend.routes import post, comment, admin

app = FastAPI()

app.include_router(post.router)
app.include_router(comment.router)
app.include_router(admin.router)


@app.on_event("startup")
async def startup():
    pg_pool = await connect_pg()
    await init_db(pg_pool)
    await init_ch()
    logger.info("Инициализация прошла успешно")


@app.get("/")
async def root():
    return {"message": "FastAPI работает!"}

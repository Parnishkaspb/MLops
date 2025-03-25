from fastapi import FastAPI
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from backend.routes import post, comment
from backend.services.init.init_db import init_db
from backend.services.init.init_ch import init_ch
from backend.services.postgresql import connect_pg, pg_pool

app = FastAPI()

app.include_router(post.router)
app.include_router(comment.router)

@app.on_event("startup")
async def startup():
    pg_pool = await connect_pg()
    await init_db(pg_pool)
    await init_ch()


@app.get("/")
async def root():
    return {"message": "FastAPI работает!"}

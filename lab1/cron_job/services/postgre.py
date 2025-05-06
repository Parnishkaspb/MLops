from datetime import datetime
from sqlalchemy.orm import declarative_base, sessionmaker,Session
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine
)
from sqlalchemy.orm import relationship

DATABASE_URL = "postgresql://user:password@postgres:5432/mydb"

engine = create_engine(DATABASE_URL, echo=True)

SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False
)

# Базовый класс моделей
Base = declarative_base()


async def get_db() -> Session:
    with SessionLocal() as session:
        yield session

class Post(Base):
    __tablename__ = 'posts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    ban = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    comments = relationship(
        "Comment",
        back_populates="post",
        cascade="all, delete-orphan",
    )

class Comment(Base):
    __tablename__ = 'comments'

    id = Column(Integer, primary_key=True, autoincrement=True)
    post_id = Column(Integer, ForeignKey('posts.id'), nullable=False)
    text = Column(Text, nullable=False)
    ban = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    post = relationship("Post", back_populates="comments")

def store_comment(data: dict) -> Comment:
    with SessionLocal() as db: 
        comment = Comment(post_id=data["post_id"], text=data["text"])
        db.add(comment)
        db.commit()
        db.refresh(comment)
        return comment
    

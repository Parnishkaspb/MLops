from fastapi import APIRouter
from backend.schemas.comment import CommentAdminOut
from backend.cruds import comment_crud

router = APIRouter(prefix="/admin/comments", tags=["AdminComments"])


@router.get("/", response_model=list[CommentAdminOut])
async def get_all_admin_comment():
    rows = await comment_crud.get_all_admin_comment()
    return [CommentAdminOut(text=row) for row in rows]

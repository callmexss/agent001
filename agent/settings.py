from pathlib import Path

_CUR = Path(__file__).parent
PROJECT_ROOT = _CUR.parent
CHAT_HISTORY_PATH = PROJECT_ROOT / "chat_history"
DB_PATH = PROJECT_ROOT / ".vector_db"
PAPERS_PATH = PROJECT_ROOT / "data/papers"
WECHAT_POST_PATH = PROJECT_ROOT / "data/articles"


for path in [CHAT_HISTORY_PATH, PAPERS_PATH, WECHAT_POST_PATH]:
    path.mkdir(parents=True, exist_ok=True)

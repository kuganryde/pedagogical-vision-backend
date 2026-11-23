import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api.server import app

if __name__ == "__main__":
    import uvicorn
    from src.config import settings
    uvicorn.run("src.api.server:app", host=settings.server.host, port=settings.server.port, reload=settings.server.debug)

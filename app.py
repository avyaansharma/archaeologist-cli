import os
import sys
import uvicorn

# Ensure the root directory is on the python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from archaeologist.web.server import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting Codebase History Analyzer on http://0.0.0.0:{port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

import importlib.util
import os
import sys
from pathlib import Path

# Resolve directories
project_dir = Path(__file__).resolve().parent
rag_dir = project_dir / "RAGChatBot"
sys.path.insert(0, str(rag_dir))
os.chdir(rag_dir)

# Import demo from RAGChatBot/app.py without root naming collision
spec = importlib.util.spec_from_file_location("rag_app", rag_dir / "app.py")
rag_app = importlib.util.module_from_spec(spec)
sys.modules["rag_app"] = rag_app
spec.loader.exec_module(rag_app)
demo = rag_app.demo

if __name__ == "__main__":
    share = "--share" in sys.argv or os.environ.get("SHARE_LIVE", "false").lower() in ("true", "1")
    print(f"Starting Patient Health RAG Chatbot on http://127.0.0.1:7860 (share={share}) ...")
    demo.launch(share=share, allowed_paths=[str(project_dir), str(rag_dir), str(project_dir / "BulkOutput")])

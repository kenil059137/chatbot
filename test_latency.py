import time
import os
import sys
import ssl

# SSL bypass for college WiFi
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

# Add project root to sys path to import backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.services.multi_agent_rag import rewrite_question, retrieve_with_scores
from backend.services.vector_store import embedding

def test_latency():
    print("Testing latency with SSL bypass...")
    
    # 1. Test embedding
    t0 = time.time()
    vector = embedding.embed_query("What is the B.Tech fee structure?")
    t1 = time.time()
    print(f"Embedding time: {t1 - t0:.3f} s")
    
    # 2. Test retrieval
    t2 = time.time()
    docs, conf = retrieve_with_scores("What is the B.Tech fee structure?")
    t3 = time.time()
    print(f"Retrieval time: {t3 - t2:.3f} s")
    
    # 3. Test question rewrite
    t4 = time.time()
    rw = rewrite_question("What about BCA?", "User asked about B.Tech fee structure before.")
    t5 = time.time()
    print(f"Rewrite time: {t5 - t4:.3f} s")

if __name__ == "__main__":
    test_latency()

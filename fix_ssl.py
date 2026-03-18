import os
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sentence_transformers import SentenceTransformer

# This will download and cache the model locally
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model downloaded successfully!")
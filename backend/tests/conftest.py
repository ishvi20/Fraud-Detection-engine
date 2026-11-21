import os
import sys

# Make `import engine` / `import app` work, and keep the API tests deterministic (no background simulator traffic).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["FRAUD_SIMULATE"] = "0"

# config/ports.py
import os

# ======================================================
# HOST & PORT CONFIGURATION FOR LOCALHOST AND SERVERS
# ======================================================

# Host address to bind the server to.
# '127.0.0.1' is for local testing.
# '0.0.0.0' allows external connections (useful for cloud/dedicated servers).
HOST = os.getenv("COMP_CALC_HOST", "127.0.0.1")

# Port to run the application on.
# Default is 8080. You can easily change this to 80, 443, or any other port.
PORT = int(os.getenv("COMP_CALC_PORT", 8080))

# Debug mode for FastAPI (automatic reload, interactive docs)
DEBUG = os.getenv("COMP_CALC_DEBUG", "True").lower() == "true"

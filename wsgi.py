"""
wsgi.py - Production WSGI Entry Point for Loan Prediction System.

Can be run with:
  Waitress (Cross-platform / Windows & Linux):
    python wsgi.py
  or:
    waitress-serve --port=5000 wsgi:app

  Gunicorn (Linux / macOS):
    gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
"""

import os
from app import app

if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Serving production WSGI server via Waitress on http://127.0.0.1:{port}")
    serve(app, host="127.0.0.1", port=port)

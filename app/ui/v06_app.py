"""Canonical Streamlit UI.

This file intentionally delegates to the modern product UI so existing commands
continue to work:

    streamlit run app/ui/v06_app.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.ui.product_app import main


if __name__ == "__main__":
    main()

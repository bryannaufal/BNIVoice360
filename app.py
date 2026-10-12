"""Streamlit entry point. Run with: streamlit run app.py"""

import sys

for mod in list(sys.modules):
    if mod.startswith("bnivoice360"):
        del sys.modules[mod]

from bnivoice360.application import main

if __name__ == "__main__":
    main()


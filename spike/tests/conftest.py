import sys
from pathlib import Path

# The spike modules import each other by bare name, matching how extract.py is
# run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

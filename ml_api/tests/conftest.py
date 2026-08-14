import os
import sys

# Make the serving/training modules importable as top-level packages in tests.
PROJECT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "project")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

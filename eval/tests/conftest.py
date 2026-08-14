import os
import sys

# Make eval/backtest.py importable as a top-level module in tests.
EVAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

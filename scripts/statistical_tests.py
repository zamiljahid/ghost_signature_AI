"""
scripts/statistical_tests.py — Runner for the McNemar (Table VI) + CI suite.
Thin wrapper that executes the project-root statistical_tests.py so the
documented command `python scripts/statistical_tests.py` works. All logic
(including Issue 4's run_mcnemar_suite) lives in the root module.
"""
import os
import sys
import runpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

if __name__ == "__main__":
    runpy.run_path(os.path.join(ROOT, "statistical_tests.py"), run_name="__main__")

"""Run a selected experiment with the current Python interpreter."""
import argparse
from pathlib import Path
import subprocess
import sys

SCRIPTS={
    'audit':'10_model_evaluation.py',
    'rf':'08_random_forest.py',
    'hybrid':'12_hybrid_comparison.py',
    'compare':'13_nested_comparison.py',
    'screening':'15_screening_comparison.py',
}

def main():
    parser=argparse.ArgumentParser(description='EEG experiments: saved-model audit, RF, hybrid, or nested comparison')
    parser.add_argument('experiment',choices=SCRIPTS)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    subprocess.run([sys.executable,str(root/SCRIPTS[args.experiment])],cwd=root,check=True)

if __name__=='__main__':main()

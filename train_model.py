import os
import sys

# Support executing from root folder D:\PROJECTS\fraud_detection
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SUB_DIR = os.path.join(CURRENT_DIR, "fraud_detection")
sys.path.insert(0, SUB_DIR)

from model.train_model import train_and_evaluate_all

if __name__ == "__main__":
    train_and_evaluate_all()

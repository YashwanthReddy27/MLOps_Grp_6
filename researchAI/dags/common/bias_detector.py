import os
import json
from pathlib import Path

from common.bias_detection_fair_learn import FairlearnBiasDetector

class BiasDetector:
    """
    Bias Detector using Fairlearn library
    """

    def __init__(self, data_path: str, output_dir: str,  data_type: str):
        """
        Initialize Fairlearn bias detector

        Args:
            data_path: Path to the data file
            data_type: Type of data - "arxiv" or "news"
        """
        self.data_path = data_path
        self.data_type = data_type
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_data(self):
        """
        Load data from the specified path
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        return os.listdir(self.data_path)

    def detect_bias(self):
        """
        Detect bias using Fairlearn

        Args:
            sensitive_feature: The sensitive attribute to analyze
            label: The target label for fairness analysis
        """
        files = self.load_data()
        print(f"Detecting bias in files: {files}")
        for file in files:
            if file.startswith(self.data_type) and file.endswith(".json"):
                print(f"Processing file: {file}")
                with open(self.data_path + file, "r") as f:
                    data = json.load(f)
                    detector = FairlearnBiasDetector(data=data, data_type=self.data_type)
                    detector.detect_all_biases()
                    detector.save_report(output_path=self.output_dir + f"/{file.replace('.json', '')}_bias_report.json")
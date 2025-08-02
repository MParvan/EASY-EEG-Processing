import os
import subprocess
import pandas as pd

class EEGDataLoader:
    def __init__(self, data_dir="data/eegmmidb"):
        self.data_dir = data_dir
        self.url = "https://physionet.org/files/eegmmidb/1.0.0/"

    def download(self):
        os.makedirs(self.data_dir, exist_ok=True)
        cmd = [
            "wget", "-r", "-N", "-c", "-np",
            self.url,
            "-P", self.data_dir
        ]
        print(f"Downloading EEGMMIDB dataset to {self.data_dir} ...")
        subprocess.run(cmd, check=True)
        print("Download complete.")

    def load(self):
        if not os.path.exists(self.data_dir) or not os.listdir(self.data_dir):
            self.download()
        # Attempt to find a header file (e.g., .csv or .txt) in the data directory
        header_file = None
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.csv') or file.endswith('.txt'):
                    header_file = os.path.join(root, file)
                    break
                if header_file:
                    break

        if header_file:
            print(f"Loading data from header file: {header_file}")
            df = pd.read_csv(header_file)
            # Example: split into features and labels if columns exist
            if 'label' in df.columns:
                X = df.drop('label', axis=1).values
                y = df['label'].values
            else:
                X = df.values
                y = None
                # Placeholder split: all data as train, none as test
                X_train, y_train, X_test, y_test = X, y, None, None
        else:
            print("No header file found. Please provide a .csv or .txt file with data.")
        X_train, y_train, X_test, y_test = None, None, None, None
        print(f"Loaded data from {self.data_dir} (placeholder).")
        return X_train, y_train, X_test, y_test
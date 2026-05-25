import os
import pandas as pd
import uuid
import json
from datetime import datetime
import tempfile
import atexit
import shutil

class TempDataManager:
    """Manages temporary data storage for the application."""
    
    def __init__(self, temp_dir="temp_data"):
        self.temp_dir = temp_dir
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        # Register cleanup function to run when the program exits
        atexit.register(self.cleanup_temp_files)
    
    def save_dataframe(self, df, prefix="dataframe"):
        """Save a dataframe to a temporary file and return the file path."""
        # Ensure directory exists before saving (defensive check)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        file_id = str(uuid.uuid4())
        filename = f"{prefix}_{file_id}.pkl"
        filepath = os.path.join(self.temp_dir, filename)
        
        # Save dataframe as pickle file (preserves all data types)
        df.to_pickle(filepath)
        
        return filepath
    
    def load_dataframe(self, filepath):
        """Load a dataframe from a temporary file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Temporary file not found: {filepath}")
        
        return pd.read_pickle(filepath)
    
    def save_analysis_results(self, results, prefix="analysis"):
        """Save analysis results to a temporary file."""
        # Ensure directory exists before saving (defensive check)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        file_id = str(uuid.uuid4())
        filename = f"{prefix}_{file_id}.json"
        filepath = os.path.join(self.temp_dir, filename)
        
        # Save results as JSON
        with open(filepath, 'w') as f:
            json.dump(results, f, default=str, indent=2)
        
        return filepath
    
    def load_analysis_results(self, filepath):
        """Load analysis results from a temporary file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def cleanup_temp_files(self):
        """Clean up all temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def cleanup_old_files(self, hours_old=1):
        """Clean up temporary files older than specified hours."""
        if not os.path.exists(self.temp_dir):
            return
        
        cutoff_time = datetime.now().timestamp() - (hours_old * 3600)
        
        for filename in os.listdir(self.temp_dir):
            filepath = os.path.join(self.temp_dir, filename)
            if os.path.isfile(filepath):
                file_time = os.path.getctime(filepath)
                if file_time < cutoff_time:
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass  # Skip if file is in use

# Create a global instance
temp_manager = TempDataManager()
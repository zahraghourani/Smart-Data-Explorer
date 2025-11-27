import pandas as pd

class DataStorage:
    """Shared dataset accessible across all GUI pages."""
    
    df = None       # holds the dataframe
    loaded = False  # track if a file was loaded

    @classmethod
    def load_csv(cls, path):
        """Loads CSV and stores it globally."""
        try:
            cls.df = pd.read_csv(path)
            cls.loaded = True
            return True
        except Exception as e:
            print("Error loading CSV:", e)
            cls.loaded = False
            cls.df = None
            return False

    @classmethod
    def clear(cls):
        """Clears dataset."""
        cls.df = None
        cls.loaded = False

    @classmethod
    def get(cls):
        """Returns DataFrame or None."""
        return cls.df

from pathlib import Path
import pandas as pd

class Olist:
    """
    The Olist class provides methods to interact with Olist's e-commerce data.

    Methods:
        get_data():
            Loads and returns a dictionary where keys are dataset names (e.g., 'sellers', 'orders')
            and values are pandas DataFrames loaded from corresponding CSV files.

        ping():
            Prints "pong" to confirm the method is callable.
    """
    def get_data(self):
        """
        This function returns a Python dict.
        Its keys should be 'sellers', 'orders', 'order_items' etc...
        Its values should be pandas.DataFrames loaded from csv files
        """
        # 1. CSV dosyalarının olduğu dizini belirle
        csv_path = Path("~/.workintech/olist/data/csv").expanduser()

        # 2. Sadece CSV dosyalarını seç
        file_paths = [f for f in csv_path.iterdir() if f.is_file() and f.suffix == '.csv']

        # 3. Dosya isimlerinden temiz anahtarlar oluştur
        key_names = [
            f.name.replace('olist_', '').replace('_dataset.csv', '').replace('.csv', '') 
            for f in file_paths
        ]

        # 4. DataFrame'leri içeren sözlüğü oluştur
        data = {}
        for key, path in zip(key_names, file_paths):
            data[key] = pd.read_csv(path)

        # 5. Sözlüğü döndür (Çok önemli!)
        return data

    def ping(self):
        """
        You call ping I print pong.
        """
        print("pong")
import os
import pandas as pd
from pathlib import Path
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
RAW_DATA_PATH = Path("../data/raw/")
PROCESSED_DATA_PATH = Path("../data/processed/")

class DataPreprocessor:
    """
    A class to preprocess and combine transaction and transfer data files.
    
    Attributes:
        raw_data_path (Path): Path to raw data directory
        clients (pd.DataFrame): Loaded clients data
        transactions_files (List[str]): List of transaction file names
        transfers_files (List[str]): List of transfer file names
        data_transactions (pd.DataFrame): Combined transactions data
        data_transfers (pd.DataFrame): Combined transfers data
    """
    
    def __init__(self, raw_data_path: Path):
        """
        Initialize the DataPreprocessor.
        
        Args:
            raw_data_path (Path): Path to the raw data directory
        """
        self.raw_data_path = Path(raw_data_path)
        self.clients: Optional[pd.DataFrame] = None
        self.transactions_files: List[str] = []
        self.transfers_files: List[str] = []
        self.data_transactions: Optional[pd.DataFrame] = None
        self.data_transfers: Optional[pd.DataFrame] = None
        
        # Validate path exists
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data path does not exist: {self.raw_data_path}")
    
    def load_clients(self, filename: str = "clients.csv") -> pd.DataFrame:
        """
        Load clients data from CSV file.
        
        Args:
            filename (str): Name of the clients file
            
        Returns:
            pd.DataFrame: Loaded clients data
            
        Raises:
            FileNotFoundError: If the clients file doesn't exist
        """
        clients_path = self.raw_data_path / filename
        
        if not clients_path.exists():
            raise FileNotFoundError(f"Clients file not found: {clients_path}")
            
        try:
            self.clients = pd.read_csv(clients_path)
            logger.info(f"Successfully loaded clients data: {len(self.clients)} records")
            return self.clients
        except Exception as e:
            logger.error(f"Error loading clients file: {e}")
            raise
    
    def _discover_files(self, pattern: str) -> List[str]:
        """
        Discover CSV files matching a pattern in the raw data directory.
        
        Args:
            pattern (str): Pattern to match in filename
            
        Returns:
            List[str]: List of matching file names
        """
        matching_files = []
        try:
            for file in self.raw_data_path.glob("*.csv"):
                if pattern in file.name:
                    matching_files.append(file.name)
            logger.info(f"Found {len(matching_files)} {pattern} files")
            return matching_files
        except Exception as e:
            logger.error(f"Error discovering {pattern} files: {e}")
            return []
    
    def discover_transactions(self) -> List[str]:
        """
        Discover all transaction CSV files.
        
        Returns:
            List[str]: List of transaction file names
        """
        self.transactions_files = self._discover_files("transactions")
        return self.transactions_files
    
    def discover_transfers(self) -> List[str]:
        """
        Discover all transfer CSV files.
        
        Returns:
            List[str]: List of transfer file names
        """
        self.transfers_files = self._discover_files("transfers")
        return self.transfers_files
    
    def _combine_csv_files(self, file_list: List[str], data_type: str) -> pd.DataFrame:
        """
        Combine multiple CSV files into a single DataFrame.
        
        Args:
            file_list (List[str]): List of CSV file names to combine
            data_type (str): Type of data being combined (for logging)
            
        Returns:
            pd.DataFrame: Combined data
        """
        if not file_list:
            logger.warning(f"No {data_type} files found to combine")
            return pd.DataFrame()
        
        combined_data = []
        for file in file_list:
            file_path = self.raw_data_path / file
            try:
                df = pd.read_csv(file_path)
                # Add source file column for traceability
                df['source_file'] = file
                combined_data.append(df)
                logger.info(f"Loaded {len(df)} records from {file}")
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
                continue
        
        if combined_data:
            result = pd.concat(combined_data, ignore_index=True)
            logger.info(f"Combined {data_type} data: {len(result)} total records from {len(combined_data)} files")
            return result
        else:
            logger.warning(f"No valid {data_type} data found")
            return pd.DataFrame()
    
    def combine_transactions(self) -> pd.DataFrame:
        """
        Combine all transaction files into a single DataFrame.
        
        Returns:
            pd.DataFrame: Combined transactions data
        """
        if not self.transactions_files:
            self.discover_transactions()
        
        self.data_transactions = self._combine_csv_files(self.transactions_files, "transactions")
        return self.data_transactions
    
    def combine_transfers(self) -> pd.DataFrame:
        """
        Combine all transfer files into a single DataFrame.
        
        Returns:
            pd.DataFrame: Combined transfers data
        """
        if not self.transfers_files:
            self.discover_transfers()
        
        self.data_transfers = self._combine_csv_files(self.transfers_files, "transfers")
        return self.data_transfers
    
    def process_all_data(self) -> dict:
        """
        Load and process all data files.
        
        Returns:
            dict: Dictionary containing all processed data
        """
        logger.info("Starting data processing...")
        
        # Load clients
        try:
            self.load_clients()
        except Exception as e:
            logger.error(f"Failed to load clients: {e}")
        
        # Discover and combine transactions
        self.discover_transactions()
        self.combine_transactions()
        
        # Discover and combine transfers
        self.discover_transfers()
        self.combine_transfers()
        
        logger.info("Data processing completed")
        
        return {
            'clients': self.clients,
            'transactions': self.data_transactions,
            'transfers': self.data_transfers
        }
    
    def get_data_summary(self) -> dict:
        """
        Get a summary of all loaded data.
        
        Returns:
            dict: Summary statistics for each dataset
        """
        summary = {}
        
        if self.clients is not None:
            summary['clients'] = {
                'records': len(self.clients),
                'columns': list(self.clients.columns)
            }
        
        if self.data_transactions is not None:
            summary['transactions'] = {
                'records': len(self.data_transactions),
                'columns': list(self.data_transactions.columns),
                'files_processed': len(self.transactions_files)
            }
        
        if self.data_transfers is not None:
            summary['transfers'] = {
                'records': len(self.data_transfers),
                'columns': list(self.data_transfers.columns),
                'files_processed': len(self.transfers_files)
            }
        
        return summary


def main():
    """Main function to demonstrate usage."""
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor(RAW_DATA_PATH)
        
        # Process all data
        data = preprocessor.process_all_data()
        
        
        # Display summary
        summary = preprocessor.get_data_summary()
        print("\nData Processing Summary:")
        print("=" * 50)
        for data_type, stats in summary.items():
            print(f"\n{data_type.upper()}:")
            print(f"  Records: {stats['records']}")
            print(f"  Columns: {stats['columns']}")
            if 'files_processed' in stats:
                print(f"  Files processed: {stats['files_processed']}")
        
        # Display first few rows if data exists
        if preprocessor.data_transactions is not None and not preprocessor.data_transactions.empty:
            print(f"\nFirst 5 transaction records:")
            print(preprocessor.data_transactions.head())
        
        if preprocessor.data_transfers is not None and not preprocessor.data_transfers.empty:
            print(f"\nFirst 5 transfer records:")
            print(preprocessor.data_transfers.head())
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
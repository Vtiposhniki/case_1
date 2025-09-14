import os
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union
import logging
from dataclasses import dataclass


# Configuration
@dataclass
class Config:
    """Configuration settings for data preprocessing."""
    raw_data_path: Path = Path(__file__).resolve().parent.parent / "data" / "raw"
    processed_data_path: Path = Path(__file__).resolve().parent.parent / "data" / "processed"
    clients_filename: str = "clients.csv"
    transactions_pattern: str = "transactions"
    transfers_pattern: str = "transfers"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


class DataPreprocessorError(Exception):
    """Custom exception for DataPreprocessor errors."""
    pass


class DataPreprocessor:
    """
    A class to preprocess financial data including clients, transactions, and transfers.

    This class handles loading, combining, and exporting various CSV files containing
    financial data while providing comprehensive error handling and logging.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the DataPreprocessor.

        Args:
            config: Configuration object containing paths and settings

        Raises:
            DataPreprocessorError: If required directories don't exist or can't be created
        """
        self.config = config or Config()
        self.logger = setup_logging()

        # Initialize file lists
        self._transactions_files: Optional[List[str]] = None
        self._transfers_files: Optional[List[str]] = None

        # Initialize data storage
        self._data_transactions: Optional[pd.DataFrame] = None
        self._data_transfers: Optional[pd.DataFrame] = None
        self._clients_data: Optional[pd.DataFrame] = None

        self._validate_and_create_directories()

    def _validate_and_create_directories(self) -> None:
        """Validate that required directories exist and create processed directory if needed."""
        if not self.config.raw_data_path.exists():
            # Создаем raw директорию, если она не существует
            try:
                self.config.raw_data_path.mkdir(parents=True, exist_ok=True)
                self.logger.warning(f"Created raw data directory: {self.config.raw_data_path}")
                self.logger.warning("Please add your data files to this directory")
            except OSError as e:
                raise DataPreprocessorError(f"Could not create raw data directory: {e}")

        # Create processed directory if it doesn't exist
        try:
            self.config.processed_data_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Ensured processed data directory exists: {self.config.processed_data_path}")
        except OSError as e:
            raise DataPreprocessorError(f"Could not create processed data directory: {e}")

    def load_clients(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        Load clients data from CSV file.

        Args:
            filename: Name of the clients file (defaults to config value)

        Returns:
            DataFrame containing clients data

        Raises:
            DataPreprocessorError: If file doesn't exist or can't be loaded
        """
        if filename is None:
            filename = self.config.clients_filename

        file_path = self.config.raw_data_path / filename

        if not file_path.exists():
            # Создаем пример файла clients.csv, если он не существует
            try:
                example_data = pd.DataFrame({
                    'client_id': [1, 2, 3],
                    'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
                    'email': ['john@example.com', 'jane@example.com', 'bob@example.com']
                })
                example_data.to_csv(file_path, index=False)
                self.logger.warning(f"Created example clients file: {file_path}")
                self.logger.warning("Please replace with your actual clients data")
            except Exception as e:
                raise DataPreprocessorError(
                    f"Clients file {file_path} does not exist and could not create example: {e}")

        try:
            self._clients_data = pd.read_csv(file_path)
            # self.logger.info(f"Successfully loaded {len(self._clients_data)} client records from {filename}")
            return self._clients_data.copy()
        except Exception as e:
            error_msg = f"Error loading clients file {filename}: {e}"
            self.logger.error(error_msg)
            raise DataPreprocessorError(error_msg) from e

    def _discover_files(self, pattern: str) -> List[str]:
        """
        Discover CSV files matching a specific pattern.

        Args:
            pattern: Pattern to search for in filenames

        Returns:
            List of matching filenames
        """
        matching_files = []
        try:
            for file in self.config.raw_data_path.glob("*.csv"):
                if pattern.lower() in file.name.lower():
                    matching_files.append(file.name)

            # self.logger.info(f"Found {len(matching_files)} files matching pattern '{pattern}': {matching_files}")
            return sorted(matching_files)  # Sort for consistent ordering

        except Exception as e:
            self.logger.error(f"Error discovering files with pattern '{pattern}': {e}")
            return []

    def discover_transactions(self) -> List[str]:
        """
        Discover transaction files.

        Returns:
            List of transaction filenames
        """
        self._transactions_files = self._discover_files(self.config.transactions_pattern)
        return self._transactions_files.copy()

    def discover_transfers(self) -> List[str]:
        """
        Discover transfer files.

        Returns:
            List of transfer filenames
        """
        self._transfers_files = self._discover_files(self.config.transfers_pattern)
        return self._transfers_files.copy()

    def _create_example_data_file(self, filename: str, data_type: str) -> None:
        """Create example data file for demonstration purposes."""
        file_path = self.config.raw_data_path / filename

        if data_type == "transactions":
            example_data = pd.DataFrame({
                'transaction_id': [1, 2, 3],
                'client_id': [1, 2, 1],
                'amount': [100.50, 75.25, 200.00],
                'date': ['2024-01-15', '2024-01-16', '2024-01-17']
            })
        elif data_type == "transfers":
            example_data = pd.DataFrame({
                'transfer_id': [1, 2, 3],
                'client_id': [1, 3, 2],
                'amount': [50.00, 125.75, 80.50],
                'date': ['2024-01-15', '2024-01-16', '2024-01-17']
            })
        else:
            return

        try:
            example_data.to_csv(file_path, index=False)
            self.logger.warning(f"Created example {data_type} file: {file_path}")
            self.logger.warning(f"Please replace with your actual {data_type} data")
        except Exception as e:
            self.logger.error(f"Could not create example {data_type} file: {e}")

    def _combine_files(self, file_list: List[str], data_type: str) -> pd.DataFrame:
        """
        Combine multiple CSV files into a single DataFrame.

        Args:
            file_list: List of filenames to combine
            data_type: Type of data being combined (for logging)

        Returns:
            Combined DataFrame
        """
        if not file_list:
            self.logger.warning(f"No {data_type} files found! Creating example file...")
            # Создаем пример файла
            example_filename = f"example_{data_type}.csv"
            self._create_example_data_file(example_filename, data_type)

            # Пытаемся снова найти файлы
            if data_type == "transactions":
                self.discover_transactions()
                file_list = self._transactions_files or []
            else:
                self.discover_transfers()
                file_list = self._transfers_files or []

            if not file_list:
                self.logger.warning(f"No {data_type} files available after creating example")
                return pd.DataFrame()

        combined_data = []
        failed_files = []

        for filename in file_list:
            file_path = self.config.raw_data_path / filename
            try:
                df = pd.read_csv(file_path)
                if df.empty:
                    self.logger.warning(f"File {filename} is empty, skipping")
                    continue

                combined_data.append(df)
                #  self.logger.info(f"Loaded {len(df)} records from {filename}")

            except Exception as e:
                error_msg = f"Error loading {filename}: {e}"
                self.logger.error(error_msg)
                failed_files.append(filename)
                continue

        if failed_files:
            self.logger.warning(f"Failed to load {len(failed_files)} {data_type} files: {failed_files}")

        if combined_data:
            result = pd.concat(combined_data, ignore_index=True)
            self.logger.info(
                f"Combined {data_type} data: {len(result)} total records from "
                f"{len(combined_data)} files (out of {len(file_list)} attempted)"
            )
            return result
        else:
            self.logger.warning(f"No valid {data_type} data found")
            return pd.DataFrame()

    def combine_transactions(self) -> pd.DataFrame:
        """
        Combine all transaction files into a single DataFrame.

        Returns:
            Combined transactions DataFrame
        """
        if self._transactions_files is None:
            self.discover_transactions()

        self._data_transactions = self._combine_files(self._transactions_files, "transactions")
        return self._data_transactions.copy() if self._data_transactions is not None else pd.DataFrame()

    def combine_transfers(self) -> pd.DataFrame:
        """
        Combine all transfer files into a single DataFrame.

        Returns:
            Combined transfers DataFrame
        """
        if self._transfers_files is None:
            self.discover_transfers()

        self._data_transfers = self._combine_files(self._transfers_files, "transfers")
        return self._data_transfers.copy() if self._data_transfers is not None else pd.DataFrame()

    def _export_file(self, df: pd.DataFrame, filename: str) -> None:
        """
        Export DataFrame to CSV file.

        Args:
            df: DataFrame to export
            filename: Output filename

        Raises:
            DataPreprocessorError: If export fails
        """
        if df.empty:
            self.logger.warning(f"DataFrame is empty, skipping export of {filename}")
            return

        # Ensure filename has .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'

        file_path = self.config.processed_data_path / filename

        try:
            df.to_csv(file_path, index=False)
            self.logger.info(f"Successfully exported {len(df)} rows to {file_path}")
        except Exception as e:
            error_msg = f"Error exporting to {file_path}: {e}"
            self.logger.error(error_msg)
            raise DataPreprocessorError(error_msg) from e

    def export_all_data(self) -> None:
        """
        Load, combine, and export all data files.

        This method performs the complete preprocessing pipeline:
        1. Loads clients data
        2. Combines transaction files
        3. Combines transfer files
        4. Exports all processed data
        """
        try:
            # Load and process all data
            clients = self.load_clients()
            transactions = self.combine_transactions()
            transfers = self.combine_transfers()

            # Export processed data
            self._export_file(clients, "clients.csv")
            self._export_file(transactions, "combined_transactions.csv")
            self._export_file(transfers, "combined_transfers.csv")

            self.logger.info("Successfully completed data preprocessing and export")

        except Exception as e:
            error_msg = f"Error during data export process: {e}"
            self.logger.error(error_msg)
            raise DataPreprocessorError(error_msg) from e

    def get_summary(self) -> dict:
        """
        Get a summary of processed data.

        Returns:
            Dictionary containing data summary statistics
        """
        summary = {
            'clients_count': len(self._clients_data) if self._clients_data is not None else 0,
            'transactions_count': len(self._data_transactions) if self._data_transactions is not None else 0,
            'transfers_count': len(self._data_transfers) if self._data_transfers is not None else 0,
            'transaction_files': len(self._transactions_files) if self._transactions_files else 0,
            'transfer_files': len(self._transfers_files) if self._transfers_files else 0
        }
        return summary


def start():
    """Main function to run the data preprocessing."""
    try:
        # Create preprocessor with default config
        preprocessor = DataPreprocessor()

        # Run the complete preprocessing pipeline
        preprocessor.export_all_data()

        # Print summary
        summary = preprocessor.get_summary()
        print("\nData Processing Summary:")
        print("-" * 30)
        # for key, value in summary.items():
        #     print(f"{key.replace('_', ' ').title()}: {value}")

    except DataPreprocessorError as e:
        print(f"Data preprocessing error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


# Добавляем возможность запуска напрямую
if __name__ == "__main__":
    start()
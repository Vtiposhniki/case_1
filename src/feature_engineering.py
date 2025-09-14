from dataclasses import dataclass, field
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransactionsConfig:
    """Configuration for transaction categories"""
    CATEGORIES: List[str] = field(default_factory=lambda: [
        "Одежда и обувь", "Продукты питания", "Кафе и рестораны", "Медицина", "Авто",
        "Спорт", "Развлечения", "АЗС", "Кино", "Питомцы", "Книги", "Цветы", "Едим дома",
        "Смотрим дома", "Играем дома", "Косметика и Парфюмерия", "Подарки", "Ремонт дома",
        "Мебель", "Спа и массаж", "Ювелирные украшения", "Такси", "Отели", "Путешествия"
    ])


@dataclass(frozen=True)
class TransfersConfig:
    """Configuration for transfer types"""
    TYPES: List[str] = field(default_factory=lambda: [
        "salary_in", "stipend_in", "family_in", "cashback_in", "refund_in",
        "card_in", "p2p_out", "card_out", "atm_withdrawal", "utilities_out",
        "loan_payment_out", "cc_repayment_out", "installment_payment_out",
        "fx_buy", "fx_sell", "invest_out", "invest_in", "deposit_topup_out",
        "deposit_fx_topup_out", "deposit_fx_withdraw_in", "gold_buy_out",
        "gold_sell_in"
    ])


@dataclass(frozen=True)
class DataPaths:
    """Configuration for data file paths"""
    clients_path: Path = Path("../data/processed/clients.csv")
    transactions_path: Path = Path("../data/processed/combined_transactions.csv")
    transfers_path: Path = Path("../data/processed/combined_transfers.csv")


class FeatureEngineering:
    """Efficient feature engineering for financial data"""

    def __init__(self,
                 transactions_config: Optional[TransactionsConfig] = None,
                 transfers_config: Optional[TransfersConfig] = None,
                 data_paths: Optional[DataPaths] = None):

        # Use default configs if not provided
        self.tx_config = transactions_config or TransactionsConfig()
        self.tr_config = transfers_config or TransfersConfig()
        self.data_paths = data_paths or DataPaths()

        # Define column structure
        self._base_columns = [
            "client_code", "name", "status", "age", "city",
            "avg_monthly_balance_KZT", "total_spending", "tx_count", "avg_tx_amount"
        ]
        self._derived_columns = [
            "travel_spend", "restaurants_spend", "fx_flag", "loan_flag", "high_balance_client"
        ]

        # Initialize dataframes
        self.clients: Optional[pd.DataFrame] = None
        self.transactions: Optional[pd.DataFrame] = None
        self.transfers: Optional[pd.DataFrame] = None
        self.features_df: Optional[pd.DataFrame] = None

    def load_data(self) -> bool:
        """Load all required data files"""
        try:
            logger.info("Loading data files...")

            # Check if files exist
            for path_name, path in [
                ("clients", self.data_paths.clients_path),
                ("transactions", self.data_paths.transactions_path),
                ("transfers", self.data_paths.transfers_path)
            ]:
                if not path.exists():
                    logger.error(f"{path_name} file not found: {path}")
                    return False

            # Load data
            self.clients = pd.read_csv(self.data_paths.clients_path)
            self.transactions = pd.read_csv(self.data_paths.transactions_path)
            self.transfers = pd.read_csv(self.data_paths.transfers_path)

            # Validate data
            if any(df.empty for df in [self.clients, self.transactions, self.transfers]):
                logger.error("One or more data files are empty")
                return False

            logger.info(f"Loaded: {len(self.clients)} clients, "
                        f"{len(self.transactions)} transactions, "
                        f"{len(self.transfers)} transfers")
            return True

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def _prepare_transaction_features(self) -> pd.DataFrame:
        """Efficiently prepare transaction-based features using vectorized operations"""
        logger.info("Processing transaction features...")

        # Group by client and category, sum amounts
        tx_pivot = (self.transactions
                    .groupby(['client_code', 'category'])['amount']
                    .sum()
                    .unstack(fill_value=0))

        # Ensure all categories are present
        for category in self.tx_config.CATEGORIES:
            if category not in tx_pivot.columns:
                tx_pivot[category] = 0

        # Calculate transaction aggregates
        tx_agg = (self.transactions
        .groupby('client_code')['amount']
        .agg(['sum', 'count', 'mean'])
        .rename(columns={
            'sum': 'total_spending',
            'count': 'tx_count',
            'mean': 'avg_tx_amount'
        }))

        # Combine transaction features
        tx_features = pd.concat([tx_agg, tx_pivot], axis=1)
        return tx_features.fillna(0)

    def _prepare_transfer_features(self) -> pd.DataFrame:
        """Efficiently prepare transfer-based features using vectorized operations"""
        logger.info("Processing transfer features...")

        # Group by client and type, sum amounts
        tr_pivot = (self.transfers
                    .groupby(['client_code', 'type'])['amount']
                    .sum()
                    .unstack(fill_value=0))

        # Ensure all transfer types are present
        for transfer_type in self.tr_config.TYPES:
            if transfer_type not in tr_pivot.columns:
                tr_pivot[transfer_type] = 0

        return tr_pivot.fillna(0)

    def _calculate_derived_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features using vectorized operations"""
        logger.info("Calculating derived features...")

        # Travel spend (vectorized)
        travel_columns = ["Такси", "Отели", "Путешествия"]
        available_travel = [col for col in travel_columns if col in features_df.columns]
        features_df["travel_spend"] = features_df[available_travel].sum(axis=1)

        # Restaurant spend
        if "Кафе и рестораны" in features_df.columns:
            features_df["restaurants_spend"] = features_df["Кафе и рестораны"]
        else:
            features_df["restaurants_spend"] = 0

        # FX flag (vectorized)
        fx_columns = ["fx_buy", "fx_sell"]
        available_fx = [col for col in fx_columns if col in features_df.columns]
        if available_fx:
            features_df["fx_flag"] = (features_df[available_fx].sum(axis=1) > 0).astype(int)
        else:
            features_df["fx_flag"] = 0

        # Loan flag
        if "loan_payment_out" in features_df.columns:
            features_df["loan_flag"] = (features_df["loan_payment_out"] > 0).astype(int)
        else:
            features_df["loan_flag"] = 0

        # High balance client flag
        if "avg_monthly_balance_KZT" in features_df.columns:
            features_df["high_balance_client"] = (
                    features_df["avg_monthly_balance_KZT"] > 1_000_000
            ).astype(int)
        else:
            features_df["high_balance_client"] = 0

        return features_df

    def create_features(self) -> bool:
        """Main feature creation pipeline using efficient vectorized operations"""
        try:
            if not self.load_data():
                return False

            logger.info("Starting feature engineering...")

            # Start with client base data
            base_features = self.clients.set_index('client_code')[
                ['name', 'status', 'age', 'city', 'avg_monthly_balance_KZT']
            ].copy()

            # Process transaction features
            tx_features = self._prepare_transaction_features()

            # Process transfer features
            tr_features = self._prepare_transfer_features()

            # Combine all features
            logger.info("Combining features...")
            self.features_df = pd.concat([base_features, tx_features, tr_features], axis=1)

            # Fill missing values for clients with no transactions/transfers
            self.features_df = self.features_df.fillna(0)

            # Calculate derived features
            self.features_df = self._calculate_derived_features(self.features_df)

            # Reset index to make client_code a column
            self.features_df = self.features_df.reset_index()

            # Ensure proper column order
            all_columns = (self._base_columns +
                           self.tx_config.CATEGORIES +
                           self.tr_config.TYPES +
                           self._derived_columns)

            # Reorder columns and fill any missing ones
            for col in all_columns:
                if col not in self.features_df.columns:
                    self.features_df[col] = 0

            self.features_df = self.features_df.reindex(columns=all_columns)

            logger.info(f"Feature engineering completed. Shape: {self.features_df.shape}")
            return True

        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return False

    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the features"""
        if self.features_df is None:
            return {"error": "Features not created yet"}

        return {
            "total_clients": len(self.features_df),
            "total_features": len(self.features_df.columns),
            "high_balance_clients": self.features_df["high_balance_client"].sum(),
            "fx_users": self.features_df["fx_flag"].sum(),
            "loan_users": self.features_df["loan_flag"].sum(),
            "avg_spending": self.features_df["total_spending"].mean(),
            "avg_transaction_count": self.features_df["tx_count"].mean()
        }

    def save_features(self, output_path: str = "../data/processed/features.csv") -> bool:
        """Save features to CSV file"""
        try:
            if self.features_df is None:
                logger.error("No features to save. Run create_features() first.")
                return False

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            self.features_df.to_csv(output_path, index=False)
            logger.info(f"Features saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving features: {e}")
            return False


def main():
    """Main execution function"""
    # Initialize with default configurations
    feature_engineer = FeatureEngineering()

    # Create features
    if not feature_engineer.create_features():
        logger.error("Feature creation failed")
        return 1

    # Display summary
    summary = feature_engineer.get_feature_summary()
    logger.info("Feature Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:,.2f}")
        else:
            logger.info(f"  {key}: {value:,}")

    # Save features
    feature_engineer.save_features()

    # Display first few rows
    print("\nFirst 5 rows of features:")
    print(feature_engineer.features_df.head())

    return 0


if __name__ == "__main__":
    exit(main())
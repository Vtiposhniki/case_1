import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


from src.data_preprocessing import *


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


def test_data_preprocessing():
    logger.info("=" * 60)
    logger.info("STEP 1: DATA PREPROCESSING")
    logger.info("=" * 60)
    try:
        preprocessor = DataPreprocessor()
        preprocessor.export_all_data()
        summary = preprocessor.get_summary()
        logger.debug("✅ Preprocessing completed successfully!")
        for key, value in summary.items():
            logger.info(f"  📊 {key}: {value:,}")
    except Exception as e:
        logger.error(e)
        return False
    finally:
        logger.info("=" * 60)
        logger.info("STEP 2: DATA PREPROCESSING")
        logger.info("=" * 60)
        return True

if __name__ == "__main__":
    test_data_preprocessing()
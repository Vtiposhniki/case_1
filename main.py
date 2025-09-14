import pandas as pd
import logging
from pathlib import Path

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineering
from src.model import ModelManager
from src.push_generator import generate_push_notifications, generate_push_report
from src.scoring_system import BankingPersonalizationSystem, create_scoring_dashboard
from src.utils import save_json

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reports/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Основной пайплайн обработки данных и генерации персональных предложений"""

    try:
        logger.info("Запуск пайплайна персональных банковских предложений")

        # Создаем директории для отчетов
        Path("reports").mkdir(exist_ok=True)
        Path("reports/figures").mkdir(exist_ok=True)

        # 1. Data Preprocessing
        print("🚀 Step 1: Data Preprocessing...")
        logger.info("Начало предобработки данных")

        preprocessor = DataPreprocessor()
        preprocessor.export_all_data()

        logger.info("Предобработка данных завершена")

        # 2. Feature Engineering
        print("🚀 Step 2: Feature Engineering...")
        logger.info("Начало инженерии признаков")

        fe = FeatureEngineering()
        if not fe.create_features():
            raise RuntimeError("❌ Feature engineering failed")
        fe.save_features()

        logger.info("Инженерия признаков завершена")

        # Загружаем подготовленные данные
        print("📊 Loading processed data...")
        logger.info("Загрузка обработанных данных")

        profiles_df = pd.read_csv("data/processed/clients.csv")
        transactions_df = pd.read_csv("data/processed/combined_transactions.csv")
        transfers_df = pd.read_csv("data/processed/combined_transfers.csv")

        logger.info(f"Загружено: {len(profiles_df)} клиентов, "
                    f"{len(transactions_df)} транзакций, "
                    f"{len(transfers_df)} переводов")

        # Базовая валидация данных
        if profiles_df.empty:
            raise ValueError("Файл клиентов пуст")
        if transactions_df.empty:
            logger.warning("Файл транзакций пуст")
        if transfers_df.empty:
            logger.warning("Файл переводов пуст")

        # 3. Model Training / Loading
        print("🚀 Step 3: Training or Loading Model...")
        logger.info("Обучение/загрузка модели")

        model_manager = ModelManager()
        if not model_manager.model_path.exists():
            logger.info("Модель не найдена, начинаем обучение")
            model_manager.train(fe.features_df)
        else:
            logger.info("Загрузка существующей модели")
            model_manager.load()

        # 4. Client Scoring and Personalization
        print("🚀 Step 4: Client Scoring...")
        logger.info("Начало скоринга клиентов")

        scoring_system = BankingPersonalizationSystem()

        # Обрабатываем клиентов (исправляем дублирование)
        offers_df = scoring_system.process_all_clients(profiles_df, transactions_df, transfers_df)

        if offers_df.empty:
            raise ValueError("Не удалось сгенерировать предложения для клиентов")

        logger.info(f"Сгенерировано {len(offers_df)} предложений")

        # 5. Save Results and Generate Analytics
        print("🚀 Step 5: Generating Analytics...")
        logger.info("Сохранение результатов и генерация аналитики")

        scoring_system.save_results(offers_df)

        stats = scoring_system.generate_analytics_report(offers_df)
        save_json(stats, "reports/scoring_summary.json")

        logger.info("Базовая аналитика сохранена")

        # 6. Push Notifications
        print("🚀 Step 6: Generating Push Notifications...")
        logger.info("Генерация пуш-уведомлений")

        # Преобразуем DataFrame в нужный формат для push_generator
        offers_dict = (
            offers_df.groupby("client_code")
            .apply(lambda g: g.to_dict("records"))
            .to_dict()
        )

        # Генерируем пуш-уведомления
        push_notifications = generate_push_notifications(
            offers_dict,
            fe.features_df,
            "reports/push_notifications.json"
        )

        # Генерируем отчет по пуш-уведомлениям
        push_report = generate_push_report(
            push_notifications,
            "reports/push_report.json"
        )

        logger.info(f"Сгенерировано {len(push_notifications)} пуш-уведомлений")
        logger.info(f"Процент соответствия требованиям: {push_report['summary']['compliance_rate']}%")

        # 7. Dashboard Creation
        print("🚀 Step 7: Creating Dashboard...")
        logger.info("Создание дашборда")

        try:
            # Создаем дашборд с исправленными данными
            create_scoring_dashboard(
                offers_dict,  # Словарь предложений
                stats,  # Статистика из analytics report
                fe.features_df,  # Данные о клиентах
                show=True
            )
            logger.info("Дашборд успешно создан")

        except Exception as e:
            logger.error(f"Ошибка создания дашборда: {e}")
            print(f"⚠️ Dashboard creation failed: {e}")
            print("Продолжаем без дашборда...")

        # Финальная сводка
        print("\n✅ Pipeline finished successfully!")
        print("\n📋 Results Summary:")
        print(f"   • Processed clients: {len(profiles_df)}")
        print(f"   • Generated offers: {len(offers_df)}")
        print(f"   • Push notifications: {len(push_notifications)}")
        print(f"   • Push compliance rate: {push_report['summary']['compliance_rate']}%")
        print(f"   • Most recommended product: {stats.get('most_recommended_product', 'N/A')}")

        print("\n📁 Output files:")
        print("   • reports/client_recommendations.csv")
        print("   • reports/scoring_summary.json")
        print("   • reports/push_notifications.json")
        print("   • reports/push_report.json")
        print("   • reports/figures/scoring_dashboard.png")
        print("   • reports/pipeline.log")

        logger.info("Пайплайн успешно завершен")

        return {
            'offers_df': offers_df,
            'push_notifications': push_notifications,
            'stats': stats,
            'push_report': push_report
        }

    except Exception as e:
        logger.error(f"Критическая ошибка пайплайна: {e}")
        print(f"❌ Pipeline failed: {e}")
        raise


def validate_pipeline_results(results: dict) -> bool:
    """Валидация результатов пайплайна"""

    validation_issues = []

    # Проверка наличия предложений
    if results['offers_df'].empty:
        validation_issues.append("Не сгенерировано ни одного предложения")

    # Проверка пуш-уведомлений
    if not results['push_notifications']:
        validation_issues.append("Не сгенерировано ни одного пуш-уведомления")

    # Проверка качества пуш-уведомлений
    push_compliance = results['push_report']['summary']['compliance_rate']
    if push_compliance < 80:
        validation_issues.append(f"Низкое качество пуш-уведомлений: {push_compliance}%")

    # Проверка разнообразия продуктов
    product_count = len(results['stats'].get('product_distribution', {}))
    if product_count < 3:
        validation_issues.append(f"Мало разнообразия продуктов: {product_count}")

    if validation_issues:
        logger.warning("Обнаружены проблемы с качеством результатов:")
        for issue in validation_issues:
            logger.warning(f"  - {issue}")
        return False

    logger.info("Валидация результатов пройдена успешно")
    return True


def run_pipeline_with_validation():
    """Запуск пайплайна с валидацией результатов"""

    results = main()

    print("\n🔍 Validating results...")
    is_valid = validate_pipeline_results(results)

    if is_valid:
        print("✅ All validation checks passed!")
    else:
        print("⚠️ Some quality issues detected. Check logs for details.")

    return results, is_valid


if __name__ == "__main__":
    try:
        results, is_valid = run_pipeline_with_validation()
    except Exception as e:
        print(f"💥 Critical error: {e}")
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
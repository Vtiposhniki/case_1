
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineering
from src.model import ModelManager
from src.scoring_system import BankingScoringSystem, create_scoring_dashboard
from src.push_generator import generate_push_notifications
from src.utils import save_json


def main():
    # 1. Data Preprocessing
    print("🚀 Step 1: Data Preprocessing...")
    preprocessor = DataPreprocessor()
    preprocessor.export_all_data()

    # 2. Feature Engineering
    print("🚀 Step 2: Feature Engineering...")
    fe = FeatureEngineering()
    if not fe.create_features():
        raise RuntimeError("❌ Feature engineering failed")
    fe.save_features()
    features_df = fe.features_df

    # 3. Model Training / Loading
    print("🚀 Step 3: Training or Loading Model...")
    model_manager = ModelManager()
    if not model_manager.model_path.exists():
        model_manager.train(features_df)
    else:
        model_manager.load()

    # 4. Scoring
    print("🚀 Step 4: Client Scoring...")
    probs = model_manager.predict_proba(features_df)
    scoring_system = BankingScoringSystem()

    client_scores = {}
    for i, row in features_df.iterrows():
        client_scores[row["client_code"]] = {
            "credit_risk_score": probs[i] * 100,
            "income_estimation": row.get("total_spending", 0) * 2,
            "ltv_score": min(100, row.get("avg_monthly_balance_KZT", 0) / 50000),
            "churn_risk_score": 100 - probs[i] * 100,
            "financial_stability": scoring_system._calculate_stability_score(row),
            "engagement_score": scoring_system._calculate_engagement_score(row),
            "segment": scoring_system._determine_client_segment(row, row.get("total_spending", 0) * 2),
        }

    # 5. Generate Offers
    print("🚀 Step 5: Generating Offers...")
    offers = scoring_system.generate_personalized_offers(client_scores, features_df)
    scoring_system.save_offers_report(offers, "reports/client_offers.json")

    # Save summary statistics
    stats = scoring_system.generate_summary_statistics(offers)
    save_json(stats, "reports/scoring_summary.json")

    # 6. Generate Push Notifications
    print("🚀 Step 6: Generating Push Notifications...")
    generate_push_notifications(offers, features_df, "reports/push_notifications.json")

    # 7. Dashboard
    print("🚀 Step 7: Creating Dashboard...")
    create_scoring_dashboard(offers, stats, features_df)

    print("✅ Pipeline finished successfully!")


if __name__ == "__main__":
    main()

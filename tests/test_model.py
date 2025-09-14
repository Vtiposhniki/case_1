#!/usr/bin/env python3
"""
Enhanced main script with integrated scoring and benefits system
Полный пайплайн: обработка данных → обучение моделей → скоринг → рекомендации → A/B тесты
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
# Import existing modules
from feature_engineering import FeatureEngineering
from model import (
    create_customer_segmentation_model,
    create_classification_model, 
    create_regression_model
)

# Import scoring system
from scoring_system import (
    BankingScoringSystem,
    OfferOptimizer,
    ABTestFramework,
    create_scoring_dashboard,
    integrate_with_push_generator
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class EnhancedFinancialAITrainer:
    """Complete AI training pipeline with scoring and recommendations"""
    
    def __init__(self):
        self.features_df = None
        self.models = {}
        self.results = {}
        self.scoring_system = None
        self.client_scores = {}
        self.personalized_offers = {}
        self.optimization_results = {}
        self.ab_test_framework = None

    
    def run_feature_engineering(self) -> bool:
        """Run feature engineering pipeline"""
        logger.info("="*60)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("="*60)
        
        try:
            feature_engineer = FeatureEngineering()
            
            if not feature_engineer.create_features():
                logger.error("❌ Feature engineering failed")
                return False
            
            # Save features and get DataFrame
            feature_engineer.save_features()
            self.features_df = feature_engineer.features_df.copy()
            
            # Print summary
            summary = feature_engineer.get_feature_summary()
            logger.info("✅ Feature engineering completed!")
            for key, value in summary.items():
                if isinstance(value, float):
                    logger.info(f"  📈 {key}: {value:,.2f}")
                else:
                    logger.info(f"  📊 {key}: {value:,}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Feature engineering failed: {e}")
            return False
    
    def train_all_models(self) -> bool:
        """Train all ML models"""
        logger.info("="*60)
        logger.info("STEP 3: MACHINE LEARNING MODEL TRAINING")
        logger.info("="*60)
        
        success_count = 0
        total_models = 3
        
        # Train customer segmentation
        if self._train_customer_segmentation():
            success_count += 1
            logger.info("✅ Customer segmentation model trained")
        else:
            logger.error("❌ Customer segmentation training failed")
        
        # Train churn prediction
        if self._train_churn_prediction():
            success_count += 1
            logger.info("✅ Churn prediction model trained")
        else:
            logger.error("❌ Churn prediction training failed")
        
        # Train spending prediction
        if self._train_spending_prediction():
            success_count += 1
            logger.info("✅ Spending prediction model trained")
        else:
            logger.error("❌ Spending prediction training failed")
        
        logger.info(f"🎯 Successfully trained {success_count}/{total_models} models")
        return success_count > 0
    
    def _train_customer_segmentation(self) -> bool:
        """Train customer segmentation model"""
        try:
            n_features = len([col for col in self.features_df.columns 
                            if col not in ['client_code', 'name', 'status', 'city']])
            
            model, trainer = create_customer_segmentation_model(n_features)
            data_dict = trainer.prepare_data(self.features_df)
            history = trainer.train(data_dict, epochs=50, batch_size=32)
            
            # Get embeddings for clustering
            model.eval()
            import torch
            with torch.no_grad():
                features_tensor = torch.FloatTensor(data_dict['X_train']).to(trainer.device)
                embeddings = model.encode(features_tensor).cpu().numpy()
            
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=4, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            
            self.models['segmentation'] = {'model': model, 'trainer': trainer, 'kmeans': kmeans}
            self.results['segmentation'] = {
                'history': history,
                'clusters': clusters,
                'n_clusters': len(np.unique(clusters))
            }
            
            trainer.save_model('models/customer_segmentation.pth')
            return True
            
        except Exception as e:
            logger.error(f"Customer segmentation error: {e}")
            return False
    
    def _train_churn_prediction(self) -> bool:
        """Train churn prediction model"""
        try:
            df_with_targets = self._create_synthetic_targets()
            
            n_features = len([col for col in df_with_targets.columns 
                            if col not in ['client_code', 'name', 'status', 'city', 'churn_risk', 
                                         'customer_segment', 'next_month_spending']])
            
            model, trainer = create_classification_model(n_features, 2)
            data_dict = trainer.prepare_data(df_with_targets, 'churn_risk')
            history = trainer.train(data_dict, epochs=100, batch_size=32)
            metrics = trainer.evaluate(data_dict)
            
            self.models['churn'] = {'model': model, 'trainer': trainer}
            self.results['churn'] = {'history': history, 'metrics': metrics}
            
            trainer.save_model('models/churn_prediction.pth')
            return True
            
        except Exception as e:
            logger.error(f"Churn prediction error: {e}")
            return False
    
    def _train_spending_prediction(self) -> bool:
        """Train spending prediction model"""
        try:
            df_with_targets = self._create_synthetic_targets()
            
            n_features = len([col for col in df_with_targets.columns 
                            if col not in ['client_code', 'name', 'status', 'city', 'churn_risk', 
                                         'customer_segment', 'next_month_spending']])
            
            model, trainer = create_regression_model(n_features)
            data_dict = trainer.prepare_data(df_with_targets, 'next_month_spending')
            history = trainer.train(data_dict, epochs=100, batch_size=32)
            metrics = trainer.evaluate(data_dict)
            
            self.models['spending'] = {'model': model, 'trainer': trainer}
            self.results['spending'] = {'history': history, 'metrics': metrics}
            
            trainer.save_model('models/spending_prediction.pth')
            return True
            
        except Exception as e:
            logger.error(f"Spending prediction error: {e}")
            return False
    
    def _create_synthetic_targets(self) -> pd.DataFrame:
        """Create synthetic targets for demonstration"""
        df = self.features_df.copy()
        
        # Customer segments
        spending_q = df['total_spending'].quantile([0.33, 0.66])
        balance_q = df['avg_monthly_balance_KZT'].quantile([0.33, 0.66])
        
        def classify_customer_value(row):
            spending = row['total_spending']
            balance = row['avg_monthly_balance_KZT']
            
            if spending > spending_q[0.66] and balance > balance_q[0.66]:
                return 'High Value'
            elif spending < spending_q[0.33] and balance < balance_q[0.33]:
                return 'Low Value'
            else:
                return 'Medium Value'
        
        df['customer_segment'] = df.apply(classify_customer_value, axis=1)
        
        # Churn risk
        low_activity_threshold = df['tx_count'].quantile(0.2)
        df['churn_risk'] = (df['tx_count'] <= low_activity_threshold).astype(int)
        
        # Next month spending
        np.random.seed(42)
        noise = np.random.normal(0, 0.1, len(df))
        df['next_month_spending'] = df['total_spending'] * (1 + noise)
        df['next_month_spending'] = np.maximum(df['next_month_spending'], 0)
        
        return df
    
    def run_scoring_system(self) -> bool:
        """Run comprehensive scoring and recommendation system"""
        logger.info("="*60)
        logger.info("STEP 4: BANKING SCORING & RECOMMENDATIONS")
        logger.info("="*60)
        
        try:
            # Initialize scoring system
            self.scoring_system = BankingScoringSystem()
            
            # Calculate client scores (using heuristic approach for demo)
            logger.info("🧮 Calculating comprehensive client scores...")
            self._calculate_client_scores()
            
            # Generate personalized offers
            logger.info("🎯 Generating personalized banking offers...")
            self.personalized_offers = self.scoring_system.generate_personalized_offers(
                self.client_scores, self.features_df, max_offers_per_client=3
            )
            
            # Optimize offer allocation
            logger.info("⚡ Optimizing offer allocation...")
            self._optimize_offers()
            
            # A/B testing setup
            logger.info("🧪 Setting up A/B testing framework...")
            self._setup_ab_testing()
            
            # Generate comprehensive reports
            logger.info("📊 Generating scoring reports...")
            self.scoring_system.save_offers_report(self.personalized_offers)
            summary_stats = self.scoring_system.generate_summary_statistics(self.personalized_offers)
            logger.info(f"Summary statistics: {summary_stats}")
            create_scoring_dashboard(self.personalized_offers, summary_stats, self.features_df)
            return True
        except Exception as e:
            logger.error(f"Scoring system error: {e}")
            return False
        
    def _calculate_client_scores(self):
        # Example: heuristic scoring for demonstration
        self.client_scores = {}
        for _, client in self.features_df.iterrows():
            client_code = client['client_code']
            balance = client.get('avg_monthly_balance_KZT', 0)
            spending = client.get('total_spending', 0)
            tx_count = client.get('tx_count', 0)
            self.client_scores[client_code] = {
                'credit_risk_score': min(100, max(0, balance / 10000 + tx_count / 10)),
                'income_estimation': spending * 2,
                'ltv_score': min(100, balance / 50000 + spending / 5000),
                'churn_risk_score': max(0, 50 - tx_count / 2),
                'financial_stability': self.scoring_system._calculate_stability_score(client),
                'engagement_score': self.scoring_system._calculate_engagement_score(client),
                'segment': self.scoring_system._determine_client_segment(client, spending * 2)
            }

    def _optimize_offers(self):
        optimizer = OfferOptimizer()
        self.optimization_results = optimizer.optimize_offer_allocation(self.personalized_offers)

    def _setup_ab_testing(self):
        self.ab_test_framework = ABTestFramework()
        test_info = self.ab_test_framework.create_ab_test(self.personalized_offers)
        test_results = self.ab_test_framework.simulate_test_results("Offer AB Test")
        logger.info(f"A/B Test Results: {test_results}")

if __name__ == "__main__":
    trainer = EnhancedFinancialAITrainer();
    if trainer.run_feature_engineering():
        if trainer.train_all_models():
            pass
        #trainer.run_scoring_system()
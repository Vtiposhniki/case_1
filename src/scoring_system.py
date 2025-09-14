"""
Система скоринга и рекомендации выгодных предложений для банковских клиентов
Banking Scoring and Benefits Recommendation System
"""
import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Уровни риска клиента"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    VERY_HIGH = "very_high"


class ClientSegment(Enum):
    """Сегменты клиентов"""
    PREMIUM = "premium"           # Премиум клиенты
    MASS_AFFLUENT = "mass_affluent"  # Состоятельные
    MASS_MARKET = "mass_market"   # Массовый сегмент
    BASIC = "basic"               # Базовые клиенты
    STUDENT = "student"           # Студенты


@dataclass
class BankingProduct:
    """Банковский продукт"""
    id: str
    name: str
    category: str  # credit, deposit, card, insurance, investment
    min_income: float
    max_risk_level: RiskLevel
    target_segments: List[ClientSegment]
    commission_rate: float
    interest_rate: float
    requirements: Dict[str, Any]
    profit_margin: float  # для банка


@dataclass
class ClientOffer:
    """Персональное предложение для клиента"""
    client_code: str
    product: BankingProduct
    score: float  # 0-100, вероятность принятия
    expected_revenue: float  # ожидаемая выручка
    priority: int  # 1-5, приоритет предложения
    reasoning: str  # обоснование предложения
    conditions: Dict[str, Any]  # персональные условия


class CreditScoringModel(nn.Module):
    """PyTorch модель для кредитного скоринга"""
    
    def __init__(self, input_dim: int):
        super(CreditScoringModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Выходы для разных типов скоринга
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Отдельные головы для разных задач
        self.credit_risk_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()  # Вероятность дефолта
        )
        
        self.income_estimation_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.ReLU()  # Оценка дохода
        )
        
        self.ltv_prediction_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()  # Life Time Value (нормализованный)
        )
        
        self.churn_risk_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()  # Риск оттока
        )
    
    def forward(self, x):
        features = self.network(x)
        
        return {
            'credit_risk': self.credit_risk_head(features),
            'income_estimation': self.income_estimation_head(features),
            'ltv_prediction': self.ltv_prediction_head(features),
            'churn_risk': self.churn_risk_head(features)
        }


class BankingScoringSystem:
    """Комплексная система скоринга и рекомендаций"""
    
    def __init__(self):
        self.scoring_model: Optional[CreditScoringModel] = None
        self.feature_scaler = None
        self.products_catalog = self._initialize_products_catalog()
        self.client_profiles: Dict[str, Dict] = {}
        
    def _initialize_products_catalog(self) -> List[BankingProduct]:
        """Инициализация каталога банковских продуктов"""
        return [
            # Кредитные карты
            BankingProduct(
                id="cc_premium",
                name="Премиум кредитная карта",
                category="credit_card",
                min_income=500000,
                max_risk_level=RiskLevel.MEDIUM,
                target_segments=[ClientSegment.PREMIUM, ClientSegment.MASS_AFFLUENT],
                commission_rate=0.02,
                interest_rate=0.24,
                requirements={"min_age": 25, "employment_months": 12},
                profit_margin=0.15
            ),
            
            BankingProduct(
                id="cc_standard",
                name="Стандартная кредитная карта",
                category="credit_card",
                min_income=150000,
                max_risk_level=RiskLevel.HIGH,
                target_segments=[ClientSegment.MASS_MARKET, ClientSegment.MASS_AFFLUENT],
                commission_rate=0.015,
                interest_rate=0.28,
                requirements={"min_age": 21, "employment_months": 6},
                profit_margin=0.12
            ),
            
            # Депозиты
            BankingProduct(
                id="deposit_premium",
                name="Премиум депозит",
                category="deposit",
                min_income=200000,
                max_risk_level=RiskLevel.VERY_HIGH,
                target_segments=[ClientSegment.PREMIUM, ClientSegment.MASS_AFFLUENT],
                commission_rate=0.0,
                interest_rate=0.08,
                requirements={"min_amount": 1000000},
                profit_margin=0.03
            ),
            
            # Потребительские кредиты
            BankingProduct(
                id="personal_loan",
                name="Потребительский кредит",
                category="personal_loan",
                min_income=100000,
                max_risk_level=RiskLevel.MEDIUM,
                target_segments=[ClientSegment.MASS_MARKET, ClientSegment.MASS_AFFLUENT],
                commission_rate=0.01,
                interest_rate=0.18,
                requirements={"min_age": 23, "employment_months": 6},
                profit_margin=0.08
            ),
            
            # Инвестиционные продукты
            BankingProduct(
                id="investment_portfolio",
                name="Инвестиционный портфель",
                category="investment",
                min_income=300000,
                max_risk_level=RiskLevel.LOW,
                target_segments=[ClientSegment.PREMIUM, ClientSegment.MASS_AFFLUENT],
                commission_rate=0.025,
                interest_rate=0.0,
                requirements={"min_investment": 500000, "risk_tolerance": "medium"},
                profit_margin=0.20
            ),
            
            # Страхование
            BankingProduct(
                id="life_insurance",
                name="Страхование жизни",
                category="insurance",
                min_income=80000,
                max_risk_level=RiskLevel.HIGH,
                target_segments=[ClientSegment.MASS_MARKET, ClientSegment.MASS_AFFLUENT, ClientSegment.PREMIUM],
                commission_rate=0.05,
                interest_rate=0.0,
                requirements={"min_age": 18, "max_age": 65},
                profit_margin=0.30
            )
        ]
    
    def load_scoring_model(self, model_path: str, scaler_path: str):
        """Загрузка обученной модели скоринга"""
        try:
            # Загружаем чекпоинт
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Создаем модель (нужно знать размер входа)
            input_dim = checkpoint.get('input_dim', 50)  # значение по умолчанию
            self.scoring_model = CreditScoringModel(input_dim)
            self.scoring_model.load_state_dict(checkpoint['model_state_dict'])
            self.scoring_model.eval()
            
            # Загружаем скейлер
            import joblib
            self.feature_scaler = joblib.load(scaler_path)
            
            logger.info("Модель скоринга успешно загружена")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise
    
    def calculate_client_scores(self, client_features: pd.DataFrame) -> Dict[str, Dict]:
        """Расчет скоринговых показателей для клиентов"""
        if self.scoring_model is None:
            raise ValueError("Модель не загружена. Используйте load_scoring_model()")
        
        # Подготавливаем данные
        feature_columns = [col for col in client_features.columns 
                          if col not in ['client_code', 'name', 'status', 'city']]
        
        X = client_features[feature_columns].values
        X_scaled = self.feature_scaler.transform(X)
        
        # Предсказания модели
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            predictions = self.scoring_model(X_tensor)
        
        # Формируем результаты
        results = {}
        
        for idx, client_code in enumerate(client_features['client_code']):
            client_data = client_features.iloc[idx]
            
            # Базовые скоры
            credit_risk = float(predictions['credit_risk'][idx])
            estimated_income = float(predictions['income_estimation'][idx])
            ltv_score = float(predictions['ltv_prediction'][idx])
            churn_risk = float(predictions['churn_risk'][idx])
            
            # Дополнительные метрики
            results[client_code] = {
                'credit_risk_score': min(100, max(0, (1 - credit_risk) * 100)),  # Инвертируем риск
                'income_estimation': estimated_income * client_data.get('total_spending', 50000),  # Денормализуем
                'ltv_score': ltv_score * 100,
                'churn_risk_score': churn_risk * 100,
                'financial_stability': self._calculate_stability_score(client_data),
                'engagement_score': self._calculate_engagement_score(client_data),
                'segment': self._determine_client_segment(client_data, estimated_income)
            }
        
        return results
    
    def _calculate_stability_score(self, client_data: pd.Series) -> float:
        """Расчет индекса финансовой стабильности"""
        # Базируется на соотношении доходов, расходов и активности
        balance = client_data.get('avg_monthly_balance_KZT', 0)
        spending = client_data.get('total_spending', 0)
        tx_count = client_data.get('tx_count', 0)
        
        # Нормализуем показатели
        balance_score = min(100, balance / 100000)  # Максимум при 10M тенге
        spending_ratio = spending / max(balance, 1) if balance > 0 else 0
        activity_score = min(100, tx_count / 10)  # Максимум при 1000+ транзакций
        
        # Комбинированный скор
        stability = (balance_score * 0.4 + 
                    (1 - min(1, spending_ratio)) * 50 * 0.4 + 
                    activity_score * 0.2)
        
        return min(100, max(0, stability))
    
    def _calculate_engagement_score(self, client_data: pd.Series) -> float:
        """Расчет индекса вовлеченности клиента"""
        # Базируется на разнообразии операций и использовании продуктов
        tx_count = client_data.get('tx_count', 0)
        has_transfers = client_data.get('p2p_out', 0) > 0
        uses_fx = client_data.get('fx_flag', 0) > 0
        has_investments = client_data.get('invest_out', 0) > 0
        
        # Подсчет активности
        activity_score = min(100, tx_count / 5)  # Базовая активность
        diversity_bonus = (has_transfers * 20 + uses_fx * 20 + has_investments * 30)
        
        return min(100, activity_score + diversity_bonus)
    
    def _determine_client_segment(self, client_data: pd.Series, estimated_income: float) -> ClientSegment:
        """Определение сегмента клиента"""
        balance = client_data.get('avg_monthly_balance_KZT', 0)
        age = client_data.get('age', 30)
        
        # Логика сегментации
        if balance > 5000000 and estimated_income > 1000000:
            return ClientSegment.PREMIUM
        elif balance > 1000000 or estimated_income > 500000:
            return ClientSegment.MASS_AFFLUENT
        elif age < 25:
            return ClientSegment.STUDENT
        elif balance > 100000 or estimated_income > 200000:
            return ClientSegment.MASS_MARKET
        else:
            return ClientSegment.BASIC
    
    def generate_personalized_offers(self, 
                                   client_scores: Dict[str, Dict],
                                   client_features: pd.DataFrame,
                                   max_offers_per_client: int = 3) -> Dict[str, List[ClientOffer]]:
        """Генерация персональных предложений для клиентов"""
        
        all_offers = {}
        
        for client_code, scores in client_scores.items():
            client_data = client_features[client_features['client_code'] == client_code].iloc[0]
            client_offers = []
            
            # Определяем подходящие продукты
            suitable_products = self._filter_suitable_products(scores, client_data)
            
            # Ранжируем продукты по привлекательности
            ranked_products = self._rank_products_for_client(suitable_products, scores, client_data)
            
            # Генерируем предложения
            for i, (product, offer_score) in enumerate(ranked_products[:max_offers_per_client]):
                offer = self._create_client_offer(
                    client_code, product, offer_score, scores, client_data, i + 1
                )
                client_offers.append(offer)
            
            all_offers[client_code] = client_offers
        
        return all_offers
    
    def _filter_suitable_products(self, scores: Dict, client_data: pd.Series) -> List[BankingProduct]:
        """Фильтрация подходящих продуктов для клиента"""
        suitable = []
        
        estimated_income = scores['income_estimation']
        risk_level = self._get_risk_level(scores['credit_risk_score'])
        segment = scores['segment']
        age = client_data.get('age', 25)
        
        for product in self.products_catalog:
            # Проверяем базовые требования
            if (estimated_income >= product.min_income and
                self._risk_level_acceptable(risk_level, product.max_risk_level) and
                segment in product.target_segments):
                
                # Дополнительные проверки
                if self._check_additional_requirements(product, client_data, age):
                    suitable.append(product)
        
        return suitable
    
    def _get_risk_level(self, credit_score: float) -> RiskLevel:
        """Определение уровня риска по кредитному скору"""
        if credit_score >= 80:
            return RiskLevel.LOW
        elif credit_score >= 60:
            return RiskLevel.MEDIUM
        elif credit_score >= 40:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH
    
    def _risk_level_acceptable(self, client_risk: RiskLevel, max_product_risk: RiskLevel) -> bool:
        """Проверка приемлемости уровня риска"""
        risk_hierarchy = {
            RiskLevel.LOW: 0,
            RiskLevel.MEDIUM: 1, 
            RiskLevel.HIGH: 2,
            RiskLevel.VERY_HIGH: 3
        }
        return risk_hierarchy[client_risk] <= risk_hierarchy[max_product_risk]
    
    def _check_additional_requirements(self, product: BankingProduct, 
                                     client_data: pd.Series, age: int) -> bool:
        """Проверка дополнительных требований продукта"""
        reqs = product.requirements
        
        # Возрастные ограничения
        if 'min_age' in reqs and age < reqs['min_age']:
            return False
        if 'max_age' in reqs and age > reqs['max_age']:
            return False
            
        # Другие требования можно добавить здесь
        return True
    
    def _rank_products_for_client(self, products: List[BankingProduct], 
                                scores: Dict, client_data: pd.Series) -> List[Tuple[BankingProduct, float]]:
        """Ранжирование продуктов по привлекательности для клиента"""
        
        ranked = []
        
        for product in products:
            # Рассчитываем скор привлекательности предложения
            offer_score = self._calculate_offer_score(product, scores, client_data)
            ranked.append((product, offer_score))
        
        # Сортируем по убыванию скора
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
    
    def _calculate_offer_score(self, product: BankingProduct, 
                             scores: Dict, client_data: pd.Series) -> float:
        """Расчет скора привлекательности предложения"""
        
        # Базовый скор от кредитного рейтинга и LTV
        base_score = (scores['credit_risk_score'] * 0.3 + 
                     scores['ltv_score'] * 0.3 +
                     scores['engagement_score'] * 0.2 +
                     (100 - scores['churn_risk_score']) * 0.2)
        
        # Бонусы за соответствие продукта
        category_bonus = self._get_category_bonus(product.category, client_data)
        
        # Штраф за высокий риск оттока
        churn_penalty = scores['churn_risk_score'] * 0.5
        
        final_score = base_score + category_bonus - churn_penalty
        
        return min(100, max(0, final_score))
    
    def _get_category_bonus(self, category: str, client_data: pd.Series) -> float:
        """Бонус за категорию продукта в зависимости от поведения клиента"""
        
        bonuses = {
            'credit_card': client_data.get('total_spending', 0) / 10000,  # Активные тратят больше
            'deposit': client_data.get('avg_monthly_balance_KZT', 0) / 100000,  # Высокие балансы
            'investment': client_data.get('fx_flag', 0) * 20,  # Уже используют валютные операции
            'insurance': client_data.get('age', 25) / 2,  # Старше = больше потребность
            'personal_loan': min(20, client_data.get('tx_count', 0) / 10)  # Активность транзакций
        }
        
        return min(30, bonuses.get(category, 0))
    
    def _create_client_offer(self, client_code: str, product: BankingProduct,
                           offer_score: float, scores: Dict, client_data: pd.Series, 
                           priority: int) -> ClientOffer:
        """Создание персонального предложения"""
        
        # Расчет ожидаемой выручки
        estimated_revenue = self._calculate_expected_revenue(product, scores, client_data)
        
        # Персональные условия
        conditions = self._generate_personal_conditions(product, scores, client_data)
        
        # Обоснование предложения
        reasoning = self._generate_reasoning(product, scores, client_data)
        
        return ClientOffer(
            client_code=client_code,
            product=product,
            score=offer_score,
            expected_revenue=estimated_revenue,
            priority=priority,
            reasoning=reasoning,
            conditions=conditions
        )
    
    def _calculate_expected_revenue(self, product: BankingProduct, 
                                  scores: Dict, client_data: pd.Series) -> float:
        """Расчет ожидаемой выручки от продукта"""
        
        # Базовая выручка зависит от типа продукта
        base_amount = scores['income_estimation'] * 0.1  # 10% от дохода
        
        # Корректировки по типу продукта
        if product.category == 'credit_card':
            revenue = base_amount * product.interest_rate * 0.3  # 30% использование лимита
        elif product.category == 'deposit':
            revenue = client_data.get('avg_monthly_balance_KZT', 0) * 0.02  # 2% маржа банка
        elif product.category == 'personal_loan':
            loan_amount = min(base_amount * 5, 3000000)  # До 3М тенге
            revenue = loan_amount * product.interest_rate * 0.1  # 10% от процентов
        else:
            revenue = base_amount * product.profit_margin
        
        # Корректировка на вероятность принятия
        acceptance_probability = scores['credit_risk_score'] / 100 * scores['engagement_score'] / 100
        
        return revenue * acceptance_probability
    
    def _generate_personal_conditions(self, product: BankingProduct,
                                    scores: Dict, client_data: pd.Series) -> Dict[str, Any]:
        """Генерация персональных условий"""
        
        conditions = {}
        
        # Персональные процентные ставки
        if product.category in ['credit_card', 'personal_loan']:
            # Снижаем ставку для низкорискованных клиентов
            risk_discount = (scores['credit_risk_score'] - 50) / 100 * 0.05
            conditions['interest_rate'] = max(0.1, product.interest_rate - risk_discount)
        
        # Лимиты и суммы
        if product.category == 'credit_card':
            credit_limit = min(scores['income_estimation'] * 3, 2000000)
            conditions['credit_limit'] = credit_limit
        
        # Льготные периоды
        if scores['ltv_score'] > 70:
            conditions['grace_period_months'] = 3
            conditions['fee_waiver'] = True
        
        return conditions
    
    def _generate_reasoning(self, product: BankingProduct,
                          scores: Dict, client_data: pd.Series) -> str:
        """Генерация обоснования предложения"""
        
        reasons = []
        
        if scores['credit_risk_score'] > 70:
            reasons.append("отличная кредитная история")
        
        if scores['engagement_score'] > 60:
            reasons.append("высокая активность использования банковских продуктов")
        
        if scores['financial_stability'] > 80:
            reasons.append("стабильное финансовое положение")
        
        if client_data.get('avg_monthly_balance_KZT', 0) > 500000:
            reasons.append("значительные денежные средства на счетах")
        
        if not reasons:
            reasons.append("соответствие требованиям продукта")
        
        return f"Рекомендуем {product.name} на основе: " + ", ".join(reasons)
    
    def save_offers_report(self, offers: Dict[str, List[ClientOffer]], 
                          output_path: str = "reports/client_offers.json"):
        """Сохранение отчета с предложениями"""
        
        # Подготавливаем данные для сериализации
        offers_data = {}
        
        for client_code, client_offers in offers.items():
            offers_data[client_code] = []
            
            for offer in client_offers:
                offer_data = {
                    'product_id': offer.product.id,
                    'product_name': offer.product.name,
                    'category': offer.product.category,
                    'score': round(offer.score, 2),
                    'expected_revenue': round(offer.expected_revenue, 2),
                    'priority': offer.priority,
                    'reasoning': offer.reasoning,
                    'conditions': offer.conditions
                }
                offers_data[client_code].append(offer_data)
        
        # Сохраняем в файл
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(offers_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Отчет с предложениями сохранен: {output_path}")
    
    def generate_summary_statistics(self, offers: Dict[str, List[ClientOffer]]) -> Dict[str, Any]:
        """Генерация сводной статистики по предложениям"""
        
        total_clients = len(offers)
        total_offers = sum(len(client_offers) for client_offers in offers.values())
        total_expected_revenue = sum(
            offer.expected_revenue 
            for client_offers in offers.values()
            for offer in client_offers
        )
        
        # Статистика по категориям продуктов
        category_stats = {}
        for client_offers in offers.values():
            for offer in client_offers:
                category = offer.product.category
                if category not in category_stats:
                    category_stats[category] = {
                        'count': 0, 
                        'avg_score': 0, 
                        'total_revenue': 0
                    }
                category_stats[category]['count'] += 1
                category_stats[category]['avg_score'] += offer.score
                category_stats[category]['total_revenue'] += offer.expected_revenue
        
        # Вычисляем средние значения
        for category, stats in category_stats.items():
            stats['avg_score'] = round(stats['avg_score'] / stats['count'], 2)
            stats['total_revenue'] = round(stats['total_revenue'], 2)
        
        return {
            'total_clients_with_offers': total_clients,
            'total_offers_generated': total_offers,
            'avg_offers_per_client': round(total_offers / max(total_clients, 1), 2),
            'total_expected_revenue': round(total_expected_revenue, 2),
            'category_breakdown': category_stats
        }


def integrate_scoring_with_main_pipeline():
    """Интеграция системы скоринга с основным пайплайном"""
    
    def enhanced_main():
        """Расширенная версия main() с системой скоринга"""
        
        # ... (существующий код обучения моделей) ...
        
        logger.info("="*50)
        logger.info("STEP 4: SCORING AND OFFERS GENERATION")
        logger.info("="*50)
        
        try:
            # Загружаем обученные данные
            features_df = pd.read_csv("data/processed/features.csv")
            
            # Инициализируем систему скоринга
            scoring_system = BankingScoringSystem()
            
            # Здесь нужно было бы загрузить обученную модель скоринга
            # scoring_system.load_scoring_model("models/scoring_model.pth", "models/scaler.pkl")
            
            # Для демонстрации создаем простые скоры
            logger.info("Calculating client scores...")
            mock_scores = {}
            
            for _, client in features_df.iterrows():
                client_code = client['client_code']
                
                # Простые эвристические скоры для демонстрации
                balance = client.get('avg_monthly_balance_KZT', 0)
                spending = client.get('total_spending', 0)
                tx_count = client.get('tx_count', 0)
                
                mock_scores[client_code] = {
                    'credit_risk_score': min(100, max(0, balance / 10000 + tx_count / 10)),
                    'income_estimation': spending * 2,  # Примерная оценка
                    'ltv_score': min(100, balance / 50000 + spending / 5000),
                    'churn_risk_score': max(0, 50 - tx_count / 2),
                    'financial_stability': scoring_system._calculate_stability_score(client),
                    'engagement_score': scoring_system._calculate_engagement_score(client),
                    'segment': scoring_system._determine_client_segment(client, spending * 2)
                }
            
            # Генерируем персональные предложения
            logger.info("Generating personalized offers...")
            offers = scoring_system.generate_personalized_offers(mock_scores, features_df)
            
            # Сохраняем отчеты
            scoring_system.save_offers_report(offers)
            
            # Генерируем статистику
            summary_stats = scoring_system.generate_summary_statistics(offers)
            
            logger.info("Scoring and Offers Summary:")
            for key, value in summary_stats.items():
                if key == 'category_breakdown':
                    logger.info(f"{key}:")
                    for category, stats in value.items():
                        logger.info(f"  {category}: {stats['count']} offers, "
                                  f"avg score: {stats['avg_score']}, "
                                  f"revenue: {stats['total_revenue']:,.0f} KZT")
                else:
                    logger.info(f"{key}: {value}")
            
            # Создаем дашборд визуализации
            create_scoring_dashboard(offers, summary_stats, features_df)
            
            return True
            
        except Exception as e:
            logger.error(f"Scoring system error: {e}")
            return False
    
    return enhanced_main


def create_scoring_dashboard(offers: Dict[str, List[ClientOffer]], 
                           stats: Dict[str, Any], 
                           features_df: pd.DataFrame):
    """Создание дашборда с визуализацией скоринга и предложений"""
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Создаем фигуру с множественными субплотами
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
    
    # 1. Распределение предложений по категориям
    ax1 = fig.add_subplot(gs[0, 0])
    categories = list(stats['category_breakdown'].keys())
    counts = [stats['category_breakdown'][cat]['count'] for cat in categories]
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    
    ax1.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Распределение предложений по категориям', fontsize=14, fontweight='bold')
    
    # 2. Средние скоры по категориям
    ax2 = fig.add_subplot(gs[0, 1])
    scores = [stats['category_breakdown'][cat]['avg_score'] for cat in categories]
    bars = ax2.bar(categories, scores, color=colors)
    ax2.set_title('Средние скоры по категориям продуктов', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Средний скор')
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    
    # Добавляем значения на столбцы
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{score:.1f}', ha='center', va='bottom')
    
    # 3. Ожидаемая выручка по категориям
    ax3 = fig.add_subplot(gs[0, 2])
    revenues = [stats['category_breakdown'][cat]['total_revenue'] for cat in categories]
    bars = ax3.bar(categories, revenues, color=colors)
    ax3.set_title('Ожидаемая выручка по категориям (KZT)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Выручка (KZT)')
    ax3.set_xticks(range(len(categories)))
    ax3.set_xticklabels(categories, rotation=45, ha='right')
    
    # Форматируем значения
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # 4. Топ клиенты по потенциальной выручке
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Собираем данные по клиентам
    client_revenues = {}
    for client_code, client_offers in offers.items():
        total_revenue = sum(offer.expected_revenue for offer in client_offers)
        client_revenues[client_code] = total_revenue
    
    # Топ 10 клиентов
    top_clients = sorted(client_revenues.items(), key=lambda x: x[1], reverse=True)[:10]
    client_codes = [f"Client_{str(code)[-2:]}" for code, _ in top_clients]
    revenues = [revenue for _, revenue in top_clients]
    
    bars = ax4.barh(client_codes, revenues, color='lightblue')
    ax4.set_title('Топ-10 клиентов по потенциальной выручке', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Ожидаемая выручка (KZT)')
    
    # 5. Распределение балансов клиентов
    ax5 = fig.add_subplot(gs[1, 1])
    balances = features_df['avg_monthly_balance_KZT'].values
    balances_filtered = balances[balances > 0]  # Убираем нули для лучшей визуализации
    
    ax5.hist(balances_filtered, bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
    ax5.set_title('Распределение балансов клиентов', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Средний месячный баланс (KZT)')
    ax5.set_ylabel('Количество клиентов')
    ax5.set_xscale('log')  # Логарифмическая шкала из-за большого разброса
    
    # 6. Корреляция между тратами и количеством транзакций
    ax6 = fig.add_subplot(gs[1, 2])
    spending = features_df['total_spending'].values
    tx_counts = features_df['tx_count'].values
    
    # Убираем выбросы для лучшей визуализации
    mask = (spending > 0) & (tx_counts > 0)
    spending_clean = spending[mask]
    tx_counts_clean = tx_counts[mask]
    
    ax6.scatter(tx_counts_clean, spending_clean, alpha=0.6, color='coral', s=20)
    ax6.set_title('Связь между активностью и тратами', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Количество транзакций')
    ax6.set_ylabel('Общие траты (KZT)')
    
    # Добавляем линию тренда
    z = np.polyfit(tx_counts_clean, spending_clean, 1)
    p = np.poly1d(z)
    ax6.plot(tx_counts_clean, p(tx_counts_clean), "r--", alpha=0.8)
    
    # 7. Heatmap популярных категорий трат
    ax7 = fig.add_subplot(gs[2, :])
    
    # Выбираем топ категории трат
    spending_categories = [
        'Продукты питания', 'Кафе и рестораны', 'Одежда и обувь', 'АЗС',
        'Медицина', 'Развлечения', 'Спорт', 'Такси', 'Путешествия'
    ]
    
    available_categories = [cat for cat in spending_categories if cat in features_df.columns]
    
    if available_categories:
        # Создаем матрицу трат по категориям для топ-20 клиентов
        top_spenders = features_df.nlargest(20, 'total_spending')
        category_data = top_spenders[['client_code'] + available_categories].set_index('client_code')
        
        # Нормализуем данные для лучшей визуализации
        category_data_norm = category_data.div(category_data.sum(axis=1), axis=0) * 100
        
        sns.heatmap(category_data_norm.T, ax=ax7, cmap='YlOrRd', 
                   cbar_kws={'label': 'Доля трат (%)'}, fmt='.1f')
        ax7.set_title('Профили трат топ-20 клиентов по категориям', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Клиенты')
        ax7.set_ylabel('Категории трат')
        
        # Улучшаем читаемость
        ax7.set_xticklabels([f'C{i+1}' for i in range(len(top_spenders))], rotation=0)
        ax7.set_yticklabels(available_categories, rotation=0)
    
    plt.tight_layout()
    os.makedirs("../reports/figures", exist_ok=True)
    plt.savefig('../reports/figures/scoring_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("Scoring dashboard saved to reports/figures/scoring_dashboard.png")


# Дополнительные утилиты для работы со скорингом

class OfferOptimizer:
    """Оптимизатор предложений для максимизации прибыли банка"""
    
    def __init__(self, capacity_constraints: Dict[str, int] = None):
        """
        Args:
            capacity_constraints: Ограничения по количеству продуктов каждого типа
        """
        self.capacity_constraints = capacity_constraints or {
            'credit_card': 1000,
            'personal_loan': 500,
            'deposit': 2000,
            'investment': 200,
            'insurance': 800
        }
    
    def optimize_offer_allocation(self, all_offers: Dict[str, List[ClientOffer]]) -> Dict[str, List[ClientOffer]]:
        """
        Оптимизация распределения предложений с учетом ограничений
        Использует жадный алгоритм для максимизации общей выручки
        """
        
        # Создаем список всех предложений с метаданными
        offer_list = []
        for client_code, client_offers in all_offers.items():
            for offer in client_offers:
                offer_list.append({
                    'client_code': client_code,
                    'offer': offer,
                    'revenue_per_score': offer.expected_revenue / max(offer.score, 1),
                    'efficiency': offer.expected_revenue * offer.score / 100
                })
        
        # Сортируем по эффективности (выручка * вероятность принятия)
        offer_list.sort(key=lambda x: x['efficiency'], reverse=True)
        
        # Отслеживаем использованные квоты
        used_capacity = {category: 0 for category in self.capacity_constraints}
        selected_offers = {}
        
        # Жадный отбор предложений
        for offer_data in offer_list:
            offer = offer_data['offer']
            client_code = offer_data['client_code']
            category = offer.product.category
            
            # Проверяем доступность квоты
            if used_capacity[category] < self.capacity_constraints[category]:
                
                # Добавляем предложение
                if client_code not in selected_offers:
                    selected_offers[client_code] = []
                
                selected_offers[client_code].append(offer)
                used_capacity[category] += 1
        
        logger.info("Offer allocation optimized:")
        for category, used in used_capacity.items():
            logger.info(f"  {category}: {used}/{self.capacity_constraints[category]} used")
        
        return selected_offers


class ABTestFramework:
    """Фреймворк для A/B тестирования предложений"""
    
    def __init__(self):
        self.test_groups = {}
        self.results = {}
    
    def create_ab_test(self, 
                      offers: Dict[str, List[ClientOffer]], 
                      test_name: str,
                      variant_a_ratio: float = 0.5) -> Dict[str, Any]:
        """
        Создание A/B теста для предложений
        
        Args:
            offers: Словарь предложений
            test_name: Название теста
            variant_a_ratio: Доля клиентов в группе A
        """
        
        client_codes = list(offers.keys())
        np.random.shuffle(client_codes)
        
        split_point = int(len(client_codes) * variant_a_ratio)
        
        group_a = client_codes[:split_point]
        group_b = client_codes[split_point:]
        
        self.test_groups[test_name] = {
            'group_a': group_a,
            'group_b': group_b,
            'variant_a_ratio': variant_a_ratio,
            'total_clients': len(client_codes)
        }
        
        logger.info(f"A/B test '{test_name}' created:")
        logger.info(f"  Group A: {len(group_a)} clients")
        logger.info(f"  Group B: {len(group_b)} clients")
        
        return self.test_groups[test_name]
    
    def simulate_test_results(self, test_name: str, 
                            base_conversion_rate: float = 0.15,
                            effect_size: float = 0.02) -> Dict[str, Any]:
        """
        Симуляция результатов A/B теста
        
        Args:
            test_name: Название теста
            base_conversion_rate: Базовый коэффициент конверсии
            effect_size: Размер эффекта (разница между группами)
        """
        
        if test_name not in self.test_groups:
            raise ValueError(f"Test '{test_name}' not found")
        
        test_data = self.test_groups[test_name]
        
        # Симулируем результаты
        group_a_size = len(test_data['group_a'])
        group_b_size = len(test_data['group_b'])
        
        # Группа A - контрольная
        group_a_conversions = np.random.binomial(group_a_size, base_conversion_rate)
        
        # Группа B - с улучшением
        group_b_conversions = np.random.binomial(group_b_size, base_conversion_rate + effect_size)
        
        # Расчет метрик
        conv_rate_a = group_a_conversions / group_a_size
        conv_rate_b = group_b_conversions / group_b_size
        
        # Простой тест значимости
        from scipy.stats import chi2_contingency
        
        contingency_table = [
            [group_a_conversions, group_a_size - group_a_conversions],
            [group_b_conversions, group_b_size - group_b_conversions]
        ]
        
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        
        results = {
            'group_a_conversion_rate': conv_rate_a,
            'group_b_conversion_rate': conv_rate_b,
            'relative_improvement': (conv_rate_b - conv_rate_a) / conv_rate_a * 100,
            'absolute_difference': conv_rate_b - conv_rate_a,
            'statistical_significance': p_value < 0.05,
            'p_value': p_value,
            'confidence_level': 95,
            'recommendation': 'Deploy variant B' if (conv_rate_b > conv_rate_a and p_value < 0.05) else 'Keep variant A'
        }
        
        self.results[test_name] = results
        
        logger.info(f"A/B test '{test_name}' results:")
        logger.info(f"  Group A conversion: {conv_rate_a:.1%}")
        logger.info(f"  Group B conversion: {conv_rate_b:.1%}")
        logger.info(f"  Relative improvement: {results['relative_improvement']:+.1f}%")
        logger.info(f"  Statistical significance: {results['statistical_significance']}")
        logger.info(f"  Recommendation: {results['recommendation']}")
        
        return results
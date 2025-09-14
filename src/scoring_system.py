"""
Адаптированная система для генерации персонализированных банковских предложений
Banking Personalized Recommendation System - адаптация под конкретный кейс
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
logger = logging.getLogger(__name__)


class ClientStatus(Enum):
    """Статусы клиентов"""
    STUDENT = "Студент"
    SALARY = "Зарплатный клиент"
    PREMIUM = "Премиальный клиент"
    STANDARD = "Стандартный клиент"


@dataclass
class BankProduct:
    """Банковский продукт из кейса"""
    id: str
    name: str
    description: str
    target_behavior_signals: List[str]
    cashback_categories: List[str] = None
    cashback_rate: float = 0.0
    special_conditions: Dict[str, Any] = None
    min_balance_for_benefits: float = 0.0


@dataclass
class ClientProfile:
    """Профиль клиента"""
    client_code: int
    name: str
    status: ClientStatus
    age: int
    city: str
    avg_monthly_balance_kzt: float


@dataclass
class ClientBehavior:
    """Поведенческие данные клиента"""
    client_code: int
    transactions: List[Dict]  # date, category, amount, currency
    transfers: List[Dict]     # date, type, direction, amount, currency
    spending_by_category: Dict[str, float]
    transfer_patterns: Dict[str, float]
    travel_activity: Dict[str, Any]
    fx_activity: Dict[str, Any]
    investment_activity: Dict[str, Any]


@dataclass
class PersonalizedOffer:
    """Персональное предложение"""
    client_code: int
    client_name: str
    product: BankProduct
    expected_benefit: float  # ожидаемая выгода в тенге
    confidence_score: float  # уверенность в подходящести (0-100)
    push_notification: str


class BankingPersonalizationSystem:
    """Система персонализированных банковских рекомендаций"""

    def __init__(self):
        self.products_catalog = self._initialize_products()
        self.spending_categories = [
            'Одежда и обувь', 'Продукты питания', 'Кафе и рестораны', 'Медицина',
            'Авто', 'Спорт', 'Развлечения', 'АЗС', 'Кино', 'Питомцы', 'Книги',
            'Цветы', 'Едим дома', 'Смотрим дома', 'Играем дома', 'Косметика и Парфюмерия',
            'Подарки', 'Ремонт дома', 'Мебель', 'Спа и массаж', 'Ювелирные украшения',
            'Такси', 'Отели', 'Путешествия'
        ]

        # Маппинг категорий трат к продуктам
        self.category_to_product_mapping = {
            'travel_card': ['Такси', 'Отели', 'Путешествия'],
            'premium_card': ['Кафе и рестораны', 'Ювелирные украшения', 'Косметика и Парфюмерия'],
            'credit_card': ['Продукты питания', 'Одежда и обувь', 'Развлечения', 'Смотрим дома', 'Играем дома'],
            'fx_exchange': [],  # определяется по валютным операциям
            'cash_loan': [],    # определяется по паттернам нехватки средств
            'deposits': [],     # определяется по остаткам
            'investments': [],  # определяется по инвестиционной активности
            'gold': []         # определяется по желанию диверсификации
        }

    def _initialize_products(self) -> List[BankProduct]:
        """Инициализация каталога продуктов из кейса"""
        return [
            BankProduct(
                id="travel_card",
                name="Карта для путешествий",
                description="4% кешбэк на путешествия и такси, привилегии Visa Signature",
                target_behavior_signals=["frequent_travel", "taxi_usage", "hotel_bookings"],
                cashback_categories=["Путешествия", "Такси", "Отели"],
                cashback_rate=0.04
            ),

            BankProduct(
                id="premium_card",
                name="Премиальная карта",
                description="До 4% кешбэк, бесплатные снятия до 3 млн ₸/мес",
                target_behavior_signals=["high_balance", "frequent_withdrawals", "restaurant_spending"],
                cashback_categories=["Ювелирные украшения", "Косметика и Парфюмерия", "Кафе и рестораны"],
                cashback_rate=0.04,
                min_balance_for_benefits=1000000,
                special_conditions={
                    "base_cashback": 0.02,
                    "enhanced_cashback_1m": 0.03,
                    "enhanced_cashback_6m": 0.04,
                    "free_withdrawal_limit": 3000000,
                    "cashback_limit_monthly": 100000
                }
            ),

            BankProduct(
                id="credit_card",
                name="Кредитная карта",
                description="До 10% в любимых категориях, до 2 млн ₸ кредитный лимит",
                target_behavior_signals=["category_optimization", "online_services", "installment_usage"],
                cashback_rate=0.10,
                special_conditions={
                    "credit_limit": 2000000,
                    "grace_period_days": 60,
                    "online_services_cashback": 0.10,
                    "installment_available": True
                }
            ),

            BankProduct(
                id="fx_exchange",
                name="Обмен валют",
                description="Выгодный курс 24/7, авто-покупка по целевому курсу",
                target_behavior_signals=["fx_transactions", "multi_currency_usage"],
                special_conditions={
                    "commission": 0.0,
                    "auto_buy_available": True,
                    "24_7_available": True
                }
            ),

            BankProduct(
                id="cash_loan",
                name="Кредит наличными",
                description="До 12% годовых, без справок и залога",
                target_behavior_signals=["cash_need", "large_purchases"],
                special_conditions={
                    "rate_1_year": 0.12,
                    "rate_over_1_year": 0.21,
                    "early_repayment_penalty": 0.0
                }
            ),

            BankProduct(
                id="multicurrency_deposit",
                name="Депозит Мультивалютный",
                description="14,50% годовых, пополнение и снятие без ограничений",
                target_behavior_signals=["currency_diversification", "flexible_access_needed"],
                special_conditions={
                    "rate": 0.145,
                    "currencies": ["KZT", "USD", "RUB", "EUR"],
                    "flexible_access": True
                }
            ),

            BankProduct(
                id="savings_deposit",
                name="Депозит Сберегательный",
                description="16,50% годовых, защита KDIF",
                target_behavior_signals=["high_savings", "long_term_planning"],
                special_conditions={
                    "rate": 0.165,
                    "early_withdrawal": False,
                    "top_up": False,
                    "kdif_protection": True
                }
            ),

            BankProduct(
                id="accumulation_deposit",
                name="Депозит Накопительный",
                description="15,50% годовых, можно пополнять",
                target_behavior_signals=["regular_savings", "systematic_accumulation"],
                special_conditions={
                    "rate": 0.155,
                    "top_up": True,
                    "early_withdrawal": False
                }
            ),

            BankProduct(
                id="investments",
                name="Инвестиции",
                description="0% комиссий первый год, от 6 ₸",
                target_behavior_signals=["small_investment_start", "cost_conscious"],
                special_conditions={
                    "commission_first_year": 0.0,
                    "min_investment": 6,
                    "withdrawal_commission": 0.0
                }
            ),

            BankProduct(
                id="gold",
                name="Золотые слитки",
                description="999,9 пробы, хранение в сейфовых ячейках",
                target_behavior_signals=["diversification", "long_term_store"],
                special_conditions={
                    "purity": 999.9,
                    "storage_available": True,
                    "app_preorder": True
                }
            )
        ]

    def load_client_data(self, profiles_df: pd.DataFrame, transactions_df: pd.DataFrame,
                        transfers_df: pd.DataFrame) -> Dict[int, Tuple[ClientProfile, ClientBehavior]]:
        """Загрузка и обработка данных клиентов"""
        clients_data = {}

        for _, profile_row in profiles_df.iterrows():
            client_code = profile_row['client_code']

            # Создаем профиль клиента
            profile = ClientProfile(
                client_code=client_code,
                name=profile_row['name'],
                status=ClientStatus(profile_row['status']),
                age=profile_row['age'],
                city=profile_row['city'],
                avg_monthly_balance_kzt=profile_row['avg_monthly_balance_KZT']
            )

            # Обрабатываем транзакции
            client_transactions = transactions_df[transactions_df['client_code'] == client_code]
            client_transfers = transfers_df[transfers_df['client_code'] == client_code]

            behavior = self._analyze_client_behavior(client_transactions, client_transfers)
            behavior.client_code = client_code

            clients_data[client_code] = (profile, behavior)

        return clients_data

    def _analyze_client_behavior(self, transactions: pd.DataFrame,
                               transfers: pd.DataFrame) -> ClientBehavior:
        """Анализ поведения клиента"""

        # Анализ трат по категориям
        spending_by_category = {}
        for category in self.spending_categories:
            category_spending = transactions[transactions['category'] == category]['amount'].sum()
            if category_spending > 0:
                spending_by_category[category] = float(category_spending)

        # Анализ переводов
        transfer_patterns = {}
        for _, transfer in transfers.iterrows():
            transfer_type = transfer['type']
            direction = transfer['direction']
            amount = transfer['amount']

            key = f"{transfer_type}_{direction}"
            if key not in transfer_patterns:
                transfer_patterns[key] = 0
            transfer_patterns[key] += float(amount)

        # Анализ активности путешествий
        travel_spending = (
            spending_by_category.get('Путешествия', 0) +
            spending_by_category.get('Отели', 0) +
            spending_by_category.get('Такси', 0)
        )
        travel_activity = {
            'total_travel_spending': travel_spending,
            'taxi_count': len(transactions[transactions['category'] == 'Такси']),
            'hotel_spending': spending_by_category.get('Отели', 0),
            'travel_spending': spending_by_category.get('Путешествия', 0)
        }

        # Анализ валютной активности
        fx_activity = {
            'fx_buy': transfer_patterns.get('fx_buy_out', 0),
            'fx_sell': transfer_patterns.get('fx_sell_in', 0),
            'has_fx_activity': any(key.startswith('fx_') for key in transfer_patterns.keys())
        }

        # Анализ инвестиционной активности
        investment_activity = {
            'invest_out': transfer_patterns.get('invest_out_out', 0),
            'invest_in': transfer_patterns.get('invest_in_in', 0),
            'has_investment_activity': any(key.startswith('invest_') for key in transfer_patterns.keys()),
            'gold_activity': transfer_patterns.get('gold_buy_out', 0) + transfer_patterns.get('gold_sell_in', 0)
        }

        return ClientBehavior(
            client_code=0,  # будет установлен позже
            transactions=transactions.to_dict('records'),
            transfers=transfers.to_dict('records'),
            spending_by_category=spending_by_category,
            transfer_patterns=transfer_patterns,
            travel_activity=travel_activity,
            fx_activity=fx_activity,
            investment_activity=investment_activity
        )

    def calculate_product_benefits(self, profile: ClientProfile,
                                 behavior: ClientBehavior) -> Dict[str, Tuple[float, float]]:
        """Расчет ожидаемой выгоды по каждому продукту"""
        benefits = {}

        for product in self.products_catalog:
            expected_benefit, confidence = self.calculate_single_product_benefit(
                product, profile, behavior
            )
            benefits[product.id] = (expected_benefit, confidence)

        return benefits

    def calculate_single_product_benefit(self, product: BankProduct,
                                        profile: ClientProfile,
                                        behavior: ClientBehavior) -> Tuple[float, float]:
        """Расчет выгоды и уверенности для одного продукта"""

        if product.id == "travel_card":
            return self.calculate_travel_card_benefit(product, profile, behavior)
        elif product.id == "premium_card":
            return self.calculate_premium_card_benefit(product, profile, behavior)
        elif product.id == "credit_card":
            return self.calculate_credit_card_benefit(product, profile, behavior)
        elif product.id == "fx_exchange":
            return self.calculate_fx_benefit(product, profile, behavior)
        elif product.id == "cash_loan":
            return self.calculate_cash_loan_benefit(product, profile, behavior)
        elif product.id in ["multicurrency_deposit", "savings_deposit", "accumulation_deposit"]:
            return self.calculate_deposit_benefit(product, profile, behavior)
        elif product.id == "investments":
            return self.calculate_investment_benefit(product, profile, behavior)
        elif product.id == "gold":
            return self.calculate_gold_benefit(product, profile, behavior)
        else:
            return 0.0, 0.0

    def calculate_travel_card_benefit(self, product: BankProduct,
                                     profile: ClientProfile,
                                     behavior: ClientBehavior) -> Tuple[float, float]:
        """Расчет выгоды от карты для путешествий"""

        travel_spending = behavior.travel_activity['total_travel_spending']

        # Ожидаемый кешбэк за год
        annual_benefit = travel_spending * product.cashback_rate

        # Уверенность зависит от регулярности поездок
        taxi_transactions = behavior.travel_activity['taxi_count']
        confidence = min(100, (taxi_transactions * 10) + (travel_spending / 1000))

        return annual_benefit, confidence


    def calculate_premium_card_benefit(self, product: BankProduct,
                                      profile: ClientProfile,
                                      behavior: ClientBehavior) -> Tuple[float, float]:
        """Расчет выгоды от премиальной карты"""

        balance = profile.avg_monthly_balance_kzt
        conditions = product.special_conditions

        # Определяем уровень кешбэка в зависимости от баланса
        if balance >= 6000000:
            cashback_rate = conditions['enhanced_cashback_6m']
        elif balance >= 1000000:
            cashback_rate = conditions['enhanced_cashback_1m']
        else:
            cashback_rate = conditions['base_cashback']

        # Расчет кешбэка с премиум категорий
        premium_categories_spending = sum(
            behavior.spending_by_category.get(cat, 0)
            for cat in product.cashback_categories
        )

        total_spending = sum(behavior.spending_by_category.values())

        # Ежемесячный кешбэк (с учетом лимита)
        monthly_premium_cashback = premium_categories_spending * product.cashback_rate / 12
        monthly_base_cashback = (total_spending - premium_categories_spending) * cashback_rate / 12
        monthly_total_cashback = min(
            monthly_premium_cashback + monthly_base_cashback,
            conditions['cashback_limit_monthly']
        )

        annual_cashback_benefit = monthly_total_cashback * 12

        # Выгода от бесплатных снятий (если есть активность снятий)
        withdrawal_activity = behavior.transfer_patterns.get('atm_withdrawal_out', 0)
        annual_withdrawal_savings = (withdrawal_activity * 0.02)  # предполагаем 2% комиссия

        total_benefit = annual_cashback_benefit + annual_withdrawal_savings

        confidence = min(100, (balance / 100000) + (total_spending / 10000))

        return total_benefit, confidence

    def calculate_credit_card_benefit(self, product: BankProduct,
                                     profile: ClientProfile,
                                     behavior: ClientBehavior) -> Tuple[float, float]:
        """Расчет выгоды от кредитной карты"""

        # Находим топ-3 категории трат
        sorted_spending = sorted(
            behavior.spending_by_category.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        top_categories_spending = sum(amount for _, amount in sorted_spending)

        # Онлайн-сервисы (примерно оцениваем)
        online_categories = ['Смотрим дома', 'Играем дома', 'Кино']
        online_spending = sum(
            behavior.spending_by_category.get(cat, 0)
            for cat in online_categories
        )

        # Ожидаемый кешбэк
        monthly_top_categories_cashback = (top_categories_spending / 3) * product.cashback_rate
        monthly_online_cashback = (online_spending / 3) * product.cashback_rate
        annual_benefit = (monthly_top_categories_cashback + monthly_online_cashback) * 12

        # Выгода от рассрочки (сложно оценить, добавляем бонус при активном использовании)
        installment_activity = behavior.transfer_patterns.get('installment_payment_out', 0)
        if installment_activity > 0:
            annual_benefit += 5000  # условная выгода от рассрочки

        # Уверенность зависит от разнообразия трат и их объема
        categories_count = len([v for v in behavior.spending_by_category.values() if v > 1000])
        confidence = min(100, categories_count * 15 + (top_categories_spending / 5000))

        return annual_benefit, confidence

    def calculate_fx_benefit(self, product: BankProduct,
                            profile: ClientProfile,
                            behavior: ClientBehavior) -> Tuple[float, float]:
        """Расчет выгоды от валютного обмена"""

        fx_volume = behavior.fx_activity['fx_buy'] + behavior.fx_activity['fx_sell']
        if fx_volume == 0:
            return 0.0, 0.0

        # Определяем основную валюту по переводам
        main_currency = None
        if behavior.transfers:
            currencies = [t['currency'] for t in behavior.transfers if 'currency' in t]
            if currencies:
                main_currency = max(set(currencies), key=currencies.count)

        # Предполагаем экономию 0.5% от объема операций за счет выгодного курса
        annual_savings = fx_volume * 0.005

        confidence = min(100, fx_volume / 10000 + 20)

        return annual_savings, confidence

    def calculate_cash_loan_benefit(self, product: BankProduct,
                                   profile: ClientProfile,
                                   behavior: ClientBehavior) -> Tuple[float, float]:
        """Расчет выгоды от кредита наличными"""

        # Ищем признаки потребности в кредите
        large_purchases = sum(1 for t in behavior.transactions if t['amount'] > 100000)
        balance = profile.avg_monthly_balance_kzt
        total_spending = sum(behavior.spending_by_category.values())

        # Кредит выгоден только при явной потребности
        if large_purchases == 0 and balance > total_spending:
            return 0.0, 10.0  # низкая уверенность даже при нулевой выгоде

        # Условная выгода от доступности кредита (сложно оценить денежно)
        potential_benefit = 10000  # условная оценка удобства

        # Уверенность зависит от признаков потребности в кредите
        confidence = min(100, large_purchases * 20 + max(0, (total_spending - balance) / 1000))

        return potential_benefit, confidence

    def calculate_deposit_benefit(self, product: BankProduct,
                                 profile: ClientProfile,
                                 behavior: ClientBehavior) -> Tuple[float, float]:
        """Расчет выгоды от депозитов"""

        balance = profile.avg_monthly_balance_kzt
        conditions = product.special_conditions

        if balance < 50000:  # минимальная сумма для депозита
            return 0.0, 0.0

        # Оцениваем свободные средства (остаток минус месячные траты)
        monthly_spending = sum(behavior.spending_by_category.values()) / 3
        free_funds = max(0, balance - monthly_spending * 2)  # оставляем 2-месячный запас

        # Годовой доход от депозита
        annual_income = free_funds * conditions['rate']

        # Уверенность зависит от стабильности остатков и объема свободных средств
        confidence = min(100, (free_funds / 100000) * 20 + 30)

        # Корректировка уверенности в зависимости от типа депозита
        if product.id == "savings_deposit" and not conditions.get('flexible_access'):
            confidence *= 0.8  # снижаем для строгого депозита

        return annual_income, confidence

    def calculate_investment_benefit(self, product: BankProduct,
                                    profile: ClientProfile,
                                    behavior: ClientBehavior) -> Tuple[float, float]:
        """Расчет выгоды от инвестиций"""

        balance = profile.avg_monthly_balance_kzt
        has_investment_experience = behavior.investment_activity['has_investment_activity']

        # Условная выгода от инвестиций (сложно предсказать доходность)
        potential_investment_amount = min(balance * 0.1, 100000)  # до 10% от баланса
        expected_annual_return = potential_investment_amount * 0.1  # условные 10% годовых

        # Экономия на комиссиях в первый год
        commission_savings = potential_investment_amount * 0.01  # 1% экономия

        total_benefit = expected_annual_return + commission_savings

        # Уверенность зависит от опыта и возраста
        base_confidence = 30
        if has_investment_experience:
            base_confidence += 30
        if profile.age >= 25 and profile.age <= 45:
            base_confidence += 20
        if balance > 200000:
            base_confidence += 20

        confidence = min(100, base_confidence)

        return total_benefit, confidence

    def calculate_gold_benefit(self, product: BankProduct,
                              profile: ClientProfile,
                              behavior: ClientBehavior) -> Tuple[float, float]:
        """Расчет выгоды от золотых слитков"""

        balance = profile.avg_monthly_balance_kzt
        has_diversification_activity = (
            behavior.fx_activity['has_fx_activity'] or
            behavior.investment_activity['has_investment_activity']
        )

        if balance < 500000:  # минимальная сумма для золота
            return 0.0, 0.0

        # Условная выгода от диверсификации (защита от инфляции)
        potential_gold_investment = min(balance * 0.05, 200000)  # до 5% портфеля
        annual_benefit = potential_gold_investment * 0.03  # условная защита от инфляции

        # Уверенность зависит от склонности к диверсификации
        confidence = 20
        if has_diversification_activity:
            confidence += 40
        if balance > 1000000:
            confidence += 30
        if profile.status in [ClientStatus.PREMIUM, ClientStatus.SALARY]:
            confidence += 20

        confidence = min(100, confidence)

        return annual_benefit, confidence

    def select_best_product(self, benefits: Dict[str, Tuple[float, float]]) -> str:
        """Выбор наилучшего продукта на основе выгоды и уверенности"""

        best_product = None
        best_score = 0

        for product_id, (benefit, confidence) in benefits.items():
            # Комбинированный скор: выгода * уверенность
            score = benefit * (confidence / 100)

            if score > best_score:
                best_score = score
                best_product = product_id

        return best_product

    def normalize_push_length(self, text: str) -> str:
        """Приведение push-уведомления к длине 180–220 символов"""
        if len(text) < 180:
            extra = " Подробнее в приложении."
            while len(text + extra) < 180:
                extra += " Узнайте больше."
            text += extra
        elif len(text) > 220:
            text = text[:217] + "..."
        return text

    def generate_push_notification(self, profile: ClientProfile,
                                 behavior: ClientBehavior,
                                 product: BankProduct,
                                 expected_benefit: float) -> str:
        """Генерация персонализированного push-уведомления"""

        templates = {
            "travel_card": self._generate_travel_card_push,
            "premium_card": self._generate_premium_card_push,
            "credit_card": self._generate_credit_card_push,
            "fx_exchange": self._generate_fx_push,
            "cash_loan": self._generate_cash_loan_push,
            "multicurrency_deposit": self._generate_deposit_push,
            "savings_deposit": self._generate_deposit_push,
            "accumulation_deposit": self._generate_deposit_push,
            "investments": self._generate_investment_push,
            "gold": self._generate_gold_push
        }

        generator = templates.get(product.id)
        if generator:
            push = generator(profile, behavior, product, expected_benefit)
        else:
            push = f"{profile.name}, рекомендуем {product.name}. Подробнее в приложении."

        return self.normalize_push_length(push)

    def _generate_travel_card_push(self, profile: ClientProfile,
                                 behavior: ClientBehavior,
                                 product: BankProduct,
                                 expected_benefit: float) -> str:
        """Генерация уведомления для карты путешествий"""

        taxi_count = behavior.travel_activity['taxi_count']
        taxi_spending = behavior.spending_by_category.get('Такси', 0)

        if taxi_count > 0 and taxi_spending > 0:
            monthly_cashback = (taxi_spending * 0.04) / 3
            return (f"{profile.name}, в последние месяцы вы сделали {taxi_count} поездок на такси "
                   f"на {taxi_spending:,.0f} ₸. С картой для путешествий вернули бы "
                   f"≈{monthly_cashback:,.0f} ₸ ежемесячно. Откройте карту в приложении.")

        travel_spending = behavior.travel_activity['total_travel_spending']
        if travel_spending > 0:
            return (f"{profile.name}, ваши расходы на поездки составили {travel_spending:,.0f} ₸. "
                   f"Карта для путешествий вернула бы 4% кешбэком. Оформить карту.")

        return f"{profile.name}, карта для путешествий даст 4% кешбэк на поездки и такси. Узнать подробнее."

    def _generate_premium_card_push(self, profile: ClientProfile,
                                  behavior: ClientBehavior,
                                  product: BankProduct,
                                  expected_benefit: float) -> str:
        """Генерация уведомления для премиальной карты"""

        balance = profile.avg_monthly_balance_kzt
        restaurant_spending = behavior.spending_by_category.get('Кафе и рестораны', 0)

        if balance > 1000000:
            return (f"{profile.name}, у вас стабильно крупный остаток {balance:,.0f} ₸. "
                   f"Премиальная карта даст до 4% кешбэк и бесплатные снятия до 3 млн ₸/мес. "
                   f"Оформить сейчас.")

        if restaurant_spending > 10000:
            return (f"{profile.name}, ваши траты в ресторанах {restaurant_spending:,.0f} ₸ дают "
                   f"право на премиальную карту с повышенным кешбэком. Подключить карту.")

        return (f"{profile.name}, премиальная карта подойдет для ваших трат — "
               f"до 4% кешбэк и привилегии. Оформить карту.")

    def _generate_credit_card_push(self, profile: ClientProfile,
                                 behavior: ClientBehavior,
                                 product: BankProduct,
                                 expected_benefit: float) -> str:
        """Генерация уведомления для кредитной карты"""

        # Находим топ-3 категории
        sorted_spending = sorted(
            behavior.spending_by_category.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        if len(sorted_spending) >= 3:
            top_categories = [cat for cat, _ in sorted_spending[:3]]
            return (f"{profile.name}, ваши топ-категории — {', '.join(top_categories)}. "
                   f"Кредитная карта даёт до 10% в любимых категориях и на онлайн-сервисы. "
                   f"Оформить карту.")

        online_spending = sum(
            behavior.spending_by_category.get(cat, 0)
            for cat in ['Смотрим дома', 'Играем дома', 'Кино']
        )

        if online_spending > 5000:
            return (f"{profile.name}, вы тратите на онлайн-сервисы {online_spending:,.0f} ₸. "
                   f"Кредитная карта вернет 10% с таких покупок. Оформить карту.")

        return (f"{profile.name}, кредитная карта даст до 10% кешбэк в выбранных категориях "
               f"и на онлайн-покупки. Узнать условия.")

    def _generate_fx_push(self, profile: ClientProfile,
                        behavior: ClientBehavior,
                        product: BankProduct,
                        expected_benefit: float) -> str:
        """Генерация уведомления для обмена валют"""

        fx_volume = behavior.fx_activity['fx_buy'] + behavior.fx_activity['fx_sell']

        # Определяем основную валюту операций
        main_currency = "USD"  # упрощение, в реальности нужен анализ

        if fx_volume > 0:
            return (f"{profile.name}, вы часто меняете валюту на {fx_volume:,.0f} ₸. "
                   f"В приложении выгодный курс без комиссий и авто-покупка по целевому курсу. "
                   f"Настроить обмен.")

        # Ищем траты в валюте (условно по категориям путешествий)
        travel_spending = behavior.travel_activity.get('total_travel_spending', 0)
        if travel_spending > 10000:
            return (f"{profile.name}, при ваших тратах на поездки выгодно менять валюту "
                   f"по лучшему курсу в приложении. Настроить автообмен.")

        return f"{profile.name}, удобный обмен валют 24/7 с выгодным курсом в приложении. Попробовать."

    def _generate_cash_loan_push(self, profile: ClientProfile,
                               behavior: ClientBehavior,
                               product: BankProduct,
                               expected_benefit: float) -> str:
        """Генерация уведомления для кредита наличными"""

        # Осторожно с кредитными предложениями - только при явной потребности
        large_purchases = sum(1 for t in behavior.transactions if t['amount'] > 100000)

        if large_purchases > 0:
            return (f"{profile.name}, для крупных покупок доступен кредит наличными "
                   f"до 12% годовых без справок. Узнать доступный лимит.")

        return (f"{profile.name}, кредит наличными до 12% с гибким погашением "
               f"и без скрытых комиссий. Рассчитать условия.")

    def _generate_deposit_push(self, profile: ClientProfile,
                             behavior: ClientBehavior,
                             product: BankProduct,
                             expected_benefit: float) -> str:
        """Генерация уведомления для депозитов"""

        balance = profile.avg_monthly_balance_kzt
        monthly_spending = sum(behavior.spending_by_category.values()) / 3

        if balance > monthly_spending * 3:
            deposit_rate = product.special_conditions['rate']
            potential_income = balance * 0.5 * deposit_rate  # 50% свободных средств

            deposit_names = {
                'multicurrency_deposit': 'мультивалютный депозит',
                'savings_deposit': 'сберегательный депозит',
                'accumulation_deposit': 'накопительный депозит'
            }

            deposit_name = deposit_names.get(product.id, 'депозит')

            return (f"{profile.name}, у вас есть свободные средства на счету. "
                   f"Разместите их на {deposit_name} под {deposit_rate*100:.1f}% — "
                   f"дополнительно {potential_income:,.0f} ₸ в год. Открыть депозит.")

        return (f"{profile.name}, откройте депозит под {product.special_conditions['rate']*100:.1f}% "
               f"для сохранения и приумножения средств. Выбрать депозит.")

    def _generate_investment_push(self, profile: ClientProfile,
                                behavior: ClientBehavior,
                                product: BankProduct,
                                expected_benefit: float) -> str:
        """Генерация уведомления для инвестиций"""

        if profile.age <= 35:
            return (f"{profile.name}, начните инвестировать уже сегодня — "
                   f"первый год без комиссий и минимальный порог от 6 ₸. Открыть счёт.")

        if behavior.investment_activity['has_investment_activity']:
            return (f"{profile.name}, расширьте инвестиционные возможности — "
                   f"нулевые комиссии на старте и удобное управление. Открыть счёт.")

        balance = profile.avg_monthly_balance_kzt
        if balance > 100000:
            return (f"{profile.name}, попробуйте инвестиции с низким порогом входа "
                   f"и без комиссий на старт. Начать инвестировать.")

        return (f"{profile.name}, начните инвестиционный путь с минимальными затратами "
               f"и профессиональной поддержкой. Узнать больше.")

    def _generate_gold_push(self, profile: ClientProfile,
                          behavior: ClientBehavior,
                          product: BankProduct,
                          expected_benefit: float) -> str:
        """Генерация уведомления для золотых слитков"""

        balance = profile.avg_monthly_balance_kzt

        if balance > 1000000:
            return (f"{profile.name}, диверсифицируйте портфель золотыми слитками 999,9 пробы. "
                   f"Надежное сохранение стоимости с возможностью хранения в банке. "
                   f"Узнать условия.")

        if behavior.investment_activity['has_investment_activity']:
            return (f"{profile.name}, золотые слитки — классический инструмент "
                   f"для защиты капитала от инфляции. Добавить в портфель.")

        return (f"{profile.name}, золотые слитки 999,9 пробы для долгосрочного "
               f"сохранения капитала. Заказать в приложении.")

    def process_all_clients(self, profiles_df: pd.DataFrame,
                          transactions_df: pd.DataFrame,
                          transfers_df: pd.DataFrame) -> pd.DataFrame:
        """Обработка всех клиентов и генерация рекомендаций"""

        logger.info("Загрузка данных клиентов...")
        clients_data = self.load_client_data(profiles_df, transactions_df, transfers_df)

        results = []

        logger.info("Генерация персональных предложений...")
        for client_code, (profile, behavior) in clients_data.items():
            try:
                # Расчет выгод по всем продуктам
                benefits = self.calculate_product_benefits(profile, behavior)

                # Выбор лучшего продукта
                best_product_id = self.select_best_product(benefits)

                if best_product_id:
                    # Находим продукт
                    product = next(p for p in self.products_catalog if p.id == best_product_id)
                    expected_benefit = benefits[best_product_id][0]

                    # Генерируем push-уведомление
                    push_notification = self.generate_push_notification(
                        profile, behavior, product, expected_benefit
                    )

                    results.append({
                        'client_code': client_code,
                        'product': product.name,
                        'push_notification': push_notification
                    })
                else:
                    # Fallback на базовый продукт
                    fallback_product = self.products_catalog[0]  # Карта для путешествий
                    push_notification = self.generate_push_notification(
                        profile, behavior, fallback_product, 0
                    )

                    results.append({
                        'client_code': client_code,
                        'product': fallback_product.name,
                        'push_notification': push_notification
                    })

            except Exception as e:
                logger.error(f"Ошибка обработки клиента {client_code}: {e}")
                continue

        return pd.DataFrame(results)

    def save_results(self, results_df: pd.DataFrame, output_path: str = "client_recommendations.csv"):
        """Сохранение результатов в CSV"""
        results_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Результаты сохранены в {output_path}")

        # Выводим примеры для проверки
        sample = results_df if len(results_df) <= 5 else results_df.head(5)
        logger.info("\nПримеры сгенерированных предложений:")
        for _, row in sample.iterrows():
            logger.info(f"Клиент {row['client_code']}: {row['product']}")
            logger.info(f"  Push ({len(row['push_notification'])} символов): {row['push_notification']}")
            logger.info("")


    def generate_analytics_report(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Генерация аналитического отчета"""

        product_distribution = results_df['product'].value_counts()

        # Анализ длины уведомлений
        results_df['push_length'] = results_df['push_notification'].str.len()
        avg_push_length = results_df['push_length'].mean()

        # Проверка соблюдения лимитов длины (180-220 символов)
        within_limit = results_df[
            (results_df['push_length'] >= 180) &
            (results_df['push_length'] <= 220)
        ].shape[0]

        report = {
            'total_clients_processed': len(results_df),
            'product_distribution': product_distribution.to_dict(),
            'most_recommended_product': product_distribution.index[0],
            'avg_push_notification_length': round(avg_push_length, 1),
            'push_notifications_within_limit': within_limit,
            'compliance_rate': round(within_limit / len(results_df) * 100, 1)
        }

        return report


# Пример использования системы
def main():
    """Главная функция для демонстрации работы системы"""

    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    logger.info("Инициализация системы персонализированных банковских рекомендаций")

    # Создаем систему
    system = BankingPersonalizationSystem()

    # Демонстрационные данные (в реальности загружаются из файлов)
    sample_profiles = pd.DataFrame({
        'client_code': [1, 2, 3],
        'name': ['Рамазан', 'Алия', 'Дмитрий'],
        'status': ['Стандартный клиент', 'Премиальный клиент', 'Студент'],
        'age': [28, 34, 22],
        'city': ['Алматы', 'Нур-Султан', 'Шымкент'],
        'avg_monthly_balance_KZT': [150000, 2500000, 45000]
    })

    sample_transactions = pd.DataFrame({
        'client_code': [1, 1, 1, 2, 2, 3],
        'date': ['2024-08-01', '2024-08-15', '2024-08-20', '2024-08-05', '2024-08-25', '2024-08-10'],
        'category': ['Такси', 'Такси', 'Путешествия', 'Кафе и рестораны', 'Ювелирные украшения', 'Продукты питания'],
        'amount': [2500, 3200, 45000, 15000, 85000, 8000],
        'currency': ['KZT', 'KZT', 'KZT', 'KZT', 'KZT', 'KZT']
    })

    sample_transfers = pd.DataFrame({
        'client_code': [1, 2, 3],
        'date': ['2024-08-01', '2024-08-05', '2024-08-10'],
        'type': ['salary_in', 'deposit_topup_out', 'stipend_in'],
        'direction': ['in', 'out', 'in'],
        'amount': [200000, 500000, 25000],
        'currency': ['KZT', 'KZT', 'KZT']
    })

    # Обрабатываем клиентов
    results = system.process_all_clients(sample_profiles, sample_transactions, sample_transfers)

    # Сохраняем результаты
    system.save_results(results)

    # Генерируем аналитический отчет
    analytics = system.generate_analytics_report(results)

    logger.info("Аналитический отчет:")
    for key, value in analytics.items():
        logger.info(f"  {key}: {value}")

    return results


def create_scoring_dashboard(offers: Dict[str, List[Dict]],
                             stats: Dict[str, Any],
                             features_df: pd.DataFrame,
                             show: bool = True):
    """Создание дашборда с визуализацией скоринга и предложений"""

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    # Настройка шрифтов для корректного отображения русского текста
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    try:
        # Создаем фигуру с сеткой 3x3
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])

        # Проверяем наличие данных в stats
        if not stats or 'product_distribution' not in stats:
            logger.warning("Stats не содержат необходимых данных для дашборда")
            # Создаем базовую статистику из offers
            stats = create_stats_from_offers(offers)

        # 1. Распределение предложений по продуктам
        ax1 = fig.add_subplot(gs[0, 0])
        try:
            if 'product_distribution' in stats and stats['product_distribution']:
                products = list(stats['product_distribution'].keys())
                counts = list(stats['product_distribution'].values())
                colors = plt.cm.Set3(np.linspace(0, 1, len(products)))

                wedges, texts, autotexts = ax1.pie(counts, labels=products, colors=colors,
                                                   autopct='%1.1f%%', startangle=90)
                # Улучшаем читаемость текста
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(8)

                ax1.set_title('Распределение предложений по продуктам', fontsize=12, fontweight='bold', pad=20)
            else:
                ax1.text(0.5, 0.5, 'Нет данных\nо продуктах', ha='center', va='center',
                         transform=ax1.transAxes, fontsize=14)
                ax1.set_title('Распределение продуктов', fontsize=12, fontweight='bold')
        except Exception as e:
            logger.error(f"Ошибка создания pie chart: {e}")
            ax1.text(0.5, 0.5, f'Ошибка:\n{str(e)[:50]}', ha='center', va='center',
                     transform=ax1.transAxes, fontsize=10)

        # 2. Средние значения по продуктам (если есть product_stats)
        ax2 = fig.add_subplot(gs[0, 1])
        try:
            if 'product_stats' in stats and stats['product_stats']:
                products = list(stats['product_stats'].keys())
                # Используем разные метрики в зависимости от доступных данных
                if 'avg_benefit' in list(stats['product_stats'].values())[0]:
                    values = [stats['product_stats'][prod]['avg_benefit'] for prod in products]
                    ylabel = 'Средняя выгода (₸)'
                    title = 'Средняя выгода по продуктам'
                else:
                    values = [stats['product_stats'][prod].get('count', 0) for prod in products]
                    ylabel = 'Количество предложений'
                    title = 'Количество предложений по продуктам'

                colors = plt.cm.Set3(np.linspace(0, 1, len(products)))
                bars = ax2.bar(range(len(products)), values, color=colors)

                ax2.set_title(title, fontsize=12, fontweight='bold')
                ax2.set_ylabel(ylabel, fontsize=10)
                ax2.set_xticks(range(len(products)))
                ax2.set_xticklabels([prod[:15] + '...' if len(prod) > 15 else prod
                                     for prod in products], rotation=45, ha='right', fontsize=8)

                # Добавляем значения на столбцы
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    if height > 0:
                        ax2.text(bar.get_x() + bar.get_width() / 2., height + max(values) * 0.01,
                                 f'{value:,.0f}', ha='center', va='bottom', fontsize=8)
            else:
                ax2.text(0.5, 0.5, 'Нет данных\nо статистике\nпродуктов', ha='center', va='center',
                         transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Статистика продуктов', fontsize=12, fontweight='bold')
        except Exception as e:
            logger.error(f"Ошибка создания bar chart: {e}")
            ax2.text(0.5, 0.5, f'Ошибка:\n{str(e)[:30]}', ha='center', va='center',
                     transform=ax2.transAxes, fontsize=10)

        # 3. Общая статистика
        ax3 = fig.add_subplot(gs[0, 2])
        try:
            # Создаем текстовую сводку
            summary_text = f"Всего предложений: {stats.get('total_offers', 0)}\n"
            summary_text += f"Уникальных клиентов: {stats.get('unique_clients', len(offers))}\n"

            if 'total_expected_revenue' in stats:
                revenue = stats['total_expected_revenue']
                summary_text += f"Ожидаемая выручка: {revenue:,.0f} ₸\n"

            if 'avg_benefit_per_client' in stats:
                avg_benefit = stats['avg_benefit_per_client']
                summary_text += f"Средняя выгода: {avg_benefit:,.0f} ₸\n"

            if 'avg_confidence' in stats:
                confidence = stats['avg_confidence']
                summary_text += f"Средняя уверенность: {confidence:.1f}%"

            ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            ax3.set_title('Общая статистика', fontsize=12, fontweight='bold')
            ax3.axis('off')
        except Exception as e:
            logger.error(f"Ошибка создания summary: {e}")
            ax3.text(0.5, 0.5, f'Ошибка:\n{str(e)[:30]}', ha='center', va='center',
                     transform=ax3.transAxes, fontsize=10)

        # 4. Анализ клиентов (если есть данные)
        ax4 = fig.add_subplot(gs[1, 0])
        try:
            if not features_df.empty and 'avg_monthly_balance_KZT' in features_df.columns:
                balances = features_df['avg_monthly_balance_KZT']
                balances_clean = balances[balances > 0]

                if len(balances_clean) > 0:
                    ax4.hist(balances_clean, bins=min(30, len(balances_clean) // 2),
                             color='lightgreen', alpha=0.7, edgecolor='black')
                    ax4.set_title('Распределение балансов', fontsize=12, fontweight='bold')
                    ax4.set_xlabel('Баланс (₸)', fontsize=10)
                    ax4.set_ylabel('Количество клиентов', fontsize=10)

                    # Логарифмическая шкала если большой разброс
                    if balances_clean.max() / balances_clean.min() > 100:
                        ax4.set_xscale('log')
                else:
                    ax4.text(0.5, 0.5, 'Нет данных\nо балансах', ha='center', va='center',
                             transform=ax4.transAxes, fontsize=12)
            else:
                ax4.text(0.5, 0.5, 'Нет данных\nо клиентах', ha='center', va='center',
                         transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Анализ клиентов', fontsize=12, fontweight='bold')
        except Exception as e:
            logger.error(f"Ошибка анализа балансов: {e}")
            ax4.text(0.5, 0.5, f'Ошибка:\n{str(e)[:30]}', ha='center', va='center',
                     transform=ax4.transAxes, fontsize=10)

        # 5. Активность клиентов
        ax5 = fig.add_subplot(gs[1, 1])
        try:
            if not features_df.empty:
                # Ищем колонки с транзакциями или тратами
                spending_cols = [col for col in features_df.columns
                                 if 'total_spending' in col or 'tx_count' in col or 'amount' in col.lower()]

                if len(spending_cols) >= 2:
                    x_col, y_col = spending_cols[:2]
                    x_data = features_df[x_col].dropna()
                    y_data = features_df[y_col].dropna()

                    # Берем пересечение индексов
                    common_idx = x_data.index.intersection(y_data.index)
                    if len(common_idx) > 1:
                        x_clean = x_data.loc[common_idx]
                        y_clean = y_data.loc[common_idx]

                        ax5.scatter(x_clean, y_clean, alpha=0.6, color='coral', s=20)
                        ax5.set_xlabel(x_col, fontsize=10)
                        ax5.set_ylabel(y_col, fontsize=10)
                        ax5.set_title('Активность клиентов', fontsize=12, fontweight='bold')

                        # Добавляем трендлинию если достаточно точек
                        if len(x_clean) > 2:
                            try:
                                z = np.polyfit(x_clean, y_clean, 1)
                                p = np.poly1d(z)
                                ax5.plot(x_clean, p(x_clean), "r--", alpha=0.8)
                            except:
                                pass
                    else:
                        ax5.text(0.5, 0.5, 'Недостаточно\nданных для\nанализа',
                                 ha='center', va='center', transform=ax5.transAxes, fontsize=12)
                else:
                    ax5.text(0.5, 0.5, 'Нет данных\nо активности', ha='center', va='center',
                             transform=ax5.transAxes, fontsize=12)
            else:
                ax5.text(0.5, 0.5, 'Нет данных\nо клиентах', ha='center', va='center',
                         transform=ax5.transAxes, fontsize=12)

            ax5.set_title('Активность клиентов', fontsize=12, fontweight='bold')
        except Exception as e:
            logger.error(f"Ошибка анализа активности: {e}")
            ax5.text(0.5, 0.5, f'Ошибка:\n{str(e)[:30]}', ha='center', va='center',
                     transform=ax5.transAxes, fontsize=10)

        # 6. Топ клиенты
        ax6 = fig.add_subplot(gs[1, 2])
        try:
            if offers and isinstance(offers, dict):
                # Вычисляем метрику для каждого клиента
                client_metrics = {}
                for client_code, client_offers in offers.items():
                    if client_offers:
                        # Берем максимальную выгоду или количество предложений
                        if isinstance(client_offers[0], dict) and 'expected_benefit' in client_offers[0]:
                            metric = sum(offer.get('expected_benefit', 0) for offer in client_offers)
                        else:
                            metric = len(client_offers)
                        client_metrics[client_code] = metric

                if client_metrics:
                    # Топ-10 клиентов
                    top_clients = sorted(client_metrics.items(), key=lambda x: x[1], reverse=True)[:10]

                    if top_clients:
                        client_codes = [f"Клиент {str(code)[-3:]}" for code, _ in top_clients]
                        values = [metric for _, metric in top_clients]

                        ax6.barh(client_codes, values, color='lightblue')
                        ax6.set_title('Топ-10 клиентов', fontsize=12, fontweight='bold')
                        ax6.set_xlabel('Метрика', fontsize=10)

                        # Поворачиваем подписи для лучшей читаемости
                        ax6.tick_params(axis='y', labelsize=8)
                    else:
                        ax6.text(0.5, 0.5, 'Нет данных\nо топ клиентах', ha='center', va='center',
                                 transform=ax6.transAxes, fontsize=12)
                else:
                    ax6.text(0.5, 0.5, 'Нет метрик\nдля клиентов', ha='center', va='center',
                             transform=ax6.transAxes, fontsize=12)
            else:
                ax6.text(0.5, 0.5, 'Нет данных\nо предложениях', ha='center', va='center',
                         transform=ax6.transAxes, fontsize=12)

            ax6.set_title('Топ клиенты', fontsize=12, fontweight='bold')
        except Exception as e:
            logger.error(f"Ошибка анализа топ клиентов: {e}")
            ax6.text(0.5, 0.5, f'Ошибка:\n{str(e)[:30]}', ha='center', va='center',
                     transform=ax6.transAxes, fontsize=10)

        # 7. Heatmap категорий трат (занимает всю нижнюю строку)
        ax7 = fig.add_subplot(gs[2, :])
        try:
            if not features_df.empty:
                # Ищем категории трат
                spending_categories = [col for col in features_df.columns
                                       if any(cat in col for cat in ['Продукты', 'Кафе', 'Одежда', 'АЗС',
                                                                     'Медицина', 'Развлечения', 'Спорт',
                                                                     'Такси', 'Путешествия'])]

                if len(spending_categories) > 2:
                    # Берем топ клиентов по общим тратам
                    total_spending_col = None
                    for col in features_df.columns:
                        if 'total_spending' in col.lower() or 'общие_траты' in col.lower():
                            total_spending_col = col
                            break

                    if total_spending_col:
                        top_spenders = features_df.nlargest(min(20, len(features_df)), total_spending_col)
                    else:
                        top_spenders = features_df.head(min(20, len(features_df)))

                    # Создаем матрицу для heatmap
                    if not top_spenders.empty and len(spending_categories) > 0:
                        # Берем только доступные категории
                        available_categories = [cat for cat in spending_categories if cat in top_spenders.columns]

                        if available_categories:
                            category_data = top_spenders[available_categories]

                            # Нормализуем по строкам (процент от общих трат клиента)
                            row_sums = category_data.sum(axis=1)
                            category_data_norm = category_data.div(row_sums, axis=0) * 100
                            category_data_norm = category_data_norm.fillna(0)

                            # Создаем heatmap
                            sns.heatmap(category_data_norm.T, ax=ax7, cmap='YlOrRd',
                                        cbar_kws={'label': 'Доля трат (%)'},
                                        xticklabels=[f'К{i + 1}' for i in range(len(top_spenders))],
                                        yticklabels=[cat[:20] + '...' if len(cat) > 20 else cat
                                                     for cat in available_categories])

                            ax7.set_title('Профили трат клиентов по категориям', fontsize=12, fontweight='bold')
                            ax7.set_xlabel('Клиенты', fontsize=10)
                            ax7.set_ylabel('Категории', fontsize=10)
                        else:
                            ax7.text(0.5, 0.5, 'Нет данных\nо категориях трат', ha='center', va='center',
                                     transform=ax7.transAxes, fontsize=12)
                    else:
                        ax7.text(0.5, 0.5, 'Недостаточно\nданных для\nheatmap', ha='center', va='center',
                                 transform=ax7.transAxes, fontsize=12)
                else:
                    ax7.text(0.5, 0.5, 'Нет данных\nо категориях\nтрат', ha='center', va='center',
                             transform=ax7.transAxes, fontsize=12)
            else:
                ax7.text(0.5, 0.5, 'Нет данных\nо клиентах', ha='center', va='center',
                         transform=ax7.transAxes, fontsize=12)

            ax7.set_title('Анализ категорий трат', fontsize=12, fontweight='bold')
        except Exception as e:
            logger.error(f"Ошибка создания heatmap: {e}")
            ax7.text(0.5, 0.5, f'Ошибка создания heatmap:\n{str(e)[:50]}',
                     ha='center', va='center', transform=ax7.transAxes, fontsize=10)

        plt.tight_layout(pad=3.0)

        # Сохранение
        os.makedirs("reports/figures", exist_ok=True)
        output_path = "reports/figures/scoring_dashboard.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

        if show:
            plt.show()
        else:
            plt.close(fig)

        logger.info(f"Dashboard сохранен в {output_path}")

    except Exception as e:
        logger.error(f"Критическая ошибка создания dashboard: {e}")
        # Создаем базовый дашборд с сообщением об ошибке
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Ошибка создания дашборда:\n{str(e)}\n\nПроверьте данные и повторите попытку',
                ha='center', va='center', transform=ax.transAxes, fontsize=14,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral"))
        ax.set_title('Ошибка Dashboard', fontsize=16, fontweight='bold')
        ax.axis('off')

        os.makedirs("reports/figures", exist_ok=True)
        plt.savefig("reports/figures/scoring_dashboard_error.png", dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)


def create_stats_from_offers(offers: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """ Создание базовой статистики из offers если stats пустой"""

    if not offers:
        return {'total_offers': 0, 'unique_clients': 0, 'product_distribution': {}}

    all_offers = []
    for client_offers in offers.values():
        all_offers.extend(client_offers)

    product_distribution = {}
    for offer in all_offers:
        product = offer.get('product', 'Неизвестный продукт')
        product_distribution[product] = product_distribution.get(product, 0) + 1

    return {
        'total_offers': len(all_offers),
        'unique_clients': len(offers),
        'product_distribution': product_distribution
    }

if __name__ == "__main__":
    results = main()
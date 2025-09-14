import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


def generate_push_notifications(offers_dict: Dict[str, List[Dict]],
                                features_df: pd.DataFrame,
                                output_path: str = "reports/push_notifications.json") -> List[Dict]:
    """
    Генерация пуш-уведомлений на основе офферов из scoring_system

    Args:
        offers_dict: словарь {client_code: [{'client_code': int, 'product': str, 'push_notification': str}, ...]}
        features_df: DataFrame с данными клиентов для получения дополнительной информации
        output_path: путь для сохранения JSON

    Returns:
        List[Dict]: список пуш-уведомлений
    """

    push_messages = []

    for client_code_str, client_offers in offers_dict.items():
        try:
            client_code = int(client_code_str)

            if not client_offers:
                continue

            # Берем первое (и обычно единственное) предложение для клиента
            best_offer = client_offers[0]

            # Получаем базовые данные из оффера
            product_name = best_offer.get('product', 'Неизвестный продукт')
            push_text = best_offer.get('push_notification', '')

            # Получаем имя клиента из features_df если доступно
            client_name = "Клиент"
            client_data = features_df[features_df["client_code"] == client_code]
            if not client_data.empty and 'name' in client_data.columns:
                client_name = client_data.iloc[0]['name']

            # Если push_notification пустой, генерируем базовый
            if not push_text or len(push_text.strip()) == 0:
                push_text = generate_fallback_push_notification(client_name, product_name)

            # Определяем категорию продукта для дополнительной логики
            category = map_product_name_to_category(product_name)

            # Создаем структуру пуш-уведомления
            push_message = {
                "client_code": client_code,
                "client_name": client_name,
                "product": product_name,
                "push_notification": push_text,
                "category": category,
                "length": len(push_text)
            }

            # Добавляем дополнительные поля если они есть в оффере
            if 'expected_benefit' in best_offer:
                push_message['expected_benefit'] = best_offer['expected_benefit']
            if 'confidence_score' in best_offer:
                push_message['confidence_score'] = best_offer['confidence_score']

            push_messages.append(push_message)

        except Exception as e:
            logger.error(f"Ошибка обработки клиента {client_code_str}: {e}")
            continue

    # Сохраняем в JSON
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(push_messages, f, ensure_ascii=False, indent=2)

        logger.info(f"Сохранено {len(push_messages)} пуш-уведомлений в {output_path}")

    except Exception as e:
        logger.error(f"Ошибка сохранения пуш-уведомлений: {e}")

    return push_messages


def map_product_name_to_category(product_name: str) -> str:
    """Маппинг названия продукта к категории"""

    product_name_lower = product_name.lower()

    if "карта" in product_name_lower:
        if "путешеств" in product_name_lower or "travel" in product_name_lower:
            return "travel_card"
        elif "премиальн" in product_name_lower or "premium" in product_name_lower:
            return "premium_card"
        elif "кредитн" in product_name_lower or "credit" in product_name_lower:
            return "credit_card"
        else:
            return "cards"

    elif "депозит" in product_name_lower or "вклад" in product_name_lower:
        return "deposit"

    elif "кредит" in product_name_lower and "наличн" in product_name_lower:
        return "personal_loan"

    elif "инвестиц" in product_name_lower or "invest" in product_name_lower:
        return "investment"

    elif "валют" in product_name_lower or "обмен" in product_name_lower:
        return "fx_services"

    elif "золот" in product_name_lower or "gold" in product_name_lower:
        return "gold"

    else:
        return "other"


def generate_fallback_push_notification(client_name: str, product_name: str) -> str:
    """Генерация fallback пуш-уведомления если основное отсутствует"""

    category = map_product_name_to_category(product_name)

    templates = {
        "credit_card": f"{client_name}, ваши траты в топ-категориях делают карту выгодной. "
                       f"До 10% кешбэка на любимые покупки и онлайн-сервисы. Оформить карту.",

        "deposit": f"{client_name}, у вас остаются свободные средства. "
                   f"Разместите их на вкладе — удобно копить и получать доход. Открыть вклад.",

        "personal_loan": f"{client_name}, если нужен запас на крупные траты — можно оформить "
                         f"кредит наличными с гибкими выплатами. Узнать лимит.",

        "investment": f"{client_name}, попробуйте инвестиции с низким порогом входа "
                      f"и без комиссий на старте. Открыть счёт.",

        "travel_card": f"{client_name}, карта для путешествий даст 4% кешбэк на поездки "
                       f"и такси. Оформить в приложении.",

        "premium_card": f"{client_name}, премиальная карта подойдет для ваших трат — "
                        f"до 4% кешбэк и привилегии. Оформить карту.",

        "fx_services": f"{client_name}, удобный обмен валют 24/7 с выгодным курсом "
                       f"в приложении. Попробовать.",

        "gold": f"{client_name}, золотые слитки — надежный способ сохранить капитал. "
                f"Заказать в приложении."
    }

    fallback_text = templates.get(category,
                                  f"{client_name}, у нас есть персональное предложение: {product_name}. "
                                  f"Узнать подробнее в приложении.")

    # Нормализуем длину до требований (180-220 символов)
    return normalize_push_length(fallback_text)


def normalize_push_length(text: str) -> str:
    """Приведение push-уведомления к длине 180–220 символов"""

    if len(text) < 180:
        # Добавляем текст до минимальной длины
        extensions = [
            " Подробнее в приложении.",
            " Узнайте больше в мобильном приложении.",
            " Оформление займет всего несколько минут.",
            " Все условия доступны в приложении банка."
        ]

        for ext in extensions:
            if len(text + ext) >= 180:
                text += ext
                break
            text += ext

        # Если все еще короткий, добавляем общий текст
        while len(text) < 180:
            text += " Подробности в приложении."

    elif len(text) > 220:
        # Обрезаем с многоточием
        text = text[:217] + "..."

    return text


def validate_push_notifications(push_messages: List[Dict]) -> Dict[str, Any]:
    """Валидация сгенерированных пуш-уведомлений"""

    validation_results = {
        "total_messages": len(push_messages),
        "valid_length_count": 0,
        "too_short_count": 0,
        "too_long_count": 0,
        "empty_messages": 0,
        "length_distribution": {},
        "category_distribution": {},
        "issues": []
    }

    for msg in push_messages:
        push_text = msg.get("push_notification", "")
        category = msg.get("category", "unknown")
        length = len(push_text)

        # Подсчет по длине
        if length == 0:
            validation_results["empty_messages"] += 1
        elif length < 180:
            validation_results["too_short_count"] += 1
        elif length > 220:
            validation_results["too_long_count"] += 1
        else:
            validation_results["valid_length_count"] += 1

        # Распределение по категориям
        validation_results["category_distribution"][category] = \
            validation_results["category_distribution"].get(category, 0) + 1

        # Группировка по длине (по диапазонам)
        length_range = f"{(length // 20) * 20}-{(length // 20) * 20 + 19}"
        validation_results["length_distribution"][length_range] = \
            validation_results["length_distribution"].get(length_range, 0) + 1

    # Вычисляем процент соответствия требованиям
    if validation_results["total_messages"] > 0:
        compliance_rate = (validation_results["valid_length_count"] /
                           validation_results["total_messages"]) * 100
        validation_results["compliance_rate"] = round(compliance_rate, 1)
    else:
        validation_results["compliance_rate"] = 0

    # Добавляем предупреждения
    if validation_results["empty_messages"] > 0:
        validation_results["issues"].append(f"Найдено {validation_results['empty_messages']} пустых сообщений")

    if validation_results["too_short_count"] > 0:
        validation_results["issues"].append(
            f"Найдено {validation_results['too_short_count']} коротких сообщений (<180 символов)")

    if validation_results["too_long_count"] > 0:
        validation_results["issues"].append(
            f"Найдено {validation_results['too_long_count']} длинных сообщений (>220 символов)")

    return validation_results


def generate_push_report(push_messages: List[Dict], output_path: str = "reports/push_report.json"):
    """Генерация отчета по пуш-уведомлениям"""

    validation_results = validate_push_notifications(push_messages)

    # Примеры сообщений для каждой категории
    examples_by_category = {}
    for msg in push_messages:
        category = msg.get("category", "unknown")
        if category not in examples_by_category:
            examples_by_category[category] = {
                "product": msg.get("product", ""),
                "push_notification": msg.get("push_notification", ""),
                "length": len(msg.get("push_notification", ""))
            }

    report = {
        "summary": validation_results,
        "examples": examples_by_category,
        "recommendations": []
    }

    # Добавляем рекомендации
    if validation_results["compliance_rate"] < 90:
        report["recommendations"].append("Необходимо улучшить соответствие требованиям длины сообщений")

    if validation_results["empty_messages"] > 0:
        report["recommendations"].append("Устранить пустые пуш-уведомления")

    if len(validation_results["category_distribution"]) < 3:
        report["recommendations"].append("Рассмотреть расширение разнообразия категорий продуктов")

    # Сохраняем отчет
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Отчет по пуш-уведомлениям сохранен в {output_path}")

    except Exception as e:
        logger.error(f"Ошибка сохранения отчета: {e}")

    return report
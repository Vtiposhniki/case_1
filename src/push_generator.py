import json
from pathlib import Path

def generate_push_notifications(offers, features_df, output_path="reports/push_notifications.json"):
    """
    Генерация пуш-уведомлений на основе лучших офферов и сохранение в JSON
    """

    push_messages = []

    for client_code, client_offers in offers.items():
        if not client_offers:
            continue

        # Берём лучший оффер (score максимальный)
        best_offer = max(client_offers, key=lambda x: x.score)
        client_data = features_df[features_df["client_code"] == client_code].iloc[0]

        # Генерируем пуш по шаблону
        if best_offer.product.category == "credit_card":
            text = f"{client_data['name']}, ваши траты в топ-категориях делают карту выгодной. До 10% кешбэка на любимые покупки и онлайн-сервисы. Оформить карту."
        elif best_offer.product.category == "deposit":
            text = f"{client_data['name']}, у вас остаются свободные средства. Разместите их на вкладе — удобно копить и получать доход. Открыть вклад."
        elif best_offer.product.category == "personal_loan":
            text = f"{client_data['name']}, если нужен запас на крупные траты — можно оформить кредит наличными с гибкими выплатами. Узнать лимит."
        elif best_offer.product.category == "investment":
            text = f"{client_data['name']}, попробуйте инвестиции с низким порогом входа и без комиссий на старте. Открыть счёт."
        elif best_offer.product.category == "insurance":
            text = f"{client_data['name']}, для вашей стабильности доступно страхование жизни. Условия простые, оформление онлайн. Подробнее."
        else:
            text = f"{client_data['name']}, у нас есть персональное предложение: {best_offer.product.name}. Узнать подробнее."

        # Упаковываем в JSON-структуру
        push_messages.append({
            "client_code": client_code,
            "product": best_offer.product.name,
            "push_notification": text,
            "expected_revenue": best_offer.expected_revenue,
            "score": best_offer.score,
            "conditions": best_offer.conditions
        })

    # Сохраняем в JSON
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(push_messages, f, ensure_ascii=False, indent=2)

    return push_messages

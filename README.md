 Decentrathon 4.0 BCC.KZ Case_1


# 🏦 Bank Push System

**Banking Personalized Recommendation System** — это система, которая анализирует поведение клиентов банка и автоматически формирует персонализированные продуктовые предложения с push-уведомлениями и визуальной аналитикой.  
Проект включает полный пайплайн: от обработки сырых данных до генерации отчётов и дашбордов.

---

## ⚙️ Подготовка окружения

Перед запуском необходимо установить [**uv**](https://docs.astral.sh/uv/) — быстрый менеджер окружений и зависимостей для Python.

Если `uv` ещё не установлен:

```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh 
```

## ⚙️ Проверьте установку:
```bash
    uv --version
```

### ⚙️ Запуск Проекта 
### 1.Клонировать проект

```bash
    git clone https://github.com/Vtiposhniki/case_1.git
    cd ./case_1
```
### ⚙️ 2.Установить зависимости
```bash
    uv sync
```


### ⚙️ 3.Запустить pipline
```bash
    uv run python main.py
```

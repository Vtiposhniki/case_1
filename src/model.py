from catboost import CatBoostClassifier, Pool
import joblib
from pathlib import Path

class ModelManager:
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model_path = self.model_dir / "catboost_model.cbm"
        self.features_path = self.model_dir / "feature_columns.pkl"
        self.model = None
        self.feature_columns = None

    def train(self, features_df, target_col="total_spending"):
        """
        Обучение CatBoost модели (пример таргета: траты > 200k)
        """
        # Формируем таргет
        y = (features_df[target_col] > 200000).astype(int)

        # Убираем служебные поля
        drop_cols = ["client_code", "name", target_col]
        X = features_df.drop(columns=[c for c in drop_cols if c in features_df.columns])

        # Определяем категориальные признаки
        cat_features = X.select_dtypes(include=["object"]).columns.tolist()

        # Сохраняем список фичей
        self.feature_columns = list(X.columns)

        # Готовим Pool
        train_pool = Pool(X, y, cat_features=cat_features)

        # Инициализация модели
        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=8,
            eval_metric="AUC",
            verbose=False,
            random_state=42
        )

        # Обучение
        model.fit(train_pool)

        # Сохранение модели и признаков
        model.save_model(str(self.model_path))
        joblib.dump(self.feature_columns, self.features_path)

        self.model = model
        return model

    def load(self):
        """
        Загружает обученную модель
        """
        if not self.model_path.exists():
            raise FileNotFoundError("Модель не найдена. Сначала обучите её.")
        self.model = CatBoostClassifier()
        self.model.load_model(str(self.model_path))
        self.feature_columns = joblib.load(self.features_path)
        return self.model

    def predict_proba(self, features_df):
        """
        Предсказание вероятности (вернёт массив p)
        """
        if self.model is None:
            self.load()
        X = features_df[self.feature_columns]
        return self.model.predict_proba(X)[:, 1]

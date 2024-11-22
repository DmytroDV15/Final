Використані технології та бібліотеки

- **Обробка даних:**
  - `os`: Робота з файлами та директоріями.
  - `pandas`: Маніпуляція та аналіз табличних даних.
  - `numpy`: Робота з багатовимірними масивами.
- **Візуалізація:**
  - `matplotlib`: Побудова графіків.
  - `seaborn`: Створення інформативних візуалізацій.
- **Машинне навчання:**
  - `scikit-learn`: Тренування моделей, розбиття даних на навчальні та тестові вибірки.
  - `RandomForestRegressor`: Модель випадкових лісів для регресії.
  - `XGBRegressor`: Модель XGBoost для побудови потужних регресійних моделей.
- **Попередня обробка:**
  - `StandardScaler`: Стандартизація ознак.
- **Глибинне навчання:**
  - `tensorflow.keras`: Побудова рекурентних нейронних мереж (LSTM).

## Функціональність

1. **Попередня обробка даних:**
   - Читання та підготовка наборів даних.
   - Стандартизація ознак.

2. **Аналіз даних:**
   - Візуалізація за допомогою `matplotlib` та `seaborn`.

3. **Тренування моделей:**
   - Випадкові ліси (`RandomForestRegressor`).
   - XGBoost (`XGBRegressor`).
   - Рекурентні нейронні мережі (LSTM).

4. **Оцінка моделей:**
   - Метрики: `mean_squared_error`, `mean_absolute_error`, `r2_score`.

## Як запустити проєкт

### 1. Вимоги

Для запуску потрібен Python 3.7 або новіший. Переконайтесь, що в системі встановлено `pip`.

### 2. Налаштування середовища

1. Створіть віртуальне середовище:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
    pip install -r requirements.txt
2. Запуск проєкта
```bash
   python Final.py




import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor


# Завантаження даних
file_path = 'get-statistics.txt'
data = pd.read_json(file_path)


# Функція для попередньої обробки даних
def preprocess_data(data, target):
    data['datetime'] = pd.to_datetime(data['datetime'], unit='s')
    data['month'] = data['datetime'].dt.month
    data['weekday'] = data['datetime'].dt.weekday
    data['hour'] = data['datetime'].dt.hour

    # Додавання сезонів
    data['season'] = data['month'] % 12 // 3 + 1

    # Циклічне кодування
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    data['weekday_sin'] = np.sin(2 * np.pi * data['weekday'] / 7)
    data['weekday_cos'] = np.cos(2 * np.pi * data['weekday'] / 7)
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

    if 'direction' in data.columns:
        data = pd.get_dummies(data, columns=['direction'], drop_first=True)

    # Видалення аномалій
    q1 = data[target].quantile(0.25)
    q3 = data[target].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    data = data[(data[target] >= lower_bound) & (data[target] <= upper_bound)]
    data = data.dropna(subset=[target])

    # Заповнення пропущених значень середнім лише для числових колонок
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

    # Визначення ознак
    features = ['temperature', 'humidity', 'wind_speed',
                'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos',
                'hour_sin', 'hour_cos', 'season'] + \
               [col for col in data.columns if 'direction_' in col]

    return data, features



# Функція для оцінки моделей
def evaluate_model(y_test, y_pred, model_name):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} Metrics:\nMSE: {mse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.2f}")


# Функція для побудови графіків
def plot_predictions(y_test, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values[:150], label="Actual", marker='o')
    plt.plot(y_pred[:150], label="Predicted", marker='x')
    plt.title(f"Actual vs Predicted ({model_name})")
    plt.xlabel("Sample Index")
    plt.ylabel("Target")
    plt.legend()
    plt.grid()
    os.makedirs("graphs", exist_ok=True)  # Створити папку "graphs", якщо її не існує
    plt.savefig(f"graphs/{model_name}_predictions_{target}.png")
    plt.close()


# Функція для побудови графіків абсолютного і відносного відхилення
def plot_deviations(y_true, y_pred, model_name, target):
    absolute_deviation = np.abs(y_true - y_pred)
    conditions = [
        y_true < 25,  # Низький рівень
        (y_true >= 25) & (y_true < 50),  # Середній рівень
        y_true >= 50  # Високий рівень
    ]
    levels = ["Low (<25 µg/m³)", "Medium (25-50 µg/m³)", "High (>50 µg/m³)"]

    # Абсолютне відхилення
    plt.figure(figsize=(10, 6))
    for condition, level in zip(conditions, levels):
        plt.scatter(
            y_true[condition],
            absolute_deviation[condition],
            alpha=0.7,
            edgecolor='k',
            label=level
        )
    plt.title(f"Absolute Deviation vs Actual ({model_name} - {target})")
    plt.xlabel(f"Actual {target}")
    plt.ylabel("Absolute Deviation")
    plt.legend()
    plt.grid(True)
    os.makedirs("graphs", exist_ok=True)  # Створити папку "graphs", якщо її не існує
    plt.savefig(f"graphs/{model_name}_absolute_deviation_{target}.png")
    plt.close()

# Функція для тренування моделей
def train_models(X_train, y_train):
    models = {}

    # ---- Random Forest ----
    rf_model = RandomForestRegressor(n_estimators=200) #, random_state=42
    rf_model.fit(X_train, y_train)
    models["Random Forest"] = rf_model

    # ---- XGBoost ----
    xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=10) #, random_state=42
    xgb_model.fit(X_train, y_train)
    models["XGBoost"] = xgb_model

    # ---- LSTM ----
    X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    lstm_model = build_lstm_model((1, X_train.shape[1]))
    lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=1)
    models["LSTM"] = lstm_model

    return models


# Функція для отримання передбачень
def predict_with_models(models, X_test):
    predictions = {}

    # ---- Random Forest ----
    predictions["Random Forest"] = models["Random Forest"].predict(X_test)

    # ---- XGBoost ----
    predictions["XGBoost"] = models["XGBoost"].predict(X_test)

    # ---- LSTM ----
    X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    predictions["LSTM"] = models["LSTM"].predict(X_test_lstm).flatten()

    return predictions


# Функція для створення LSTM моделі
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


# Основний цикл для роботи з цільовими змінними
targets = ['dust_2_5', 'dust_1_0', 'dust_10']
# results = {}

for target in targets:
    print(f"\nProcessing target: {target}\n")

    # Попередня обробка даних
    data, features = preprocess_data(data, target)
    X = data[features]
    y = data[target]

    # Масштабування
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Розбиття даних
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Тренування моделей
    models = train_models(X_train, y_train)

    # Передбачення
    predictions = predict_with_models(models, X_test)

    # Оцінка моделей і побудова графіків
    for model_name, y_pred in predictions.items():
        evaluate_model(y_test, y_pred, model_name)
        plot_predictions(y_test, y_pred, model_name)
        plot_deviations(y_test, y_pred, model_name, target)

    # # Збереження результатів
    # results[target] = predictions

print("Processing complete!")

from datetime import datetime

import dill
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# 1. Функции препроцессинга

def prepare_data_types(df):
    import pandas as pd
    df = df.copy()
    df = df.drop(['device_model', 'utm_keyword'], axis=1, errors='ignore')
    df['visit_date'] = pd.to_datetime(df['visit_date'], utc=True)
    df['visit_time'] = pd.to_timedelta(df['visit_time'])
    return df


def handle_missing_values(df):
    import pandas as pd
    df = df.copy()
    # device_os
    df.loc[(df['device_category'] == 'mobile') & (df['device_os'].isna()), 'device_os'] = 'Android'
    df.loc[(df['device_category'] == 'desktop') & (df['device_os'].isna()), 'device_os'] = 'Windows'
    df.loc[(df['device_category'] == 'tablet') & (df['device_os'].isna()), 'device_os'] = 'Android'

    brand_os = {
        'Apple': 'iOS',
        'Samsung': 'Android',
        'Xiaomi': 'Android',
        'Huawei': 'Android',
        'Realme': 'Android',
        'OPPO': 'Android',
        'Vivo': 'Android',
        'OnePlus': 'Android',
        'Asus': 'Android',
        'Nokia': 'Android',
        'Sony': 'Android',
        'ZTE': 'Android'
    }
    for brand, os in brand_os.items():
        df.loc[df['device_brand'] == brand, 'device_os'] = os

    # device_brand
    df.loc[(df['device_browser'] == 'Safari') & (df['device_brand'].isna()), 'device_brand'] = 'Apple'
    df.loc[(df['device_browser'] == 'Samsung Internet') & (df['device_brand'].isna()), 'device_brand'] = 'Samsung'
    df['device_brand'] = df['device_brand'].fillna('unknown')

    # UTM
    df['utm_adcontent'] = df['utm_adcontent'].fillna('other')
    df['utm_campaign'] = df['utm_campaign'].fillna('other')
    df['utm_source'] = df['utm_source'].fillna('not_set')
    return df


def create_features_with_top(top_cities, top_countries):
    def create_features(df):
        import pandas as pd
        df = df.copy()
        # Временные фичи
        df['visit_hour'] = df['visit_time'].dt.components.hours
        df['visit_month'] = df['visit_date'].dt.month
        df['visit_day_of_week'] = df['visit_date'].dt.dayofweek
        df['is_weekend'] = (df['visit_day_of_week'] >= 5).astype(int)
        df['is_working_hours'] = ((df['visit_hour'] >= 9) & (df['visit_hour'] <= 18)).astype(int)

        # Трафик
        df['is_organic'] = df['utm_medium'].isin(['organic', 'referral', '(none)']).astype(int)
        social_sources = ['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs',
                          'IZEXUFLARCUMynmHNBGo', 'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm']
        df['is_social'] = df['utm_source'].isin(social_sources).astype(int)

        # Гео-фичи
        for city in top_cities:
            df[f'city_{city}'] = (df['geo_city'] == city).astype(int)
        df['city_other'] = (~df['geo_city'].isin(top_cities)).astype(int)

        for country in top_countries:
            df[f'is_{country.lower()}'] = (df['geo_country'] == country).astype(int)

        # Разрешение экрана
        def res_to_area(r):
            if pd.isna(r) or r == '(not set)': return 0
            try: return int(r.split('x')[0]) * int(r.split('x')[1])
            except: return 0
        df['screen_area'] = df['device_screen_resolution'].apply(res_to_area)

        # Лояльность
        df['is_returning_user'] = (df['visit_number'] > 1).astype(int)
        df['user_loyalty'] = pd.cut(df['visit_number'], bins=[0, 1, 3, 10, 500],
                                    labels=['new', 'returning', 'active', 'loyal'])
        return df
    return create_features


def remove_original_columns(df):
    import pandas as pd
    cols_to_drop = [
        'session_id', 'client_id', 'visit_date', 'visit_time',
        'geo_city', 'geo_country',
        'device_screen_resolution'
    ]
    return df.drop(columns=cols_to_drop, errors='ignore')


# 2. MAIN
def main():
    print("SberAuto ML Pipeline")

    # Загрузка уже объединённого датафрейма с таргетом
    ml_data = pd.read_csv('data/merged_sessions.csv')

    # Определяем топы ДО сплита
    top_cities = ml_data['geo_city'].value_counts().head(20).index.tolist()
    top_countries = ml_data['geo_country'].value_counts().head(10).index.tolist()

    # Стратифицированная выборка
    _, sample_data = train_test_split(
        ml_data, test_size=100000, stratify=ml_data['target'], random_state=42
    )

    X = sample_data.drop('target', axis=1)
    y = sample_data['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Сборка пайплайна
    feature_eng = FunctionTransformer(
        create_features_with_top(top_cities, top_countries), validate=False
    )

    full_pipeline = Pipeline([
        ('prepare', FunctionTransformer(prepare_data_types, validate=False)),
        ('impute', FunctionTransformer(handle_missing_values, validate=False)),
        ('features', feature_eng),
        ('remove_raw', FunctionTransformer(remove_original_columns, validate=False)),
        ('encode_scale', ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['visit_number', 'screen_area']),
                ('cat', OneHotEncoder(handle_unknown='ignore', dtype='int8'), [
                    'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent',
                    'device_category', 'device_os', 'device_brand', 'device_browser',
                    'user_loyalty'
                ])
            ],
            remainder='passthrough'
        ))
    ])

    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced',solver='liblinear'),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    }

    best_score = 0
    best_pipe = None
    best_name = ""

    for name, model in models.items():
        print(f"Обучение {name}...")
        pipe = Pipeline([('preprocessor', full_pipeline), ('classifier', model)])
        pipe.fit(X_train, y_train)

        score = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        print(f"{name}: ROC-AUC = {score:.4f}")

        if score > best_score:
            best_score = score
            best_pipe = pipe
            best_name = name

    # Обучение на всех данных
    X_full = ml_data.drop('target', axis=1)
    y_full = ml_data['target']
    best_pipe.fit(X_full, y_full)

    # Сохранение
    with open('sberauto_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'top_cities': top_cities,
            'top_countries': top_countries,
            'metadata': {
                'name': 'SberAuto Conversion Prediction Model',
                'author': 'Denis Tagintsev',
                'version': 1,
                'date': datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'roc_auc': best_score
            }
        }, file)

    print(f" Пайплайн сохранён. Лучшая модель: {best_name}, ROC-AUC: {best_score:.4f}")


if __name__ == '__main__':
    main()
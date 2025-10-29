 Структура проекта

- API приложение (`main.py`) - веб-сервис для предсказаний
- Jupyter ноутбук (`sberauto_analysis.ipynb`) - полный анализ данных и ML
- ML пайплайн (pipeline.py) - обучение модели
- Обученная модель (`sberauto_pipe.pkl`) - готовая модель с ROC-AUC 0.69
- Примеры для тестирования (`data/examples/`) - JSON файлы с результатами предсказания 0|1  

 Запуск проекта

cd "sberauto target action"
pip install "fastapi[all]"
uvicorn main:app --reload


Тестирование API через Postman
1. Method: POST
2. URL: http://localhost:8000/predict
3. Body: raw JSON (скопируйте из папки data/examples/)
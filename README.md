# ПредиктНСО — Flask приложение

## Структура проекта
```
predikt_nso/
  app.py                  ← главный файл Flask
  requirements.txt        ← зависимости
  xgboost_model.pkl       ← модель (добавь сам из ноутбука)
  templates/
    index.html            ← HTML сайта
  static/
    css/
      main.css            ← стили
    js/
      main.js             ← скрипты
```

## Запуск локально

1. Положи `xgboost_model.pkl` в папку `predikt_nso/`

2. Установи зависимости:
```bash
pip install -r requirements.txt
```

3. Запусти:
```bash
python app.py
```

4. Открой в браузере:
```
http://localhost:5000
```

## Деплой на сервер (Render / Railway / VPS)

```bash
gunicorn app:app -b 0.0.0.0:5000
```

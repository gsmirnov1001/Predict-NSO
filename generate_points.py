"""
Генерация статического файла points.js со всеми точками из БД.
Запускать один раз: python generate_points.py
Файл сохраняется в static/js/points.js
"""
import psycopg2
import json
import os

OUTPUT = os.path.join(os.path.dirname(__file__), 'static', 'js', 'points.js')

print("Подключаемся к БД...")
conn = psycopg2.connect(
    host='127.0.0.1',
    user='postgres',
    password='admin',
    database='realty_nso '  # пробел обязателен
)

print("Загружаем точки...")
with conn.cursor() as cur:
    cur.execute("""
        SELECT geo_lat, geo_lon, price, area,
               building_type, level, rooms,
               kitchen_area, object_type
        FROM listings
        WHERE geo_lat IS NOT NULL
          AND geo_lon IS NOT NULL
          AND geo_lat BETWEEN 54.0 AND 56.0
          AND geo_lon BETWEEN 77.0 AND 86.0
        ORDER BY id
    """)
    rows = cur.fetchall()

conn.close()
print(f"Получено строк: {len(rows)}")

bt = {0:'Другой',1:'Панельный',2:'Монолитный',3:'Кирпичный',4:'Блочный',5:'Деревянный'}
ot = {0:'Неизвестно',1:'Вторичка',2:'Новостройка'}

# Компактный формат — массив массивов вместо объектов
# [lat, lon, price, area, bt_idx, level, rooms, kit, ot_idx]
# Это в 3-4 раза меньше чем JSON объекты
points = []
for r in rows:
    points.append([
        round(float(r[0]), 5),
        round(float(r[1]), 5),
        int(r[2]),
        round(float(r[3] or 0), 1),
        int(r[4] or 0),
        int(r[5] or 0),
        int(r[6] or 0),
        round(float(r[7] or 0), 1),
        int(r[8] or 0),
    ])

# Сохраняем как JS переменную
bt_js = json.dumps(bt)
ot_js = json.dumps(ot)
pts_js = json.dumps(points, separators=(',', ':'))

js_content = f"""// Автогенерировано generate_points.py — {len(points)} точек
const DB_BT = {bt_js};
const DB_OT = {ot_js};
const DB_POINTS = {pts_js};
"""

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, 'w', encoding='utf-8') as f:
    f.write(js_content)

size_mb = os.path.getsize(OUTPUT) / 1024 / 1024
print(f"Сохранено: {OUTPUT}")
print(f"Точек: {len(points):,}")
print(f"Размер файла: {size_mb:.1f} МБ")
print("Готово!")

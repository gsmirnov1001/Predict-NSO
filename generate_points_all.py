"""
Генерирует points_all.js — все ~972к точек в компактном формате.
WebGL рендеринг через leaflet.glify держит такое количество легко.
Запуск: python generate_points_all.py
"""
import psycopg2, json, os

OUTPUT = os.path.join(os.path.dirname(__file__), 'static', 'js', 'points_all.js')

print("Подключаемся к БД...")
conn = psycopg2.connect(
    host='127.0.0.1', user='postgres',
    password='admin', database='realty_nso '
)

print("Загружаем все точки...")
with conn.cursor() as cur:
    cur.execute("""
        SELECT geo_lat, geo_lon, price, area,
               building_type, level, rooms, kitchen_area, object_type
        FROM listings
        WHERE geo_lat BETWEEN 54.0 AND 56.0
          AND geo_lon BETWEEN 77.0 AND 86.0
          AND geo_lat IS NOT NULL AND geo_lon IS NOT NULL
        ORDER BY id
    """)
    rows = cur.fetchall()
conn.close()
print(f"Получено: {len(rows):,} строк")

bt = {0:'Другой',1:'Панельный',2:'Монолитный',3:'Кирпичный',4:'Блочный',5:'Деревянный'}
ot = {0:'Неизвестно',1:'Вторичка',2:'Новостройка'}

# Формат: [lat, lon, price, area, bt, level, rooms, kit, ot]
points = [[
    round(float(r[0]),5), round(float(r[1]),5),
    int(r[2]),
    round(float(r[3] or 0),1),
    int(r[4] or 0),
    int(r[5] or 0),
    int(r[6] or 0),
    round(float(r[7] or 0),1),
    int(r[8] or 0),
] for r in rows]

js = 'const ALL_PTS=' + json.dumps(points, separators=(',',':')) + ';'

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, 'w', encoding='utf-8') as f:
    f.write(js)

size = os.path.getsize(OUTPUT)/1024/1024
print(f"Готово: {len(points):,} точек, {size:.1f} МБ → {OUTPUT}")

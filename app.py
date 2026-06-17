"""
ПредиктНСО — Flask приложение
Запуск: python app.py
"""
import os
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.errorhandler(Exception)
def handle_error(e):
    return jsonify({'error': str(e)}), 500

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'xgboost_model.pkl')
model = None
shap_explainer = None

PRICE_LOW  = 2_500_000
PRICE_HIGH = 5_000_000

FEATURE_NAMES = [
    'geo_lat', 'geo_lon', 'building_type', 'level', 'levels',
    'rooms', 'area', 'kitchen_area', 'object_type',
    'level_to_levels', 'year', 'month', 'area_to_rooms',
    'property_type_encoded', 'title_geo_encoded'
]

FEATURE_LABELS = {
    'area':                   {'icon': '📐', 'name': 'Площадь квартиры'},
    'geo_lat':                {'icon': '📍', 'name': 'Широта (расположение север/юг)'},
    'geo_lon':                {'icon': '📍', 'name': 'Долгота (расположение запад/восток)'},
    'year':                   {'icon': '📅', 'name': 'Год постройки'},
    'level_to_levels':        {'icon': '🏢', 'name': 'Положение этажа (отн.)'},
    'rooms':                  {'icon': '🛏',  'name': 'Количество комнат'},
    'kitchen_area':           {'icon': '🍳', 'name': 'Площадь кухни'},
    'object_type':            {'icon': '🏗', 'name': 'Тип рынка'},
    'building_type':          {'icon': '🏛', 'name': 'Тип дома'},
    'month':                  {'icon': '📆', 'name': 'Месяц сделки'},
    'levels':                 {'icon': '🏙', 'name': 'Этажей в доме'},
    'level':                  {'icon': '🔢', 'name': 'Этаж'},
    'area_to_rooms':          {'icon': '📊', 'name': 'Площадь на комнату'},
    'property_type_encoded':  {'icon': '🏠', 'name': 'Тип объекта'},
    'title_geo_encoded':      {'icon': '🗺', 'name': 'Город / район НСО'},
}

# LabelEncoders для новых признаков
import re as _re

_LE_PT = None
_LE_TG = None

def get_encoders():
    global _LE_PT, _LE_TG
    if _LE_PT is None:
        try:
            _LE_PT = joblib.load(os.path.join(os.path.dirname(__file__), 'le_pt.pkl'))
            _LE_TG = joblib.load(os.path.join(os.path.dirname(__file__), 'le_tg.pkl'))
            print("✓ LabelEncoders загружены")
        except Exception as e:
            print(f"⚠ LabelEncoders не найдены: {e}")
    return _LE_PT, _LE_TG

CITIES_NSO = [
    (_re.compile(r'Новосибирский район|Новосибирский р-н', _re.I), 'Новосибирский район'),
    (_re.compile(r'Искитимский район', _re.I),  'Искитимский район'),
    (_re.compile(r'Мошковский район', _re.I),   'Мошковский район'),
    (_re.compile(r'Черепановский район', _re.I),'Черепановский район'),
    (_re.compile(r'Тогучинский район', _re.I),  'Тогучинский район'),
    (_re.compile(r'Колыванский район', _re.I),  'Колыванский район'),
    (_re.compile(r'Ордынский район', _re.I),    'Ордынский район'),
    (_re.compile(r'Краснозёрский район', _re.I),'Краснозёрский район'),
    (_re.compile(r'Новосибирск', _re.I),        'Новосибирск'),
    (_re.compile(r'Бердск', _re.I),             'Бердск'),
    (_re.compile(r'Искитим', _re.I),            'Искитим'),
    (_re.compile(r'Обь', _re.I),            'Обь'),
    (_re.compile(r'Куйбышев', _re.I),           'Куйбышев'),
    (_re.compile(r'Колывань', _re.I),           'Колывань'),
    (_re.compile(r'Барабинск', _re.I),          'Барабинск'),
    (_re.compile(r'Татарск', _re.I),            'Татарск'),
    (_re.compile(r'Карасук', _re.I),            'Карасук'),
    (_re.compile(r'Маслянино', _re.I),          'Маслянино'),
    (_re.compile(r'Мошково', _re.I),            'Мошково'),
    (_re.compile(r'Черепаново', _re.I),         'Черепаново'),
    (_re.compile(r'Тогучин', _re.I),            'Тогучин'),
    (_re.compile(r'Болотное', _re.I),           'Болотное'),
    (_re.compile(r'Чулым', _re.I),              'Чулым'),
    (_re.compile(r'Купино', _re.I),             'Купино'),
    (_re.compile(r'Каргат', _re.I),             'Каргат'),
    (_re.compile(r'Кольцово', _re.I),           'Кольцово'),
    (_re.compile(r'Краснообск', _re.I),         'Краснообск'),
]

def district_to_title_geo(district_name):
    """Маппит название района из UI в title_geo для LabelEncoder."""
    if not district_name:
        return 'Новосибирск'
    for pat, name in CITIES_NSO:
        if pat.search(district_name):
            return name
    return 'Новосибирск'

# Маппинг property_type -> encoded (как в ноутбуке при обучении модели)
PT_MAP = {
    'квартира':          0,
    'апартаменты':       1,
    'квартира-студия':   2,
    'студия':            3,
    'своб. планировка':  4,
}

def encode_features(district_name, property_type='квартира'):
    _, le_tg = get_encoders()
    # property_type_encoded — используем маппинг из ноутбука
    pt_enc = PT_MAP.get(property_type.lower().strip(), 0)
    # title_geo_encoded
    tg_val = district_to_title_geo(district_name)
    tg_enc = 18  # Новосибирск default (индекс в le_tg из processed датасета)
    if le_tg is not None:
        try:
            tg_enc = int(le_tg.transform([tg_val])[0])
        except:
            tg_enc = 18
    return pt_enc, tg_enc

try:
    model = joblib.load(MODEL_PATH)
    print(f"✓ Модель загружена: {MODEL_PATH}")
    try:
        import shap
        shap_explainer = shap.TreeExplainer(model)
        print("✓ SHAP explainer готов")
    except Exception as e:
        print(f"⚠ SHAP недоступен: {e}")
except Exception as e:
    print(f"✗ Модель не найдена: {e}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Модель не загружена'}), 500

    p = request.get_json(force=True)

    level  = float(p.get('level',  5))
    levels = float(p.get('levels', 9))
    area   = float(p.get('area',   54))
    rooms  = float(p.get('rooms',  2))
    year   = float(p.get('year',   2019))
    month  = float(p.get('month',  6))

    ltl = level / levels if levels > 0 else 0.5
    atr = area  / rooms  if rooms  > 0 else area

    district = p.get('district_name', '')
    pt_enc, tg_enc = encode_features(district, 'квартира')

    X = np.array([[
        float(p.get('geo_lat',       54.99)),
        float(p.get('geo_lon',       82.90)),
        float(p.get('building_type', 1)),
        level, levels, rooms, area,
        float(p.get('kitchen_area',  10)),
        float(p.get('object_type',   1)),
        ltl, year, month, atr,
        pt_enc, tg_enc,
    ]])

    price = float(model.predict(X)[0])
    price = max(500_000, min(50_000_000, price))
    price_rounded = round(price, -3)

    # Сегмент
    if price < PRICE_LOW:
        segment = {'title': 'Бюджетный сегмент',
                   'desc': 'Объект ниже среднего по НСО.',
                   'icon': '💛', 'cls': 'vs-red'}
    elif price < PRICE_HIGH:
        segment = {'title': 'Средний сегмент',
                   'desc': 'Цена соответствует медиане рынка НСО.',
                   'icon': '💚', 'cls': 'vs-amber'}
    else:
        segment = {'title': 'Высокобюджетный объект',
                   'desc': 'Объект выше среднего по НСО.',
                   'icon': '🏆', 'cls': 'vs-green'}

    # SHAP факторы
    factors = []
    if shap_explainer is not None:
        try:
            shap_vals = shap_explainer.shap_values(X)[0]
            indexed = sorted(enumerate(shap_vals), key=lambda x: abs(x[1]), reverse=True)

            for feat_idx, shap_val in indexed[:5]:
                fname   = FEATURE_NAMES[feat_idx]
                pct     = round(float(shap_val) / price * 100, 1)
                info    = FEATURE_LABELS.get(fname, {'icon': '📊', 'name': fname})
                val     = float(X[0][feat_idx])

                if fname == 'area':
                    label = f"{info['name']} ({val:.0f} м²)"
                elif fname == 'year':
                    label = f"{info['name']} ({val:.0f} г.)"
                elif fname == 'rooms':
                    label = f"{info['name']} ({val:.0f})"
                elif fname == 'object_type':
                    label = f"{info['name']} ({'новостройка' if val==2 else 'вторичка'})"
                elif fname == 'building_type':
                    types = {0:'другой',1:'панель',2:'монолит',3:'кирпич',4:'блок',5:'дерево'}
                    label = f"{info['name']} ({types.get(int(val), str(int(val)))})"
                elif fname == 'level_to_levels':
                    label = f"Этаж ({level:.0f} из {levels:.0f})"
                else:
                    label = info['name']

                factors.append({
                    'icon':   info['icon'],
                    'label':  label,
                    'pct':    float(pct),
                    'impact': 'pos' if pct > 2 else ('neg' if pct < -2 else 'neu'),
                    'bar':    float(min(95, abs(pct) * 2.5)),
                })
        except Exception as e:
            print(f"SHAP error: {e}")

    return jsonify({
        'price':       price_rounded,
        'pricePerSqm': round(price_rounded / area),
        'priceLow':    round(price * 0.927, -3),
        'priceHigh':   round(price * 1.073, -3),
        'segment':     segment,
        'factors':     factors,
    })


# ── Загружаем DataFrame один раз при старте ──────────────────────────────
import pandas as pd

_DF = None

def get_df():
    global _DF
    if _DF is None:
        csv_path = os.path.join(os.path.dirname(__file__), 'pars_dataset_processed.csv')
        _DF = pd.read_csv(csv_path)
        # Переименовываем колонки под единый формат
        col_map = {'geo_lat':'lat','geo_lon':'lon','level':'level','levels':'levels'}
        for old_c, new_c in col_map.items():
            if old_c in _DF.columns and new_c not in _DF.columns:
                _DF = _DF.rename(columns={old_c: new_c})
        print(f"✓ DataFrame загружен: {len(_DF):,} строк, колонки: {list(_DF.columns[:6])}")
    return _DF

try:
    get_df()
except Exception as e:
    print(f"✗ DataFrame не загружен: {e}")


@app.route('/points')
def points():
    try:
        df = get_df()

        min_lat = float(request.args.get('minLat', 54.0))
        max_lat = float(request.args.get('maxLat', 56.0))
        min_lon = float(request.args.get('minLon', 77.0))
        max_lon = float(request.args.get('maxLon', 86.0))
        zoom    = int(request.args.get('zoom', 10))

        # Фильтр по видимой области — как советовал ChatGPT
        lat_col = 'lat' if 'lat' in df.columns else 'geo_lat'
        lon_col = 'lon' if 'lon' in df.columns else 'geo_lon'
        mask = (
            (df[lat_col] >= min_lat) & (df[lat_col] <= max_lat) &
            (df[lon_col] >= min_lon) & (df[lon_col] <= max_lon)
        )
        result = df[mask]

        # Лимит по зуму — больше точек при приближении
        limit = 2000 if zoom <= 9 else 5000 if zoom <= 11 else 10000 if zoom <= 13 else 20000

        if len(result) > limit:
            result = result.sample(limit, random_state=42)

        BT = {0:'Другой',1:'Панельный',2:'Монолитный',3:'Кирпичный',4:'Блочный',5:'Деревянный'}
        OT = {0:'Неизвестно',1:'Вторичка',2:'Новостройка'}

        lat_col = 'lat' if 'lat' in df.columns else 'geo_lat'
        lon_col = 'lon' if 'lon' in df.columns else 'geo_lon'
        points_list = [
            {
                'lat':   round(float(getattr(r, lat_col)), 5),
                'lon':   round(float(getattr(r, lon_col)), 5),
                'price': int(r.price),
                'area':  round(float(r.area), 1),
                'bt':    BT.get(int(r.building_type), '—'),
                'level': int(getattr(r, 'level', 1)),
                'rooms': int(r.rooms),
                'kit':   round(float(r.kitchen_area), 1),
                'ot':    OT.get(int(r.object_type), '—'),
            }
            for r in result.itertuples()
        ]

        return jsonify({'points': points_list, 'total': len(points_list)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """Полный SHAP-разбор цены для страницы 'Разбор по факторам'."""
    if model is None:
        return jsonify({'error': 'Модель не загружена'}), 500

    p = request.get_json(force=True)

    level  = float(p.get('level',   5))
    levels = float(p.get('levels',  9))
    area   = float(p.get('area',   42))
    rooms  = float(p.get('rooms',   2))
    year   = float(p.get('year', 2003))
    month  = float(p.get('month',   6))
    kit    = float(p.get('kitchen_area', 9))
    geo_lat = float(p.get('geo_lat', 54.99))
    geo_lon = float(p.get('geo_lon', 82.90))
    btype  = int(p.get('building_type', 1))
    otype  = int(p.get('object_type',   1))
    listed = float(p.get('listed_price', 0))

    ltl = level / levels if levels > 0 else 0.5
    atr = area  / rooms  if rooms  > 0 else area

    district = p.get('district_name', '')
    pt_enc, tg_enc = encode_features(district, 'квартира')

    X = np.array([[geo_lat, geo_lon, btype, level, levels, rooms, area,
                   kit, otype, ltl, year, month, atr, pt_enc, tg_enc]])

    price = float(model.predict(X)[0])
    price = max(500_000, min(50_000_000, price))
    price_rounded = round(price, -3)

    # Разница с объявлением
    diff = listed - price_rounded if listed > 0 else 0
    diff_pct = round(diff / price_rounded * 100, 1) if price_rounded > 0 else 0

    if diff_pct > 10:
        verdict = {'status': 'high', 'icon': '⚠️', 'title': 'Цена завышена',
                   'sub': f'Продавец просит на {abs(diff_pct):.0f}% больше рынка. Есть смысл поторговаться.'}
    elif diff_pct < -8:
        verdict = {'status': 'low', 'icon': '✅', 'title': 'Выгодное предложение',
                   'sub': f'Цена ниже рынка на {abs(diff_pct):.0f}%. Хорошая возможность — проверьте документы.'}
    else:
        verdict = {'status': 'ok', 'icon': '👍', 'title': 'Справедливая цена',
                   'sub': 'Цена соответствует рыночной. Объявление честное.'}

    # SHAP — полный waterfall
    factors = []
    base_value = price_rounded  # fallback

    READABLE = {
        'geo_lat':         {'icon': '📍', 'name': 'Широта (расположение север/юг)'},
        'geo_lon':         {'icon': '📍', 'name': 'Долгота (расположение запад/восток)'},
        'building_type':   {'icon': '🏛', 'name': 'Тип дома'},
        'level':           {'icon': '🔢', 'name': 'Этаж'},
        'levels':          {'icon': '🏙', 'name': 'Этажей в доме'},
        'rooms':           {'icon': '🛏', 'name': 'Количество комнат'},
        'area':            {'icon': '📐', 'name': 'Площадь квартиры'},
        'kitchen_area':    {'icon': '🍳', 'name': 'Площадь кухни'},
        'object_type':     {'icon': '🏗', 'name': 'Тип рынка'},
        'level_to_levels': {'icon': '🏢', 'name': 'Положение этажа (отн.)'},
        'year':            {'icon': '📅', 'name': 'Год постройки'},
        'month':           {'icon': '📆', 'name': 'Месяц сделки'},
        'area_to_rooms':   {'icon': '📊', 'name': 'Площадь на комнату'},
    }
    BT_NAMES = {0:'другой',1:'панель',2:'монолит',3:'кирпич',4:'блок',5:'дерево'}
    OT_NAMES = {1:'вторичка',2:'новостройка'}

    def val_label(fname, v):
        if fname == 'area':            return f'{v:.0f} м²'
        if fname == 'kitchen_area':    return f'{v:.0f} м² кухня'
        if fname == 'year':            return f'{v:.0f} г.'
        if fname == 'rooms':           return f'{v:.0f} комн.'
        if fname == 'level':           return f'{v:.0f} эт.'
        if fname == 'levels':          return f'{v:.0f} эт. в доме'
        if fname == 'level_to_levels': return f'{v:.2f} (эт./всего)'
        if fname == 'area_to_rooms':   return f'{v:.1f} м²/комн.'
        if fname == 'building_type':   return BT_NAMES.get(int(v), str(int(v)))
        if fname == 'object_type':     return OT_NAMES.get(int(v), str(int(v)))
        if fname in ('geo_lat','geo_lon'): return f'{v:.4f}'
        return str(round(v, 2))

    if shap_explainer is not None:
        try:
            sv = shap_explainer.shap_values(X)
            shap_vals = sv[0] if hasattr(sv, '__len__') and not isinstance(sv, np.ndarray) else sv[0]
            base_value = float(shap_explainer.expected_value
                               if not hasattr(shap_explainer.expected_value, '__len__')
                               else shap_explainer.expected_value[0])

            indexed = sorted(enumerate(shap_vals), key=lambda x: abs(x[1]), reverse=True)

            for feat_idx, sv_val in indexed[:8]:
                fname = FEATURE_NAMES[feat_idx]
                info  = READABLE.get(fname, {'icon': '📊', 'name': fname})
                raw_v = float(X[0][feat_idx])
                sv_rub = float(sv_val)
                pct = round(sv_rub / price_rounded * 100, 1) if price_rounded else 0

                factors.append({
                    'icon':      info['icon'],
                    'name':      info['name'],
                    'value':     val_label(fname, raw_v),
                    'shap_rub':  round(sv_rub, -2),
                    'pct':       pct,
                    'direction': 'pos' if sv_rub > 0 else 'neg',
                    'bar':       round(min(100, abs(pct) * 3), 1),
                })
        except Exception as e:
            print(f'SHAP analyze error: {e}')

    suggest = round(price_rounded * 1.02 / 10000) * 10000 if diff_pct > 5 else None

    return jsonify({
        'model_price':   price_rounded,
        'price_per_sqm': round(price_rounded / area),
        'price_low':     round(price * 0.927, -3),
        'price_high':    round(price * 1.073, -3),
        'listed_price':  listed,
        'diff_rub':      round(diff, -3),
        'diff_pct':      diff_pct,
        'verdict':       verdict,
        'suggest':       suggest,
        'base_value':    round(base_value, -3),
        'factors':       factors,
    })




@app.route('/similar', methods=['POST'])
def similar():
    """3 похожие квартиры из датасета."""
    try:
        df = get_df()
        p = request.get_json(force=True)
        lat   = float(p.get('geo_lat', 54.99))
        lon   = float(p.get('geo_lon', 82.90))
        area  = float(p.get('area', 54))
        rooms = int(p.get('rooms', 2))

        # Фильтр: тот же тип комнат, площадь ±20%, координаты ±0.25 градуса
        lat_col = 'lat' if 'lat' in df.columns else 'geo_lat'
        lon_col = 'lon' if 'lon' in df.columns else 'geo_lon'
        mask = (
            (df['rooms'] == rooms) &
            (df['area']  >= area * 0.80) &
            (df['area']  <= area * 1.20) &
            (df[lat_col] >= lat - 0.25) &
            (df[lat_col] <= lat + 0.25) &
            (df[lon_col] >= lon - 0.25) &
            (df[lon_col] <= lon + 0.25) &
            (df['price'] > 300_000)
        )
        result = df[mask]

        if len(result) < 3:
            mask2 = (
                (df['rooms'] == rooms) &
                (df['area']  >= area * 0.70) &
                (df['area']  <= area * 1.30) &
                (df['price'] > 300_000)
            )
            result = df[mask2]

        if len(result) < 3:
            result = df[(df['rooms'] == rooms) & (df['price'] > 300_000)]

        if len(result) == 0:
            return jsonify({'items': []})

        sample = result.sample(min(3, len(result)), random_state=42)

        BT = {0:'другой',1:'панель',2:'монолит',3:'кирпич',4:'блок',5:'дерево'}
        OT = {1:'вторичка',2:'новостройка'}

        lat_col = 'lat' if 'lat' in df.columns else 'geo_lat'
        lon_col = 'lon' if 'lon' in df.columns else 'geo_lon'
        items = [{
            'price':  int(r.price),
            'area':   round(float(r.area), 1),
            'rooms':  int(r.rooms),
            'level':  int(getattr(r, 'level', 1)),
            'bt':     BT.get(int(r.building_type), '—'),
            'ot':     OT.get(int(r.object_type), '—'),
            'ppm':    round(r.price / r.area) if r.area > 0 else 0,
            'lat':    round(float(getattr(r, lat_col)), 4),
            'lon':    round(float(getattr(r, lon_col)), 4),
        } for r in sample.itertuples()]

        return jsonify({'items': items})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status':       'ok',
        'model_loaded': model is not None,
        'shap_ready':   shap_explainer is not None,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

from pathlib import Path
from typing import Any
from datetime import datetime
import html
import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from jinja2 import DictLoader, Environment, select_autoescape
import uvicorn

APP_NAME = "حِرَفي دمياط"
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "craftsmen.csv"
HOST = "0.0.0.0"
PORT = 8000

REQUIRED_COLUMNS = ["id", "name", "craft", "services", "area", "phone", "whatsapp", "price", "address", "description"]
OPTIONAL_COLUMNS = ["image", "experience", "working_hours", "portfolio", "verified"]

CRAFT_COLORS = {
    "السباكة": "#1687E8", "سباكة": "#1687E8",
    "النجارة": "#A66B35", "نجارة": "#A66B35", "نجارة مسلحة": "#7A4B25",
    "الكهرباء": "#E9A000", "كهرباء": "#E9A000",
    "صيانة الأجهزة الكهربائية": "#13A89E", "صيانة الأجهزة": "#13A89E",
    "النقاشة والدهانات": "#8C61C9", "نقاشة": "#8C61C9", "دهانات": "#8C61C9",
    "البناء والمحارة": "#D96C42", "البناء": "#D96C42", "محارة": "#D96C42",
    "الحدادة": "#333333", "حدادة": "#333333",
    "الألوميتال": "#607D8B", "ألوميتال": "#607D8B",
    "السيراميك": "#1976A3", "ديكور": "#9B59B6", "الأثاث": "#795548",
}
DEFAULT_CRAFT_COLOR = "#008F7A"
CRAFT_ICONS = {
    "السباكة": "🔧", "سباكة": "🔧", "النجارة": "🪚", "نجارة": "🪚",
    "الكهرباء": "⚡", "كهرباء": "⚡", "الحدادة": "⚒️", "حدادة": "⚒️",
    "النقاشة والدهانات": "🎨", "نقاشة": "🎨", "دهانات": "🎨",
    "البناء والمحارة": "🧱", "البناء": "🧱", "محارة": "🧱",
    "صيانة الأجهزة الكهربائية": "🔌", "صيانة الأجهزة": "🔌",
    "الألوميتال": "🪟", "ألوميتال": "🪟", "السيراميك": "▦", "ديكور": "✨", "الأثاث": "🪑"
}

_cache = None
_cache_mtime = None
_csv_error = ""

def safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    value = str(value).strip()
    return "" if value.lower() in {"nan", "none", "null", "<na>"} else value

def craft_color(craft: str) -> str:
    return CRAFT_COLORS.get(safe_text(craft), DEFAULT_CRAFT_COLOR)

def craft_icon(craft: str) -> str:
    return CRAFT_ICONS.get(safe_text(craft), "🛠️")

def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=False)
        except UnicodeDecodeError:
            continue
    raise ValueError("تعذر قراءة ملف البيانات")

def validate(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("الأعمدة المفقودة: " + ", ".join(missing))

def load_craftsmen(force_reload=False) -> pd.DataFrame:
    global _cache, _cache_mtime, _csv_error
    if not CSV_PATH.exists():
        _cache = pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)
        _cache_mtime = None
        _csv_error = f"ملف {CSV_PATH.name} غير موجود بجانب ملف البرنامج."
        return _cache
    mtime = CSV_PATH.stat().st_mtime
    if not force_reload and _cache is not None and _cache_mtime == mtime:
        return _cache
    try:
        df = read_csv(CSV_PATH)
        validate(df)
        for col in OPTIONAL_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df.columns = [safe_text(c) for c in df.columns]
        df = df.fillna("")
        _cache, _cache_mtime, _csv_error = df, mtime, ""
        return df
    except Exception as exc:
        _csv_error = f"تعذر تحميل البيانات: {exc}"
        return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)

def prepare(df):
    result = []
    for _, row in df.iterrows():
        item = {c: safe_text(row.get(c, "")) for c in set(REQUIRED_COLUMNS + OPTIONAL_COLUMNS)}
        item["craft_color"] = craft_color(item["craft"])
        item["craft_icon"] = craft_icon(item["craft"])
        item["verified"] = item.get("verified", "").lower() in {"true", "1", "yes", "نعم", "محقق", "موثق"}
        result.append(item)
    return result

def unique(df, col):
    if col not in df.columns:
        return []
    return sorted({safe_text(x) for x in df[col] if safe_text(x)}, key=lambda x: x)

def filter_rows(df, q="", craft="", area=""):
    out = df.copy()
    q, craft, area = safe_text(q).lower(), safe_text(craft).lower(), safe_text(area).lower()
    if q:
        mask = pd.Series(False, index=out.index)
        for col in ["name", "craft", "services", "area", "description"]:
            if col in out.columns:
                mask |= out[col].map(safe_text).str.lower().str.contains(q, regex=False)
        out = out[mask]
    if craft:
        out = out[out["craft"].map(safe_text).str.lower() == craft]
    if area:
        out = out[out["area"].map(safe_text).str.lower() == area]
    return out

def stats(df):
    return {"total": len(df), "crafts": df["craft"].nunique() if "craft" in df else 0, "areas": df["area"].nunique() if "area" in df else 0}

def find_by_id(df, ident):
    if "id" not in df.columns: return None
    rows = df[df["id"].map(safe_text) == safe_text(ident)]
    return prepare(rows)[0] if not rows.empty else None

app = FastAPI(title=APP_NAME, description="دليل إلكتروني للحرفيين وأصحاب الخدمات في محافظة دمياط", version="2.0.0")

BASE_TEMPLATE = '''<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="حِرَفي دمياط - دليل الحرفيين وأصحاب الخدمات في محافظة دمياط"><title>{{ title }} | حِرَفي دمياط</title><link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&display=swap" rel="stylesheet"><style>{{ css }}</style></head><body><header><div class="container nav"><a class="brand" href="/"><span class="brand-mark">🛠️</span><span><b>حِرَفي دمياط</b><small>دليلك لأصحاب الحرف والخدمات</small></span></a><nav><a href="/">الرئيسية</a><a href="/craftsmen">الحرفيون</a></nav></div></header><main>{% block content %}{% endblock %}</main><footer><div class="container footer-grid"><div><b>🛠️ حِرَفي دمياط</b><p>دليل إلكتروني يساعدك في الوصول إلى أصحاب الحرف والخدمات في محافظة دمياط.</p></div><div><b>روابط سريعة</b><a href="/">الرئيسية</a><a href="/craftsmen">كل الحرفيين</a></div><div><b>عن الدليل</b><p>الموقع لعرض بيانات الحرفيين والخدمات ووسائل التواصل، وليس منصة حجز أو ضمان لجودة الخدمة.</p></div></div><div class="bottom">© {{ year }} حِرَفي دمياط</div></footer></body></html>'''

CSS = '''*{box-sizing:border-box}body{margin:0;font-family:Cairo,Arial,sans-serif;background:#f7f8fa;color:#18212b}a{text-decoration:none;color:inherit}.container{width:min(1120px,92%);margin:auto}header{background:#fff;border-bottom:1px solid #e8ebef;position:sticky;top:0;z-index:10}.nav{min-height:74px;display:flex;align-items:center;justify-content:space-between;gap:25px}.brand{display:flex;align-items:center;gap:11px}.brand-mark{width:44px;height:44px;border-radius:14px;background:#008f7a;color:white;display:grid;place-items:center;font-size:23px}.brand b{display:block;font-size:19px}.brand small{display:block;color:#78828d;font-size:11px}.nav nav{display:flex;gap:25px;font-weight:700}.nav nav a:hover{color:#008f7a}.hero{padding:72px 0;background:linear-gradient(135deg,#f0fbf8,#fff 55%,#fff6e8)}.hero-grid{display:grid;grid-template-columns:1.15fr .85fr;gap:45px;align-items:center}.badge{display:inline-flex;gap:8px;padding:8px 13px;border-radius:99px;background:#e2f7f2;color:#007b6a;font-weight:800;font-size:13px}.hero h1{font-size:clamp(34px,5vw,58px);line-height:1.2;margin:18px 0 12px}.hero h1 span{color:#008f7a}.hero p{color:#64707c;font-size:17px;line-height:1.9}.search{display:flex;background:white;border:1px solid #e2e7eb;border-radius:18px;padding:7px;margin-top:24px;box-shadow:0 14px 40px #0000000b}.search input{flex:1;border:0;outline:0;padding:14px;font:inherit}.btn{border:0;border-radius:13px;padding:12px 20px;background:#008f7a;color:white;font:inherit;font-weight:800;cursor:pointer}.hero-art{min-height:300px;border-radius:35px;background:linear-gradient(145deg,#fff,#eef8f5);display:grid;place-items:center;font-size:115px;box-shadow:0 25px 70px #008f7a18}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin-top:-25px;position:relative}.stat{background:white;border:1px solid #e7eaed;border-radius:20px;padding:22px;display:flex;gap:14px;align-items:center;box-shadow:0 10px 30px #00000008}.stat strong{font-size:28px;display:block}.stat span{color:#697580}.section{padding:60px 0}.heading{display:flex;align-items:end;justify-content:space-between;margin-bottom:25px}.heading h2{margin:4px 0;font-size:30px}.muted{color:#72808b}.craft-grid,.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.craft-card{background:white;border:1px solid #e7eaed;border-radius:20px;padding:20px;display:flex;align-items:center;gap:14px;transition:.2s}.craft-card:hover,.card:hover{transform:translateY(-3px);box-shadow:0 15px 35px #0000000c}.craft-icon{width:52px;height:52px;border-radius:16px;display:grid;place-items:center;color:white;font-size:25px}.craft-card small{display:block;color:#7a858f;margin-top:3px}.card{background:white;border:1px solid #e5e9ed;border-radius:22px;overflow:hidden;transition:.2s}.card-top{height:92px;display:flex;align-items:center;gap:14px;padding:18px}.avatar{width:60px;height:60px;border-radius:18px;background:#eef3f4;display:grid;place-items:center;font-size:30px;overflow:hidden}.avatar img{width:100%;height:100%;object-fit:cover}.card-body{padding:18px}.card h3{margin:0 0 6px}.tag{display:inline-block;border-radius:99px;padding:5px 10px;font-size:11px;color:white;font-weight:800}.info{color:#68747f;font-size:13px;line-height:2}.actions{display:flex;gap:8px;margin-top:13px}.action{flex:1;text-align:center;padding:10px;border-radius:11px;font-weight:800;background:#f0f4f4}.action.primary{background:#008f7a;color:white}.directory{padding:50px 0}.filters{background:white;border:1px solid #e4e8eb;border-radius:20px;padding:18px;display:grid;grid-template-columns:2fr 1fr 1fr auto;gap:12px;margin-bottom:25px}.filters input,.filters select{width:100%;padding:13px;border:1px solid #dfe4e8;border-radius:12px;font:inherit;background:#fff}.profile{padding:45px 0}.profile-layout{display:grid;grid-template-columns:300px 1fr;gap:22px}.side,.main-box{background:white;border:1px solid #e3e7ea;border-radius:24px;padding:25px}.side{text-align:center}.big-avatar{height:220px;border-radius:20px;background:#f0f4f4;display:grid;place-items:center;font-size:90px;overflow:hidden;margin-bottom:15px}.big-avatar img{width:100%;height:100%;object-fit:cover}.verified{display:inline-block;background:#e5f7ef;color:#087b54;padding:6px 11px;border-radius:99px;font-size:12px;font-weight:800;margin-bottom:15px}.profile h1{margin:5px 0}.details{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin-top:25px}.detail{background:#f7f9fa;padding:15px;border-radius:15px}.detail b{display:block;margin-bottom:5px}.footer-grid{display:grid;grid-template-columns:2fr 1fr 1.5fr;gap:35px;padding:45px 0}.site-footer a{display:block;margin-top:8px}.footer-grid p{color:#78838d;line-height:1.9}.bottom{border-top:1px solid #e6e9eb;text-align:center;padding:18px;color:#7b858e;font-size:13px}@media(max-width:800px){.hero-grid,.profile-layout{grid-template-columns:1fr}.craft-grid,.cards{grid-template-columns:1fr 1fr}.filters{grid-template-columns:1fr}.stats{grid-template-columns:1fr}.footer-grid{grid-template-columns:1fr}.nav nav{gap:10px}}@media(max-width:520px){.craft-grid,.cards{grid-template-columns:1fr}.hero{padding:45px 0}.hero-art{min-height:220px;font-size:80px}.details{grid-template-columns:1fr}}'''

def render(name, context):
    env = Environment(loader=DictLoader({"base.html":BASE_TEMPLATE, **TEMPLATES}), autoescape=select_autoescape(["html"]))
    context = {**context, "css": CSS, "year": datetime.now().year}
    return HTMLResponse(env.get_template(name).render(**context))

TEMPLATES = {}
TEMPLATES["home.html"] = '''{% extends "base.html" %}{% block content %}<section class="hero"><div class="container hero-grid"><div><span class="badge">🛠️ دليل الحرف والخدمات في دمياط</span><h1>ابحث عن <span>الحِرفي المناسب</span> في دمياط</h1><p>اكتشف أصحاب الحرف والخدمات حسب نوع الحرفة والمنطقة، وتواصل معهم مباشرة.</p><form class="search" action="/craftsmen" method="get"><input name="q" placeholder="ابحث باسم الحرفي أو الحرفة أو المنطقة..."><button class="btn">بحث</button></form></div><div class="hero-art">🛠️</div></div></section><div class="container"><div class="stats"><div class="stat"><span>👷</span><div><strong>{{ s.total }}</strong><span>حرفي</span></div></div><div class="stat"><span>🧰</span><div><strong>{{ s.crafts }}</strong><span>حرفة</span></div></div><div class="stat"><span>📍</span><div><strong>{{ s.areas }}</strong><span>منطقة</span></div></div></div></div><section class="section"><div class="container"><div class="heading"><div><h2>الحرف والخدمات</h2><p class="muted">اختار الحرفة التي تبحث عنها</p></div><a class="btn" href="/craftsmen">كل الحرفيين</a></div><div class="craft-grid">{% for c in crafts %}<a class="craft-card" href="/craftsmen?craft={{ c|urlencode }}"><span class="craft-icon" style="background:{{ colors[c] }}">{{ icons[c] }}</span><span><b>{{ c }}</b><small>عرض الحرفيين</small></span></a>{% endfor %}</div></div></section><section class="section" style="padding-top:10px"><div class="container"><div class="heading"><div><h2>حرفيون في الدليل</h2><p class="muted">نماذج من أصحاب الحرف المسجلين</p></div></div><div class="cards">{% for x in featured %}{% include "card.html" %}{% endfor %}</div></div></section>{% endblock %}'''
TEMPLATES["card.html"] = '''<article class="card"><div class="card-top" style="border-top:5px solid {{ x.craft_color }}"><div class="avatar">{% if x.image %}<img src="{{ x.image }}" alt="{{ x.name }}">{% else %}{{ x.craft_icon }}{% endif %}</div><div><h3>{{ x.name }}</h3><span class="tag" style="background:{{ x.craft_color }}">{{ x.craft }}</span></div></div><div class="card-body"><div class="info">📍 {{ x.area or "غير محدد" }}<br>🧰 {{ x.services or "خدمات حرفية" }}<br>{% if x.experience %}⏱️ {{ x.experience }}{% endif %}</div><div class="actions"><a class="action primary" href="/craftsmen/{{ x.id }}">التفاصيل</a>{% if x.phone %}<a class="action" href="tel:{{ x.phone }}">📞 اتصال</a>{% endif %}</div></div></article>'''
TEMPLATES["directory.html"] = '''{% extends "base.html" %}{% block content %}<section class="directory"><div class="container"><div class="heading"><div><h2>دليل الحرفيين</h2><p class="muted">{{ count }} نتيجة مطابقة للبحث</p></div></div><form class="filters" method="get"><input name="q" value="{{ q }}" placeholder="اسم الحرفي، الحرفة، الخدمة، المنطقة..."><select name="craft"><option value="">كل الحرف</option>{% for c in crafts %}<option value="{{ c }}" {% if c==craft %}selected{% endif %}>{{ c }}</option>{% endfor %}</select><select name="area"><option value="">كل المناطق</option>{% for a in areas %}<option value="{{ a }}" {% if a==area %}selected{% endif %}>{{ a }}</option>{% endfor %}</select><button class="btn">بحث</button></form>{% if error %}<div style="padding:15px;background:#fff0f0;border-radius:12px;margin-bottom:20px">{{ error }}</div>{% endif %}<div class="cards">{% for x in items %}{% include "card.html" %}{% else %}<div style="grid-column:1/-1;text-align:center;padding:60px;background:white;border-radius:20px">🔎<h2>لا توجد نتائج</h2><p class="muted">جرّب تغيير كلمات البحث أو الفلاتر.</p></div>{% endfor %}</div></div></section>{% endblock %}'''
TEMPLATES["profile.html"] = '''{% extends "base.html" %}{% block content %}<section class="profile"><div class="container"><a class="muted" href="/craftsmen">← العودة إلى دليل الحرفيين</a><div class="profile-layout" style="margin-top:18px"><aside class="side"><div class="big-avatar">{% if x.image %}<img src="{{ x.image }}" alt="{{ x.name }}">{% else %}{{ x.craft_icon }}{% endif %}</div>{% if x.verified %}<span class="verified">✓ بيانات موثقة</span>{% endif %}<h2>{{ x.name }}</h2><span class="tag" style="background:{{ x.craft_color }}">{{ x.craft }}</span><div class="actions" style="margin-top:20px">{% if x.phone %}<a class="action primary" href="tel:{{ x.phone }}">📞 اتصال</a>{% endif %}{% if x.whatsapp %}<a class="action" target="_blank" href="https://wa.me/{{ x.whatsapp|replace('+','')|replace(' ','') }}">💬 واتساب</a>{% endif %}</div></aside><div class="main-box"><span class="tag" style="background:{{ x.craft_color }}">{{ x.craft_icon }} {{ x.craft }}</span><h1>{{ x.name }}</h1><p class="muted">{{ x.description or "حرفي يقدم خدمات مهنية في محافظة دمياط." }}</p><div class="details"><div class="detail"><b>🧰 الخدمات</b>{{ x.services or "غير محددة" }}</div><div class="detail"><b>📍 المنطقة</b>{{ x.area or "غير محددة" }}</div><div class="detail"><b>🏠 العنوان</b>{{ x.address or "غير محدد" }}</div><div class="detail"><b>💰 السعر</b>{{ x.price or "يحدد حسب الخدمة" }}</div><div class="detail"><b>⏱️ الخبرة</b>{{ x.experience or "غير محددة" }}</div><div class="detail"><b>🕐 مواعيد العمل</b>{{ x.working_hours or "غير محددة" }}</div></div>{% if x.portfolio %}<div style="margin-top:25px"><h3>نماذج الأعمال</h3><p>{{ x.portfolio }}</p></div>{% endif %}</div></div></div></section>{% endblock %}'''
TEMPLATES["error.html"] = '''{% extends "base.html" %}{% block content %}<section class="section"><div class="container" style="text-align:center;background:white;padding:70px 20px;border-radius:25px"><div style="font-size:70px">🛠️</div><h1>{{ code }}</h1><h2>{{ message }}</h2><a class="btn" href="/craftsmen">العودة للدليل</a></div></section>{% endblock %}'''

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    df = load_craftsmen()
    cs = unique(df, "craft")
    colors = {c: craft_color(c) for c in cs}; icons = {c: craft_icon(c) for c in cs}
    return render("home.html", {"request":request,"title":"الرئيسية","s":stats(df),"crafts":cs,"colors":colors,"icons":icons,"featured":prepare(df.head(6))})

@app.get("/craftsmen", response_class=HTMLResponse)
async def craftsmen(request: Request, q: str = Query("", max_length=100), craft: str = Query("", max_length=100), area: str = Query("", max_length=100)):
    df = load_craftsmen(); filtered = filter_rows(df,q,craft,area)
    return render("directory.html", {"request":request,"title":"الحرفيون","items":prepare(filtered),"count":len(filtered),"crafts":unique(df,"craft"),"areas":unique(df,"area"),"q":html.escape(q),"craft":html.escape(craft),"area":html.escape(area),"error":_csv_error})

@app.get("/craftsmen/{craftsman_id}", response_class=HTMLResponse)
async def profile(request: Request, craftsman_id: str):
    x = find_by_id(load_craftsmen(), craftsman_id)
    if not x:
        return render("error.html", {"request":request,"title":"غير موجود","code":404,"message":"الحرفي غير موجود في الدليل."})
    return render("profile.html", {"request":request,"title":x["name"],"x":x})

@app.get("/reload")
async def reload_csv():
    load_craftsmen(True)
    return RedirectResponse("/craftsmen", status_code=303)

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)

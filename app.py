import re
import json
import pandas as pd
import PIL.Image
import streamlit as st
from google import genai
from google.genai import types
from rapidfuzz import process, fuzz

st.title("Pharma Finder - تجربة قراءة الروشتة")
try:
    app_password = st.secrets["APP_PASSWORD"]
except Exception:
    app_password = None
if app_password:
    if st.text_input("كلمة السر", type="password") != app_password:
        st.stop()

PROMPT = """اقرأ الروشتة دي واستخرج الأدوية المكتوبة. ارجع JSON فقط، قايمة، وكل دواء فيها: name (الاسم التجاري بحروف إنجليزي بس من غير جرعة) و strength_mg (رقم الجرعة بس، أو null لو مش مكتوب). لو الاسم مش واضح اكتب اللي شايفه بالظبط ومتخمنش اسم دواء."""
MODELS = ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-latest"]
THRESHOLD = 70

@st.cache_data
def load_drugs():
    df = pd.read_csv("eg_drugs.csv")
    df["brand"] = df["name"].fillna("").str.lower().apply(lambda x: re.split(r"\d", x)[0].strip())
    return df

df_drugs = load_drugs()
brands = df_drugs["brand"].unique().tolist()

def read_prescription(img, key):
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=60000))
    for m in MODELS:
        try:
            r = client.models.generate_content(model=m, contents=[img, PROMPT], config={"response_mime_type": "application/json"})
            return m, json.loads(r.text)
        except Exception:
            continue
    return None, None

def check_name(name):
    best, score, _ = process.extractOne(name.lower(), brands, scorer=fuzz.ratio)
    return best, score

def available_strengths(brand):
    rows = df_drugs[df_drugs["brand"] == brand]["name"]
    return sorted(set(re.findall(r"(\d+(?:\.\d+)?)\s*mg", " ".join(rows))))

def check_drug(name, strength):
    brand, score = check_name(name)
    if score < THRESHOLD:
        return "غير مسجل", None, []
    options = available_strengths(brand)
    if str(strength) in options:
        return "موجود في قاعدة الأدوية", brand, options
    return "جرعة مختلفة", brand, options

try:
    key = st.secrets["GEMINI_API_KEY"]
except Exception:
    key = st.sidebar.text_input("API key", type="password")
file = st.file_uploader("ارفعي صورة الروشتة", type=["jpg", "jpeg", "png"])

if file and key:
    img = PIL.Image.open(file)
    img.thumbnail((1600, 1600))
    st.image(img, width=400)
    if st.button("اقرأ الروشتة"):
        with st.spinner("بقرا الروشتة..."):
            model, drugs = read_prescription(img, key)
        if drugs is None:
            st.error("كل الموديلات مشغولة دلوقتي، جربي بعد شوية")
        else:
            st.write("اشتغل مع:", model)
            rows = []
            for d in drugs:
                nums = re.findall(r"\d+(?:\.\d+)?", str(d.get("strength_mg") or ""))
                strength = nums[0] if nums else None
                status, brand, options = check_drug(d.get("name", ""), strength)
                if strength is None and status != "غير مسجل":
                    status = "الجرعة غير مقروءة"
                rows.append({"قراءة Gemini": d.get("name", ""), "الاسم المقترح": brand or "-", "الجرعة": strength or "-", "الحالة": status, "الجرعات المتاحة": ", ".join(options) or "-"})
            st.dataframe(pd.DataFrame(rows))
elif file:
    st.info("اكتبي الـ API key في القايمة الجانبية الأول")
    
    st.divider()
st.subheader("جرّبي تصحيح اسم دواء يدويًا")
test_name = st.text_input("اكتبي اسم دواء (حتى لو غلط)")
if test_name:
    best, score = check_name(test_name)
    if score < THRESHOLD:
        st.warning(f"غير مسجل (أقرب اسم: {best}، النسبة {score:.0f}%)")
    else:
        st.success(f"قصدك: {best} (النسبة {score:.0f}%)")
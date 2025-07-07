import streamlit as st
import json
import difflib

st.set_page_config(page_title="Counterfactual Example Selector", layout="wide")

# 전체 앱 배경/글씨색 스타일 적용 (다크 테마)
st.markdown('''
    <style>
    body, .stApp { background: #111 !important; color: #fff !important; }
    .stMarkdown, .stText, .stHeader, .stSubheader, .stTitle, .stCaption, .stDataFrame { color: #fff !important; }
    .cf-card { background: #222; border-radius: 14px; box-shadow: 0 2px 8px #222; padding: 22px 28px 18px 28px; margin-bottom: 28px; }
    .cf-orig { margin-bottom: 16px; }
    .cf-label { font-size:1.1em; color:#fff; font-family:"Noto Sans",Arial,sans-serif; }
    .cf-sent { font-size:1.15em; color:#fff; font-family:"Noto Sans",Arial,sans-serif; }
    .cf-pred { font-size:1em; color:#bbb; font-family:"Noto Sans",Arial,sans-serif; }
    .cf-cf { color:#fff; font-family:"Noto Sans",Arial,sans-serif; }
    </style>
''', unsafe_allow_html=True)

# 레이블 숫자를 텍스트로 변환
LABEL_MAP = {0: 'B1', 1: 'B2'}

# control code별 색상 및 번호 (어두운 배경에 어울리는 색상 유지)
CONTROL_TYPE_STYLE = {
    'lexical':    {'color': '#e57373', 'num': 1},
    'shuffle':    {'color': '#64b5f6', 'num': 2},
    'restructure':{'color': '#81c784', 'num': 3},
    'negation':   {'color': '#ffd54f', 'num': 4},
    'quantifier': {'color': '#ba68c8', 'num': 5},
    'resemantic': {'color': '#ffb74d', 'num': 6},
    'insert':     {'color': '#4db6ac', 'num': 7},
    'delete':     {'color': '#a1887f', 'num': 8},
}

# 두 문장 diff를 HTML로 반환 (control code 색상으로 강조, 어두운 배경에 잘 보이게)
def diff_highlight(a, b, highlight_color):
    a_words = a.split()
    b_words = b.split()
    s = difflib.SequenceMatcher(None, a_words, b_words)
    b_out = []
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'equal':
            b_out.extend(b_words[j1:j2])
        elif tag == 'replace' or tag == 'insert':
            b_out.append(f'<span style="background-color:{highlight_color};color:#111;font-weight:bold; border-radius:4px; padding:1px 3px;">{" ".join(b_words[j1:j2])}</span>')
    return ' '.join(b_out)

# 데이터 로드
with open('outputs/analysis/counterfactual_predictions.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

st.title('Counterfactual Example Selector')

contrastive_pairs = []

for idx, item in enumerate(data):
    # 카드 전체 div
    st.markdown(f"<div class='cf-card'>", unsafe_allow_html=True)
    # original sentence 영역
    st.markdown(f"""
    <div class='cf-orig'>
        <span class='cf-label'><b>Original #{item['original_id']}</b></span><br>
        <span class='cf-label'><b>Original Sentence:</b></span><br>
        <span class='cf-sent'>{item['original_sentence']}</span><br>
        <span class='cf-pred'><b>Original Prediction:</b> {LABEL_MAP[item['original_prediction']]}</span>
    </div>
    <div style='margin-bottom:8px;'><b>Counterfactuals:</b></div>
    """, unsafe_allow_html=True)
    # counterfactual 예시를 2열(왼쪽: 내용, 오른쪽: 체크박스)로 배치
    selected = []
    for cf in item['counterfactuals']:
        ctype = cf['type']
        cstyle = CONTROL_TYPE_STYLE.get(ctype, {'color':'#bbb','num':'?'})
        highlight_color = cstyle['color']
        cf_diff = diff_highlight(item['original_sentence'], cf['sentence'], highlight_color)
        label_flip_color = 'red' if cf['is_label_flipped'] else '#fff'
        type_label = f"<span style='display:inline-block;min-width:80px;background:{highlight_color};color:#111;padding:3px 12px 3px 12px;border-radius:14px;font-weight:bold;font-family:\"Noto Sans\",Arial,sans-serif;font-size:1em;'>#{cstyle['num']} {ctype.capitalize()}</span>"
        left, right = st.columns([8,2])
        with left:
            st.markdown(f"""
            <div class='cf-cf' style='border-radius:8px; margin-bottom:8px; padding:12px 18px;'>
                {type_label}<br>
                <span style='font-size:1.08em; font-family:"Noto Sans",Arial,sans-serif; color:#fff;'><b>Sentence:</b></span> <span style='font-size:1.08em; font-family:"Noto Sans",Arial,sans-serif'>{cf_diff}</span><br>
                <span style='font-size:1em; font-family:"Noto Sans",Arial,sans-serif; color:#bbb;'><b>Prediction:</b> {LABEL_MAP[cf['prediction']]}</span><br>
                <span style='font-size:1em; font-family:"Noto Sans",Arial,sans-serif;'><b>Label Flipped:</b> <span style='color:{label_flip_color}; font-weight:bold'>{cf['is_label_flipped']}</span></span>
            </div>
            """, unsafe_allow_html=True)
        with right:
            checked = st.checkbox("", key=f"{idx}_{cf['type']}" )
            if checked:
                selected.append(cf)
    # 선택된 쌍 저장
    for cf in selected:
        contrastive_pairs.append({
            'original_id': item['original_id'],
            'original_sentence': item['original_sentence'],
            'original_prediction': LABEL_MAP[item['original_prediction']],
            'cf_type': cf['type'],
            'cf_sentence': cf['sentence'],
            'cf_prediction': LABEL_MAP[cf['prediction']],
            'is_label_flipped': cf['is_label_flipped']
        })
    st.markdown("</div>", unsafe_allow_html=True)  # 카드 닫기

# 다운로드 버튼
if contrastive_pairs:
    st.markdown("---")
    st.subheader("Download Selected Contrastive Pairs")
    st.download_button(
        label="Download JSON",
        data=json.dumps(contrastive_pairs, ensure_ascii=False, indent=2),
        file_name="contrastive_pairs.json",
        mime="application/json"
    )
else:
    st.info("Select at least one contrastive pair to enable download.")

import streamlit as st
import json
import difflib

# 레이블 숫자를 텍스트로 변환
LABEL_MAP = {0: 'B1', 1: 'B2'}

# control code별 색상 및 번호
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

# 두 문장 diff를 HTML로 반환 (control code 색상으로 강조)
def diff_highlight(a, b, is_label_flipped, highlight_color):
    a_words = a.split()
    b_words = b.split()
    s = difflib.SequenceMatcher(None, a_words, b_words)
    b_out = []
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'equal':
            b_out.extend(b_words[j1:j2])
        elif tag == 'replace' or tag == 'insert':
            # control code 색상 강조, 글씨는 검정
            b_out.append(f'<span style="background-color:{highlight_color};color:#111;font-weight:bold;">{" ".join(b_words[j1:j2])}</span>')
        # 'delete'는 cf에는 없는 부분이므로 표시 안 함
    return ' '.join(b_out)

# 데이터 로드
with open('outputs/analysis/counterfactual_predictions.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

st.set_page_config(page_title="Counterfactual Example Selector", layout="wide")
st.title('Counterfactual Example Selector')

contrastive_pairs = []

for idx, item in enumerate(data):
    st.header(f"Original #{item['original_id']}")
    st.markdown(f"""
    <div style='background-color:#fff; border-radius:8px; padding:14px 18px 10px 18px; margin-bottom:8px; box-shadow:0 1px 4px #eee;'>
        <span style='font-size:1.1em; color:#222; font-family:"Noto Sans",Arial,sans-serif;'><b>Original Sentence:</b></span><br>
        <span style='font-size:1.15em; color:#222; font-family:"Noto Sans",Arial,sans-serif;'>{item['original_sentence']}</span><br>
        <span style='font-size:1em; color:#555; font-family:"Noto Sans",Arial,sans-serif;'><b>Original Prediction:</b> {LABEL_MAP[item['original_prediction']]}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Counterfactuals:**")
    selected = []
    for cf in item['counterfactuals']:
        # control code 스타일
        ctype = cf['type']
        cstyle = CONTROL_TYPE_STYLE.get(ctype, {'color':'#bbb','num':'?'})
        highlight_color = cstyle['color']
        # diff 하이라이트 (control code 색상)
        cf_diff = diff_highlight(item['original_sentence'], cf['sentence'], cf['is_label_flipped'], highlight_color)
        # Label Flipped 색상
        label_flip_color = 'red' if cf['is_label_flipped'] else '#222'
        # control code 라벨
        type_label = f"<span style='display:inline-block;min-width:80px;background:{highlight_color};color:#fff;padding:3px 12px 3px 12px;border-radius:14px;font-weight:bold;font-family:\"Noto Sans\",Arial,sans-serif;font-size:1em;'>#{cstyle['num']} {ctype.capitalize()}</span>"
        # 한 줄에 표시 (배경 없음)
        st.markdown(f"""
        <div style='border-radius:8px; margin-bottom:12px; padding:12px 18px;'>
            {type_label}<br>
            <span style='font-size:1.08em; font-family:"Noto Sans",Arial,sans-serif; color:#222;'><b>Sentence:</b></span> <span style='font-size:1.08em; font-family:"Noto Sans",Arial,sans-serif'>{cf_diff}</span><br>
            <span style='font-size:1em; font-family:"Noto Sans",Arial,sans-serif; color:#555;'><b>Prediction:</b> {LABEL_MAP[cf['prediction']]}</span><br>
            <span style='font-size:1em; font-family:"Noto Sans",Arial,sans-serif;'><b>Label Flipped:</b> <span style='color:{label_flip_color}; font-weight:bold'>{cf['is_label_flipped']}</span></span>
            <br>
        </div>
        """, unsafe_allow_html=True)
        # streamlit 체크박스(별도)
        checked = st.checkbox("Contrastive Pair", key=f"{idx}_{cf['type']}")
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

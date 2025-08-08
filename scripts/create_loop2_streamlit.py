import json
import os

def create_loop2_streamlit():
    """Create a Streamlit app for loop_2_data.json"""
    
    # Load loop_2_data.json
    with open('data/Loop/loop_2/loop_2_data.json', 'r', encoding='utf-8') as f:
        loop2_data = json.load(f)
    
    print(f"Loaded {len(loop2_data)} items from loop_2_data.json")
    
    # Create the Streamlit app code
    streamlit_code = r'''import streamlit as st
import json
import difflib
import random
import datetime

st.set_page_config(page_title="Loop 2 Counterfactual Example Selector", layout="wide")

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

# 1. 데이터 로드
# loop_2_data.json 로드
with open('data/Loop/loop_2/loop_2_data.json', 'r', encoding='utf-8') as f:
    loop2_data = json.load(f)

# 5개 버전으로 분할 (각 버전: 40개씩)
version = st.sidebar.selectbox("Select survey version", [1, 2, 3, 4, 5])

# 각 버전별로 다른 문장 40개 선택
random.seed(42)
random.shuffle(loop2_data)

start_idx = (version - 1) * 40
end_idx = version * 40
survey_data = loop2_data[start_idx:end_idx]

print(f"Version {version}: {len(survey_data)}개 문장")

st.title('Loop 2 Counterfactual Example Selector')

# 버전 확인 안내
st.markdown('''
<span style="color:#d32f2f; font-size: 1.15em; font-weight:bold;">Please make sure to check that the survey version shown in the top left corner is your assigned version before you begin.</span>
''', unsafe_allow_html=True)

# 안내 서문 추가
st.markdown('''
### Part 1: Sentence Selection
Among the examples generated using each control code, please select the sentences that feel different in difficulty compared to the original sentence.<br/>
For each set (i.e., one original sentence and its variations), you must select at least one, and you may select more than one if needed.<br/><br/>
For example, if the original sentence seems to be at the B1 level, please select the ones that feel more like B2 level.<br/>
The AI model's predictions are provided for reference only.
''', unsafe_allow_html=True)

contrastive_pairs = []

for idx, item in enumerate(survey_data):
    # 카드 전체 div
    st.markdown(f"<div class='cf-card'>", unsafe_allow_html=True)
    # original sentence 영역
    st.markdown(f"""
    <div class='cf-orig'>
        <span class='cf-label'><b>Sentence #{idx+1}</b></span>
        <span style='font-size:0.9em; color:#aaa; margin-left:10px;'>(original id: {item['original_id']})</span><br>
        <span class='cf-label'><b>Original Sentence:</b></span><br>
        <span class='cf-sent'>{item['original_sentence']}</span><br>
        <span class='cf-pred'><b>Original Prediction:</b> {LABEL_MAP[item['original_prediction']]}</span>
    </div>
    <div style='margin-bottom:8px;'><b>Counterfactuals:</b></div>
    """, unsafe_allow_html=True)
    
    # counterfactual 예시를 2열(왼쪽: 내용, 오른쪽: 체크박스)로 배치
    selected = []
    
    # loop_2_data.json에는 counterfactuals가 없으므로, 원본 문장만 표시
    st.markdown(f"""
    <div class='cf-cf' style='border-radius:8px; margin-bottom:8px; padding:12px 18px;'>
        <span style='display:inline-block;min-width:80px;background:#666;color:#111;padding:3px 12px 3px 12px;border-radius:14px;font-weight:bold;font-family:"Noto Sans",Arial,sans-serif;font-size:1em;'>Original</span><br>
        <span style='font-size:1.08em; font-family:"Noto Sans",Arial,sans-serif; color:#fff;'><b>Sentence:</b></span> <span style='font-size:1.08em; font-family:"Noto Sans",Arial,sans-serif'>{item['original_sentence']}</span><br>
        <span style='font-size:1em; font-family:"Noto Sans",Arial,sans-serif; color:#bbb;'><b>Prediction:</b> {LABEL_MAP[item['original_prediction']]}</span><br>
        <span style='font-size:1em; font-family:"Noto Sans",Arial,sans-serif;'><b>Label Flipped:</b> <span style='color:#fff; font-weight:bold'>False</span></span>
    </div>
    """, unsafe_allow_html=True)
    
    # 체크박스 (선택사항)
    left, right = st.columns([8,2])
    with left:
        st.markdown("<!-- Placeholder for counterfactuals -->", unsafe_allow_html=True)
    with right:
        checked = st.checkbox("", key=f"{idx}_original")
        if checked:
            selected.append({
                'type': 'original',
                'sentence': item['original_sentence'],
                'prediction': item['original_prediction'],
                'is_label_flipped': False
            })
    
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

# Part 2: Control Code Feedback
st.markdown('---')
st.header('Part 2: Control Code Feedback')
st.markdown('''
Each item below represents a control code used by the AI to generate example sentences.<br/>
Please evaluate how each code influences your judgment of text difficulty.
''', unsafe_allow_html=True)

for code, info in CONTROL_TYPE_STYLE.items():
    with st.expander(f"{info['num']}. {code.capitalize()}"):
        st.markdown(f"**Example: {code.capitalize()}**")
        st.markdown('''
        **2-1. Did this control code affect your perception of text difficulty?**  
        (Please rate how much it contributed to distinguishing difficulty levels.)
        ''')
        st.slider(f"2-1_{code}", min_value=1, max_value=5, value=3, format="%d", key=f"2-1_{code}")

        st.markdown('''
        **2-2. Do you think this control code is necessary for future evaluations of text difficulty?**  
        1 (Not necessary) ~ 5 (Highly necessary)
        ''')
        st.slider(f"2-2_{code}", min_value=1, max_value=5, value=3, format="%d", key=f"2-2_{code}")

        st.markdown('''
        **2-3. (Important) How could this control code be modified or refined to make it more effective?**  
        (e.g., Restructure → change to passive voice / use of embedded clauses, etc.)
        ''')
        st.text_area(f"2-3_{code}", placeholder="Please write your suggestions here.", key=f"text_2-3_{code}")

# Part 3: AI Collaboration Feedback
st.markdown('---')
st.header('Part 3: AI Collaboration Feedback')
st.markdown('''
**3-1. Compared to human-written sentences, how would you rate the quality of the AI-generated sentences as examples for assessing text difficulty?**  
*(Please consider factors such as clarity of difficulty difference, fluency, and appropriateness of expression.)*  
1 (Very poor quality) ~ 5 (Very high quality)
''')
st.slider("3-1_quality", min_value=1, max_value=5, value=3, format="%d", key="3-1_quality")

st.markdown('''
**3-2. Did the example sentence pairs help you meaningfully identify difficulty boundaries?**  
1 (Not helpful at all) ~ 5 (Very helpful)  
*(Please judge based on whether they helped you compare ambiguous cases or clarify your criteria.)*
''')
st.slider("3-2_helpful", min_value=1, max_value=5, value=3, format="%d", key="3-2_helpful")

st.markdown('''
**3-3. (Important) In what ways did the AI-generated sentences help you make better difficulty judgments than if you had made them without any examples?**  
*(For example, mention specific expressions or contrasts that helped clarify your judgment.)*
''')
st.text_area("3-3_benefit", placeholder="Please write your answer here.", key="text_3-3_benefit")

# --- 저장 버튼 및 다운로드 기능 (앱 최하단) ---
st.markdown('---')
st.header('Save All Your Responses')
st.markdown('''
To save your answers for all parts of the survey (Part 1, Part 2, Part 3), please click the button below.\
All your selections and inputs will be saved in a single JSON file.
''')
st.markdown('''
<span style="color:#d32f2f; font-size: 1.25em; font-weight:bold;">After saving, you MUST send the downloaded file to jiwooryu45@ajou.ac.kr.<br>If you do not send the file, your responses will NOT be collected.</span>
''', unsafe_allow_html=True)

# 자동으로 모든 답변 수집
responses = {}
# Part 1: 선택된 contrastive pairs
responses['part1_selected_sentences'] = contrastive_pairs
# Part 2: control code feedback
part2 = {}
for code in CONTROL_TYPE_STYLE.keys():
    part2[code] = {
        '2-1': st.session_state.get(f'2-1_{code}'),
        '2-2': st.session_state.get(f'2-2_{code}'),
        '2-3': st.session_state.get(f'text_2-3_{code}')
    }
responses['part2_control_code_feedback'] = part2
# Part 3: AI collaboration feedback
responses['part3_ai_collaboration_feedback'] = {
    '3-1': st.session_state.get('3-1_quality'),
    '3-2': st.session_state.get('3-2_helpful'),
    '3-3': st.session_state.get('text_3-3_benefit')
}

ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'loop2_expert_response_{ts}.json'
st.download_button(
    label='Download All Responses',
    data=json.dumps(responses, ensure_ascii=False, indent=2),
    file_name=filename,
    mime='application/json'
)
'''
    
    # Write the Streamlit app to a file
    output_path = 'app/streamlit_app_loop2.py'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(streamlit_code)
    
    print(f"Streamlit app created: {output_path}")
    print(f"App uses {len(loop2_data)} items from loop_2_data.json")
    print(f"Each version will show 40 items")

if __name__ == "__main__":
    create_loop2_streamlit() 
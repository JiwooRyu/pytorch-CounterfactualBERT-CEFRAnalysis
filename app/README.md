# Counterfactual Example Selector

This repository contains Streamlit applications for expert evaluation of counterfactual examples.

## Apps

### 1. streamlit_app.py (Loop 1)
- **Purpose**: Expert evaluation of counterfactual examples with full data
- **Data**: Uses anchor_data.json and loop_1_data.json
- **Features**: 
  - Part 1: Sentence selection with counterfactuals
  - Part 2: Control code feedback
  - Part 3: AI collaboration feedback
  - Part 4: Final confirmation

### 2. streamlit_app_loop2.py (Loop 2)
- **Purpose**: Expert evaluation of Loop 2 data
- **Data**: Uses loop_2_data.json
- **Features**: Same structure as Loop 1 app

## File Structure

```
app/
├── streamlit_app.py              # Loop 1 app
├── streamlit_app_loop2.py        # Loop 2 app
├── requirements.txt              # Python dependencies
├── README.md                    # This file
├── Loop/                        # Data files
│   ├── anchor_data.json
│   ├── loop_1_data.json
│   ├── loop_2/
│   │   └── loop_2_data.json
│   └── triplets_expert_all.json
├── processed/                    # Classification data
│   ├── train.csv
│   ├── test.csv
│   └── base.csv
└── models/                      # BERT models
    └── baseline_bert/
```

## Deployment

### Streamlit Cloud
1. Connect this repository to Streamlit Cloud
2. Set the main file path to `app/streamlit_app.py` for Loop 1
3. Or set to `app/streamlit_app_loop2.py` for Loop 2

### Local Development
```bash
cd app
streamlit run streamlit_app.py --server.port 8501
streamlit run streamlit_app_loop2.py --server.port 8502
```

## Data Sources

- **Loop 1**: Full counterfactual examples with anchor data
- **Loop 2**: Simplified data with original sentences only
- **Models**: Pre-trained BERT models for prediction

## Features

- Dark theme UI
- Version-based data splitting
- JSON response download
- Expert evaluation interface
- Accessibility warnings suppressed 
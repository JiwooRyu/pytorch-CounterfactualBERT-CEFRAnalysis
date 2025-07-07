import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def preprocess_cefr_data():
    """
    CEFR-SP 데이터를 전처리하여 train/valid/test 세트로 분할
    """
    # 데이터 로드
    print("데이터 로드 중...")
    df = pd.read_csv('data/raw/CEFR-SP_All_3or4.csv')
    
    print(f"원본 데이터 형태: {df.shape}")
    print(f"컬럼: {df.columns.tolist()}")
    print(f"레이블 분포:\n{df['label'].value_counts()}")
    
    # 1. 결측치 처리
    print("\n결측치 확인...")
    print(f"결측치 개수:\n{df.isnull().sum()}")
    
    # 결측치가 있는 행 제거
    df_clean = df.dropna()
    print(f"결측치 제거 후 데이터 형태: {df_clean.shape}")
    
    # 2. 레이블 변경 (3->0, 4->1)
    print("\n레이블 변경 중...")
    df_clean['label'] = df_clean['label'].map({3: 0, 4: 1})
    print(f"변경된 레이블 분포:\n{df_clean['label'].value_counts()}")
    
    # 3. 데이터 분할
    print("\n데이터 분할 중...")
    
    # 먼저 train과 temp 분할 (train: 2000개, 나머지: 828개)
    train_df, temp_df = train_test_split(
        df_clean, 
        train_size=2000, 
        test_size=828, 
        random_state=42, 
        stratify=df_clean['label']
    )
    
    # temp를 valid와 test로 분할 (valid: 300개, test: 528개)
    valid_df, test_df = train_test_split(
        temp_df,
        train_size=300,
        test_size=528,
        random_state=42,
        stratify=temp_df['label']
    )
    
    print(f"Train 세트: {train_df.shape[0]}개")
    print(f"Valid 세트: {valid_df.shape[0]}개")
    print(f"Test 세트: {test_df.shape[0]}개")
    
    # 각 세트의 레이블 분포 확인
    print(f"\nTrain 레이블 분포:\n{train_df['label'].value_counts()}")
    print(f"Valid 레이블 분포:\n{valid_df['label'].value_counts()}")
    print(f"Test 레이블 분포:\n{test_df['label'].value_counts()}")
    
    # 4. processed 디렉토리 생성 및 저장
    print("\n파일 저장 중...")
    os.makedirs('data/processed', exist_ok=True)
    
    train_df.to_csv('data/processed/train.csv', index=False)
    valid_df.to_csv('data/processed/valid.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    print("전처리 완료!")
    print("저장된 파일:")
    print("- data/processed/train.csv")
    print("- data/processed/valid.csv")
    print("- data/processed/test.csv")

if __name__ == "__main__":
    preprocess_cefr_data()
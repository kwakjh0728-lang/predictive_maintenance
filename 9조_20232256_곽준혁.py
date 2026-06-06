import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt

# 1. 웹페이지 기본 설정 및 디자인
st.set_page_config(page_title="예지 정비 시스템", page_icon="⚙️", layout="wide")

st.title("⚙️ 소리 주파수 분석 기반 예지 정비 시스템")
st.markdown("### 기계·자동차·항공 분야 비파괴 음향 검사 대시보드")
st.write("설비 운전 중 발생하는 소리 데이터를 분석하여 정상 작동 여부를 실시간으로 진단합니다.")
st.markdown("---")

# 레이아웃 나누기 (왼쪽: 파일 업로드 및 결과, 오른쪽: 주파수 그래프)
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📁 데이터 업로드 및 진단")
    # 파일 업로더 컴포넌트
    uploaded_file = st.file_uploader("기계 측정 소리 파일 (.wav, .mp3)을 선택하세요.", type=["wav", "mp3"])

if uploaded_file is not None:
    # 2. 오디오 분석 로직 (Librosa 활용)
    with st.spinner("주파수 성분을 분석하는 중입니다..."):
        # 임시로 파일 로드 (샘플링 레이트 22050Hz)
        y, sr = librosa.load(uploaded_file, sr=22050)
        
        # 고속 푸리에 변환 (FFT) 수행
        fft_result = np.fft.fft(y)
        magnitude = np.abs(fft_result)
        frequency = np.linspace(0, sr, len(magnitude))
        
        # 대칭성 때문에 절반만 사용
        half_len = len(frequency) // 2
        frequency = frequency[:half_len]
        magnitude = magnitude[:half_len]
        
        # 특정 고주파 대역 (4000Hz ~ 8000Hz) 에너지 계산 (고장 진단 기준)
        high_freq_indices = np.where((frequency >= 4000) & (frequency <= 8000))
        avg_high_freq_energy = float(np.mean(magnitude[high_freq_indices]))
        
        # 고장 판단 임계값 설정
        THRESHOLD = 5.0
        is_abnormal = avg_high_freq_energy > THRESHOLD

    # 3. 왼쪽 컬럼에 분석 결과 출력
    with col1:
        st.success("분석 완료!")
        
        # 메트릭 카드로 수치 보여주기
        st.metric(label="고주파 대역 평균 에너지", value=f"{avg_high_freq_energy:.2f}")
        
        # 상태에 따른 경고창 출력
        if is_abnormal:
            st.error("🚨 **최종 판정: 고장 의심 (Abnormal)**")
            st.markdown("> **진단 근거:** 내부 베어링 마모 또는 마찰로 인한 이상 고주파 성분이 감지되었습니다. 즉각적인 점검이 필요합니다.")
        else:
            st.success("✅ **최종 판정: 정상 작동 중 (Normal)**")
            st.markdown("> **진단 근거:** 주파수 분포가 안정적이며, 설비 특이 진동음이 발견되지 않았습니다.")
            
        # 오디오 플레이어 추가 (웹에서 직접 들어볼 수 있음)
        st.audio(uploaded_file, format="audio/wav")

    # 4. 오른쪽 컬럼에 주파수 스펙트럼 시각화
    with col2:
        st.subheader("📊 주파수 스펙트럼 (FFT 변환 결과)")
        
        fig, ax = plt.subplots(figsize=(10, 4.5))
        if is_abnormal:
            ax.plot(frequency, magnitude, color='#d9534f', label="Abnormal Spectrum")
        else:
            ax.plot(frequency, magnitude, color='#5cb85c', label="Normal Spectrum")
            
        ax.set_title("Frequency Domain Analysis", fontsize=12)
        ax.set_xlabel("Frequency (Hz)", fontsize=10)
        ax.set_ylabel("Magnitude", fontsize=10)
        ax.set_xlim(0, 11000)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        
        # Streamlit 화면에 matplotlib 그래프 띄우기
        st.pyplot(fig)
else:
    with col2:
        st.info("왼쪽에서 음향 파일을 업로드하면 이곳에 주파수 시각화 그래프가 표시됩니다.")
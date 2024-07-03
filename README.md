# 트랜스포머 기반 2-Phase 학습법을 통한 EEG 데이터 분류 모델 성능 개선
> '24W Deep Daiv Deep Learning Architecture Project

## Overview
베이스라인 [EEG2IMAGE](https://github.com/prajwalsingh/EEG2Image)에서 분류 모델 성능을 개선시키고자 한 프로젝트 입니다.
기존 연구에서의 분류 모델은 **EEG 신호의 공간적 정보를 포착하지 못해** 낮은 성능을 보였고 t-SNE 시각화 결과에 의하면 **각 10가지 클래스 별 군집이 명확하지 않다**는 한계점이 있었습니다.

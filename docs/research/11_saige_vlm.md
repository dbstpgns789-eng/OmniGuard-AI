# 세이지(SAIGE SAFETY)의 VLM 재판단 구조 확인 (2026-10-20, Aside)

질문: "1차 탐지 → VLM이 맥락으로 재판단해 오탐을 거른다"는 구조를 세이지가 실제로 쓰는가, 그리고 VLM이 설비 정지 판단에 들어가는가.
saige.ai는 직접 접속 시 403이라 검색 엔진에 색인된 본문으로 확인했다.

## 결론
- **구조는 맞다 [원문].** 제품 페이지: "1차 탐지 결과를 상황·맥락으로 재판단해, 불필요한 알람을 관제실 도달 전에 차단합니다." 구성도: "기존 CCTV → 1차 AI 탐지 → VLM 검증 레이어 → 정확한 알림". 블로그(2026-06-09): "CNN 1차 탐지 → VLM 2차 맥락 검증", 기존 탐지 위에 얹는 하이브리드 구조
- **VLM은 알림·오탐 필터로만 서술된다.** 라인업 표: Standard = Lite + VLM(오탐 필터링), Plus = Standard + Depth AI(3차원 침입·**협착**). 협착은 Depth AI 항목. VLM은 "AI 제안 + 사람 확인" 원칙, 판정마다 자연어 근거 제시
- **PLC 자동 정지는 VLM보다 먼저 있던 기능이다.** 정지 문구(2025-08-19 블로그, "지능형 공간인식 기술 … PLC 연동으로 장비 자동 정지")가 VLM 탑재 발표(2026-07)보다 앞섬. 정지 신호가 VLM을 거치는지는 **확인 못 함**
- **"화면상 겹침" 문구의 원출처**: 블로그 2026-03-17 「제조 현장에 맞는 지능형 영상분석」. 표 "협착 판단 기준 | 화면상 겹침 여부 | 실제 3D 위험 거리 수치", "오탐 발생 | 지게차 통과 시 반복 오탐". 해결책은 VLM이 아니라 일반 CCTV에서 뽑은 Depth Map
- 언론(뉴스와이어·이데일리·에이빙·에너지안전신문, 2026-07)은 같은 보도자료를 옮긴 것. 전자신문·지디넷 기사는 VLM 도입 전(2025)

## 수치 (모두 세이지 자체 주장)
- "위험 구역 침입·협착 1초 이내 감지알림": Depth/3D 기능의 수치, VLM 시간 아님. 문서마다 1초 / 1~2초 / 3초로 다름
- VLM 효과: 화재 오탐 알림 차단률 95.9%(자체 실측), 선별관제로 오탐 최대 90% 감소. 협착에 대한 VLM 효과 수치와 VLM 추론 시간은 공개 안 됨

## 출처
- https://saige.ai/solution/saige-safety/
- https://saige.ai/blog/multimodal-ai-industrial-safety/ (2026-06-09)
- https://saige.ai/blog/selective-monitoring-vlm-guide/ (2026-06-11)
- https://saige.ai/blog/intelligent-video-analytics-manufacturing/ (2026-03-17)
- https://www.newswire.co.kr/newsRead.php?no=1037699 (2026-07-02)
- https://www.edaily.co.kr/News/Read?mediaCodeNo=257&newsId=04086886645510584 (2026-07-02)

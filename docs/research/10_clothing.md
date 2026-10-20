# 10. 옷·소매·머리카락·장갑 말림 위험 감지 (Aside, 2026-10-19)

공백: **회전 기계 옆에서 소매나 헐렁한 옷을 영상으로 검출한 논문은 찾지 못했다.** 가까운 것은 사람 마스크 vs 위험구역, 인체 부위 분할(human parsing: 옷·머리·장갑을 픽셀 단위로), PPE·위생 검출, 가림에 대한 견고성. 수치는 대부분 공개 벤치마크(COCO, LIP, PA-100K) 기준이지 공장 CCTV가 아니다.

배경: OSHA는 천천히 도는 매끈한 축도 옷을 잡아챌 수 있고 보호장갑 자체가 말려 들어갈 수 있다고 명시. 국내 안전교육 자료는 "회전 말림점"을 장갑·작업복이 말려 들어가는 위험점으로 정의, 끼임 사망 원인으로 회전체 작업 시 면장갑 착용을 꼽음.

## 출처
1. **SCHP (TPAMI)** https://github.com/GoGoDuck912/Self-Correction-Human-Parsing — ResNet-101 인체 부위 분할. LIP mIoU 59.36%, 20개 라벨에 Hair·Glove·Coat·Scarf·Upper-clothes·좌우 팔. ATR 82.29%. MobileNet-V2 백본 53.13 → LIP 모델 하나로 머리·장갑·코트·스카프·팔 픽셀을 한 번에
2. **SCHP ONNX/INT8 포팅** https://github.com/pirocheto/schp-human-parsing — 16코어 CPU(8스레드), 473×473: PyTorch 360ms, ONNX 243ms, **INT8 189ms**, 66MB → 노트북에서 사람 crop 기준 3~5Hz, 조건부 실행용
3. **MediaPipe Multiclass Selfie Segmentation (2023)** https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter — 배경·머리·몸 피부·얼굴 피부·옷·기타. mIoU 77.23(256)/81.10(512). **모델카드: 생명이 걸린 판단에 부적합, 큰 가림물이 있으면 품질 하락.** Pixel 6 CPU 218ms → "옷" 픽셀과 "맨살" 분리, 셀카 학습이라 CCTV 원거리 성능 확인 못 함
4. **PaddleClas PULC 사람 속성 (PA-100K)** — 사람 crop에서 26개 속성, LongSleeve/ShortSleeve/Hat. mA 78.59%, **2.01ms**(Xeon) → 트랙마다 한 번만 "긴소매 여부" 캐시, 위험 여유 확대용
5. **YOLOv8n-seg / YOLO26n-seg** https://docs.ultralytics.com/tasks/segment/ — v8n-seg mask mAP 30.5, CPU ONNX 96.1ms. 26n-seg 33.9, 53.3ms → 위험구역 주변 ROI만 넣으면 실시간에 가까움(추정). 사람 마스크는 옷 외곽을 포함하므로 키포인트가 못 보는 소매 끝 일부를 잡을 수 있음(추정)
6. **Ultralytics MobileSAM 비교표** https://docs.ultralytics.com/models/mobile-sam — Apple M4 CPU: MobileSAM 23,802ms, SAM2-t 23,430ms, FastSAM-s 58ms, **YOLOv8n-seg 24.8ms, YOLO11n-seg 24.3ms** (SAM은 "전체 분할" 모드로 보임) → CPU 상시는 YOLO-seg, SAM은 오프라인 검증용
7. **MobileSAM 2023** https://arxiv.org/abs/2306.14289 — GPU 약 10ms, CPU 수치 없음 → 키포인트·박스를 프롬프트로 소매 마스크 정밀화(저빈도)
8. **EfficientSAM (CVPR 2024)** https://github.com/yformer/EfficientSAM — MobileSAM보다 COCO AP +3.6, ONNX 공개, CPU 속도 없음
9. **Pose2Seg + OCHuman (CVPR 2019)** https://arxiv.org/abs/1803.10683 — 심한 가림에서 박스보다 포즈가 인스턴스 구분에 유리 → 키포인트와 마스크는 보완 관계(사람끼리 가림 근거, 기계 가림은 아님)
10. **SAM-pose2seg 2026** https://arxiv.org/html/2601.08982v2 — 예측 키포인트를 SAM 프롬프트로, OCHuman AP 34.7(SAM 2.1 29.5). 키포인트 대부분이 없어도 전체 마스크 복원 → RTMPose 키포인트를 프롬프트로 넣는 설계 근거
11. **미국 특허 US12705898B2 (2026)** — 열화상 + 인스턴스 분할 작업자 마스크, **마스크 면적의 10% 이상이 위험영역과 겹치면 진입**, 기계 정지까지 → 겹침 비율 정지 조건의 선례
12. **제철 현장 PPE + 위험구역 Dual-Model (2026)** https://link.springer.com/article/10.1007/s40192-026-00461-6 — YOLOv8 두 개, 작업자 박스와 다각형 교차 규칙, 위반 프레임 정확도 92% → 박스를 마스크로 바꿨을 때 비교 기준선
13. **위생 준수 경량 YOLO (Sensors 2025)** https://www.mdpi.com/1424-8220/25/19/6140 — 공개 31,371장, 장갑/맨손/헤어넷/없음 등 7클래스. YOLOv8n mAP@50 85.6 → "장갑 착용", "머리 미고정" 신호의 출발점(짧은 미세조정 필요)
14. **개선 YOLOv8 장갑 검출 (Sensors 2023)** https://www.mdpi.com/1424-8220/23/24/9906 — 유압 부품 공장 선반·밀링·드릴 2,695장, 맨손·흰·면·검은 장갑, mAP50 0.952 → 0.976 → 회전 공작기계 앞 면장갑 검출이 실제 공장에서 가능
- 보조: **YOLOE (2025)** https://docs.ultralytics.com/models/yoloe — 텍스트 프롬프트로 임의 클래스 검출·분할, "rag/cloth" 정확도 확인 못 함. **걸레 검출 전용 연구는 없음** → 경고 단계에만

## 가림
- 확인: 사람끼리 심한 가림에서 포즈가 박스보다 유리, 키포인트 프롬프트로 일부 관절이 없어도 마스크 복원(9, 10)
- 확인 못 함: 기계에 가렸을 때 마스크가 키포인트보다 견고하다는 직접 비교. 오히려 MediaPipe 모델카드는 큰 가림물에서 품질 하락
- 원리상 추론: top-down 포즈는 가려진 관절도 "추측"해서 출력, 마스크는 보이는 픽셀만 → 롤에 실제로 가까운 천 조각을 직접 보여 줌. 대체가 아니라 보완

## 이번 주에 추가할 것 (Aside 제안, CPU, 사전학습만)
1. **사람 마스크 채널(1순위):** YOLOv8n-seg/YOLO11n-seg person 마스크를 위험구역 주변 ROI에만. 위험구역 거리맵을 `cv2.distanceTransform`으로 한 번 만들어 두고 매 프레임 "마스크 픽셀 중 최소 거리"와 겹침 비율. 포즈 상태가 경고 이상일 때만 실행
2. **옷·머리 채널(조건부, 2~5Hz):** MediaPipe multiclass 또는 SCHP-LIP INT8. 핵심 지표 = **"옷 픽셀이 손목·팔꿈치 키포인트보다 롤에 얼마나 더 가까운가"(d_kp − d_cloth)**, 크면 소매가 몸보다 앞섬. 손목 주변 옷 폭이 팔 두께보다 훨씬 넓으면 "헐렁한 소매"(검증 안 됨)
3. **속성 채널(트랙당 1회 캐시):** PULC LongSleeve·Hat, LIP Glove·Hair 면적 → 정지 트리거가 아니라 **여유 확대 계수**로만
4. **상태 기계 결합(헛정지를 늘리지 않는 규칙):**
   - 옷·속성 신호는 **최대 한 단계만 올림**(정상→경고, 경고→감속)
   - 몸 마스크 팽창("헐렁한 옷 여유")은 사람 크기 비례(어깨폭 × 0.1~0.2), 속성 플래그가 켜진 사람만 크게
   - **정지**는 (a) 기존 키포인트 정지 조건, 또는 (b) 팽창하지 않은 원래 마스크가 정지 구역과 N/M 프레임 이상 겹치고 같은 사람의 손목·팔꿈치가 감속 구역 안이거나 키포인트 신뢰도가 낮을 때(가림). 겹침 비율은 특허의 10%를 참고값으로
   - 사람 없는 빈 장면에서 "person"으로 오분류되는 기계 부위 픽셀(롤 반사, 필름)을 미리 모아 무시 마스크로
5. **평가:** 필름·봉투기 사고 영상에서 새 채널이 정지를 몇 프레임 먼저 내는지, 정상 작업 영상에서 시간당 헛정지·헛감속 증가량

## Claude 메모
- 봉투기 영상의 세훈 판단 이유("헐렁한 옷이 걸릴 듯한 거리")에 직접 대응하는 지표가 "d_kp − d_cloth"다. YOLO11n-seg는 CPU 24ms라 기존 파이프라인에 붙일 수 있다.
- "옷 신호는 최대 한 단계만 올림"은 앞선 조사들의 결론(헛정지를 감속으로 흡수)과 일관된다.
- 빈 장면 무시 마스크는 실험 05·07의 기둥 오검출에도 쓸 수 있다(사람이 없을 때 "사람"으로 잡히는 위치를 모아 두기).

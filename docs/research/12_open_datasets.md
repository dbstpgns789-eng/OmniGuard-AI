# GitHub·Kaggle·Hugging Face 공개 데이터 조사 (2026-10-20)

계기: 채우 제안 "데이터 출처를 적어야 하니 되도록 GitHub, Kaggle, Hugging Face 데이터를 쓰자". 앞선 조사 `05_datasets.md`를 이 세 곳 기준으로 다시 확인했다(HF API, Kaggle 공개 API, `gh search`). 파일은 받지 않았다.

## 결론
- 세 곳에 **회전체 끼임 사고를 라벨까지 붙여 공개한 영상 데이터는 iSafetyBench 하나**뿐이다. 끼임·압착 클립 32개이고, 원본은 YouTube 영상 6개를 자른 것이다
- **iSafetyBench는 이미 쓰고 있다**: 실험 04(빨간 상의 클립 3개), 실험 10(넓은 화면 클립 4개)
- 헛정지 측정용 정상 작업 영상은 **Safe and Unsafe Behaviours(HF, CC BY 4.0, 터키 공장 고정 CCTV 691클립)**가 가장 깔끔하다

## 표
| 이름 | 위치 | 내용 | 라벨 | 시점 | 라이선스 | 쓸모 |
|---|---|---|---|---|---|---|
| iSafetyBench | HF raiyaanabdullah/isafety-bench, GitHub iSafetyBench/data | 4~8초 클립 1,100개, 끼임·압착 32개(원본 YouTube `0QHQ2nmdE78` `_4GgKoR0OOw` `9XXineiOxSo` `7QFdUqCdML8` `Zag5F6FXlj0` `_a46_s5WViY`) | 클립 행동 라벨 + 캡션, 시각 없음 | CCTV 모음 | CC BY-NC-SA 4.0 | 사고 시험용 (이미 사용) |
| Safe and Unsafe Behaviours | HF Voxel51/Safe_and_Unsafe_Behaviours | 터키 공장 691클립 1080p, 설비 개입·패널·통로·지게차 8클래스 | 클립 클래스 | 고정 IP카메라 2대 | CC BY 4.0 | 헛정지 측정용 |
| OpenMarcie | Kaggle hymalaibello/openmarciescenariobpart1 외 | 제조 작업, 외부 RGB·깊이 카메라 + 웨어러블, 파트당 130~170GB | 행동, 마스크 | 고정 외부 카메라 | CC BY-SA 4.0 | 헛정지용(일부만) |
| InHARD | GitHub vhavard/InHARD (본체 Zenodo) | 사람-로봇 조립 3카메라, 3D 스켈레톤 | 행동 | 고정 | MIT | 참고 |
| CHICO | GitHub AlessioSam/CHICO-PoseForecasting | 작업자-협동로봇 충돌 226건, 멀티뷰 3D 포즈 | 3D 포즈·충돌 | 고정 멀티뷰 | 표기 없음 | 참고(2D 겹침 vs 3D 거리) |
| LAMIS (IAR-Net) | GitHub navalkishoremehta95/IAR-NET | 선반 작업 행동(신청 후 배포) | 행동 | 확인 못 함 | 확인 못 함 | 헛정지용(신청 필요) |
| IPAD | GitHub LJF1113/IPAD | 산업 장비 이상 영상, 사람 거의 없음 | 이상 | 고정 | 확인 못 함 | 참고 |
| 1인칭 공장 영상(Verbose Tech Labs, Egocentric-10K 등) | Kaggle, HF | 머리 장착 카메라 | 일부 | 1인칭 | 여러 가지 | 못 씀(CCTV 아님) |
| 낙상(Qlean, Simuletic), PPE(SH17), 건설장비 | HF, Kaggle | 낙상·안전모·장비 | 상자·관절 | 혼합 | 여러 가지 | 못 씀 또는 참고 |
| aihub-factory-abnormal | Kaggle sbrighth | CSV 20개(센서 추정) | 확인 못 함 | 영상 아님 | Unknown | 못 씀. AI Hub 078 재업로드는 세 곳에 없음 |

## 출처를 적는 방법
1. 공개 데이터셋(iSafetyBench, Safe and Unsafe Behaviours)은 이름, 링크, 라이선스를 적는다
2. 수집 영상은 파일을 올리지 않고 목록(`docs/videos/manifest.csv`: 플랫폼, URL, 사고 시각, 설비)만 GitHub에 공개한다. iSafetyBench도 YouTube 클립을 같은 방식으로 공개했다
3. 직접 찍은 러닝머신 시연 영상은 자체 촬영으로 적는다

출처: arXiv 2508.00399 (iSafetyBench), docs.voxel51.com Safe and Unsafe Behaviours

## 확인 결과 (같은 날 추가)
### iSafetyBench 끼임 클립과 우리 수집 목록 비교
| 원본 YouTube | 클립 | 내용 | 우리 목록 |
|---|---|---|---|
| 0QHQ2nmdE78 | 3 | 롤 기계에 손을 넣다 몸이 말려 들어감 | 빨간 상의 사고 (실험 04·10에서 사용) |
| _4GgKoR0OOw | 5 + 셔츠 2 | 레버를 손으로 돌리다 발이 기계에 끼임, 셔츠가 기계에 걸려 찢김 | 없음 (새 후보) |
| 7QFdUqCdML8 | 3 | 사람이 기계에 끌려 들어가고 동료가 잡아당김 | 없음 (새 후보) |
| 9XXineiOxSo | 8 | 컨베이어 위를 걷다 발이 끼임 | 없음. 택배 컨베이어 유형(폐기)과 비슷한지 확인 필요 |
| Zag5F6FXlj0 | 7 | 기계에 낀 사람을 구조하는 장면만 | 못 씀 (사고 전 장면 없음) |
| _a46_s5WViY | 3 | 닫히는 기계 양쪽 사이에 끼임 | 회전체 아님 |
| zaV3OjWE834 | 1 | 주방 기기에 머리카락 | 참고 |

### Safe and Unsafe Behaviours
- 698개 파일, 10.05GB. 파일 이름 앞 숫자가 클래스(README 표 순서로 추정): 0 통로 위반 210, 1 무허가 개입 108, 2 패널 열림 142, 3 지게차 과적 56, 4 안전 통로 75, 5 허가 개입 38, 6 패널 닫힘 32, 7 안전 운반 30
- 1번·5번 클립 6개를 받아 봄(`OmniGuard-data/open/safe_unsafe/`): **프레스 설비 바로 앞에서 작업자 1~3명이 손을 넣어 개입하는 고정 CCTV 장면**(5~17초). 우리 프레스 평소 작업 영상과 같은 상황이라 헛정지 측정에 바로 쓸 수 있다
- 개입 클립(1·5번) 146개를 모두 받으면 약 2GB

### 세훈 확인 (같은 날)
- iSafetyBench 끼임 클립 32개와 원본 6개를 직접 봄: **빨간 상의 말고는 회전체가 보이거나 닿기 전에 다가오는 장면이 있는 영상이 없다.** 클립 대부분이 이미 끼인 뒤이거나 구조 장면이다
- 결론: 공개 데이터(GitHub·Kaggle·HF)에는 "닿기 직전" 끼임 장면이 사실상 빨간 상의 하나뿐이다. 그래서 사고 시험 영상은 웹에서 직접 모으고(`docs/videos/manifest.csv`), 공개 데이터는 빨간 상의(iSafetyBench)와 헛정지 측정용 평소 작업(Safe and Unsafe Behaviours)에 쓴다. 심사 1번(데이터)에서 수집 이유로 쓴다

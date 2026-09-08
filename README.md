# Microwave Cavity Electromagnetic Simulation (FEniCSx / DOLFINx)

이 프로젝트는 **FEniCSx (DOLFINx)** 기반의 유한요소해석(FEM) 솔버를 이용하여 전자레인지(Microwave Oven) 내부의 3D 전자기장 분포, 음식물의 발열량(Heat Source), 그리고 S-파라미터(S11)를 계산하는 시뮬레이션 코드입니다. 

초기 단일 스크립트 형태에서 출발하여, 현재 다양한 메쉬(Mesh) 생성 방식과 외부 CAD 형상 불러오기 등을 지원하며, 유지보수 및 확장을 위한 **모듈화(Modularization)** 를 진행 중입니다.

## 🚀 Key Features
* **Full-Wave 3D EM Simulation:** 맥스웰 방정식(Maxwell's equations)의 주파수 도메인(Frequency Domain) 해석 (Nédélec Edge Elements 사용).
* **Multi-Method Mesh Generation:**
  * DOLFINx 내장 `create_box`를 활용한 빠른 정형 메쉬 및 좌표 기반 형상 인식.
  * `Gmsh` Python API (`occ`)를 이용한 내부 3D 모델링 및 Conformal Mesh(`fragment`) 생성.
  * 외부 설계 파일(`.geo`, `.step`) 임포트 지원.
* **Power Scaling & Heating:** 타겟 전력(예: 700W)을 기준으로 입력 포트의 전력을 스케일링하고 음식물 내부의 발열량($Q$)을 도출.
* **Multi-Material Support:** 공기(Air), 음식(Food - 구, 정육면체 등), 유리 턴테이블(Glass, $\epsilon_r = 5.5$) 등 다양한 매질의 유전율 적용.
* **S-Parameter Calculation:** 반사계수(S11) 계산을 통한 포트 임피던스 매칭 및 효율 분석.

## 🛠 Prerequisites (Dependencies)
이 프로젝트를 실행하기 위해서는 다음과 같은 라이브러리가 필요합니다. (Conda 환경 또는 Docker 사용을 권장합니다.)

* **FEniCSx (DOLFINx)** (v0.6.0 이상 권장)
* **UFL, Basix, FFCx**
* **Gmsh** (Python API 지원 버전)
* **PETSc, SLEPc** (MUMPS 또는 비반복 솔버용)
* **NumPy, SciPy**

## 📂 Project Structure (Current)
현재 프로젝트는 3개의 독립적인 스크립트로 구성되어 있으며, 각기 다른 메쉬 생성 방식을 시연합니다.

| 파일명 | 주요 특징 | 메쉬 생성 방식 |
|---|---|---|
| `grok2pack_ver2_python314_regularlo.py` | 좌표 수식 기반 매질 할당, 700W 스케일링, S11 계산 | DOLFINx 내장 `create_box` |
| `grok2pack_ver2_python314_regularlo_gmsh.py` | 챔버 및 음식물 형상 직접 코딩, Conformal Mesh 적용 | Gmsh Python API (`occ` 커널) |
| `python314_mesh_ver5_mesh2.py` | 외부 파일 로드, 유리 재질($\epsilon_r=5.5$) 추가, 회전 중심점 추출 | 외부 파일(`.geo` 또는 `.step`) |

## ⚙️ How to Run
스크립트를 실행하려면 터미널(또는 Conda 프롬프트)에서 다음과 같이 실행합니다.

```bash
# 예시: Gmsh API 기반 시뮬레이션 실행
python grok2pack_ver2_python314_regularlo_gmsh.py

# Microwave Simulation Suite

전자레인지 EM(전자기) 해석, Gmsh 메쉬 생성, 회전 가열/transient 열전달, 스케줄 최적화를 하나의 패키지로 모듈화한 프로젝트입니다.

## 디렉터리 구조

```
Final/
├── constants.py          # 물리 상수, 재료 물성, 공통 설정
├── config/
│   └── petsc.py          # PETSc 솔버 옵션 프리셋 (LU/MUMPS, GMRES/GAMG)
├── utils/
│   └── memory.py         # MPI 메모리 사용량 리포트
├── geometry/
│   ├── pork.py           # 균일 박스 메쉬용 pork(음식) 영역 좌표 판별
│   ├── gmsh_pork.py      # Gmsh OCC로 pork 챔버 형상 + 메쉬 생성
│   └── random.py         # 랜덤 전자레인지 형상(볼, 만두, 웨이브가이드) 생성
├── mesh/
│   └── geo.py            # .geo / STEP 파일에서 DOLFINx 메쉬 로드
├── em/
│   ├── postprocess.py    # Q, H, Poynting, S11, 700W 스케일링 (공통)
│   ├── uniform_solver.py # 균일 structured box EM 해석
│   ├── gmsh_pork_solver.py # Gmsh pork 챔버 EM 해석
│   └── geo_solver.py     # .geo 메쉬 기반 EM 해석 (S11 포함)
├── heat/
│   ├── transient.py      # transient/회전 열전달 솔버 (v1~v5)
│   ├── uniformity.py     # 회전 시 가열 균일도 분석
│   └── rotation.py       # 회전 중심 좌표 계산
└── optimization/
    └── schedule.py       # 그리디/Scipy/물리 기반 스케줄 최적화

main.py                   # 통합 CLI 진입점
```

## 실행 방법

모든 MPI 해석은 `mpirun`으로 실행합니다. 형상 생성만 하는 경우는 단일 프로세스로 실행 가능합니다.

### 1. 균일 박스 EM 해석 (기존 `grok2pack_ver2_python314_regularlo.py`)

```bash
mpirun -n 4 python main.py uniform_em --nx 31 --ny 26 --nz 31 --degree 2 --cell-type tet
```

### 2. Gmsh pork 챔버 EM 해석 (기존 `grok2pack_ver2_python314_regularlo_gmsh.py`)

```bash
mpirun -n 4 python main.py gmsh_pork_em --nx 62 --ny 52 --nz 62 --degree 2
```

### 3. 랜덤 전자레인지 형상 생성 (기존 `make_random_gmsh.py`)

```bash
python main.py random_geometry --variation LOW --prefix microwave_sim
```

출력: `microwave_sim.msh`, `microwave_sim.json`, `rotation_center.txt`

### 4. .geo 파일 EM 해석

```bash
mpirun -n 4 python main.py geo_em --geo path/to/model.geo --mesh-min 0.01 --mesh-max 0.04
```

### 5. 각도별 S11 스윕 (반사 계수)

```bash
mpirun -n 4 python main.py reflection_sweep \
  --template "cal_by_gmsh3/microwave_geometry_and_stackdumpling_{angle}.geo" \
  --csv result3/reflection_results.csv
```

### 6. 가열 균일도 분석

```bash
mpirun -n 4 python main.py heat_uniformity \
  --base-dir cal_by_gmsh3 \
  --rotation-center cal_by_gmsh3/rotation_center.txt
```

### 7. 회전 transient 열전달

```bash
mpirun -n 4 python main.py heat_rotating \
  --output result3/final_temperature_normal360T.xdmf \
  --dt 10 --total-time 360
```

### 8. 전체 파이프라인 (기존 `python314_mesh_ver5_mesh2.py` main)

```bash
mpirun -n 4 python main.py full_pipeline
```

## 모듈별 기능 요약

| 모듈 | 기능 |
|------|------|
| `constants.py` | 주파수, 진공 상수, pork/공기 유전율, 700W 목표 전력 |
| `config/petsc.py` | MUMPS 직접법 / GMRES+GAMG 반복법 옵션 |
| `utils/memory.py` | MPI 프로세스별 피크 메모리 출력 |
| `geometry/pork.py` | 구/정사면체/정육면체 pork 영역 정의 |
| `geometry/gmsh_pork.py` | OpenCASCADE Boolean + Gmsh 3D 메쉬 |
| `geometry/random.py` | 볼+만두+웨이브가이드 랜덤 형상 클래스 |
| `mesh/geo.py` | `.geo` 파일 메쉬 생성, Physical Group 매핑, 회전 중심 추출 |
| `em/postprocess.py` | 열원 Q, H, Poynting, S11, CSV 저장 (중복 제거) |
| `em/uniform_solver.py` | create_box 기반 EM + Weak Form Port |
| `em/gmsh_pork_solver.py` | Gmsh 태그 기반 EM |
| `em/geo_solver.py` | dumpling/geo 형상 EM + S11/포트 벡터 |
| `heat/transient.py` | static/rotating 열전달 (회전 Q 매핑) |
| `heat/uniformity.py` | XDMF Q 필드 기반 CoV/균일도 분석 |
| `heat/rotation.py` | XDMF에서 회전 중심 계산 |
| `optimization/schedule.py` | 그리디, Scipy, 물리 기반 최적 스케줄 탐색 |

## Python API 예시

```python
from mpi4py import MPI
from microwave_sim.em.uniform_solver import run_microwave_simulation
from microwave_sim.config.petsc import lu_mumps_options

comm = MPI.COMM_WORLD
E, Q, E_vec = run_microwave_simulation(
    number="test",
    n_elem_x=31, n_elem_y=26, n_elem_z=31,
    comm=comm,
    degree=2,
    petsc_options_em=lu_mumps_options(),
)
```

## 의존성

- Python 3.10+
- DOLFINx / FEniCSx
- Gmsh
- PETSc / petsc4py
- mpi4py, numpy, ufl
- h5py, scipy (열전달/최적화 모듈)


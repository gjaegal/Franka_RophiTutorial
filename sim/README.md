# FR3 실행 방법

프로젝트 루트에서 각 명령을 별도의 터미널에서 실행합니다.

## 시뮬레이션

```bash
python sim/run_env.py
```

```bash
python sim/run_controller.py
```

## 실제 로봇

```bash
python sim/run_hardware.py --ip <robot_ip> --control
```

```bash
python sim/run_mirror.py
```

```bash
python sim/run_controller.py --hardware
```

컨트롤러를 실행한 터미널에 키보드 포커스를 두세요.

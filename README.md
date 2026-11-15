# Aloha Pick & Place（TQC + HER）

用 MuJoCo 中的 Aloha 左臂（vx300s）抓起地面上的方块，搬到随机目标点（贴地或空中）。策略用 TQC + HER、稀疏奖励训练。

下面的命令都在 PowerShell 里、仓库根目录 `D:\AIControl\LiteFinalHomework` 下执行。解释器是 conda 环境 `LiteAlohaPAP` 的 `D:\anaconda3\envs\LiteAlohaPAP\python.exe`。不要用 Windows 商店占位的 `python`。

本次正式训练的运行目录（下文称 `RUN`）是 `runs/tqc_AlohaPickAndPlaceSparse-v0_20261114_103005`。`runs/` 已在 `.gitignore` 里，模型和视频都是本机路径，需要自行训练或把运行目录拷过来才会存在。

## 环境安装

- conda 环境名 `LiteAlohaPAP`，Python 3.12。本次训练记录的解释器版本是 3.12.16（`RUN/config.json` 的 `versions.python`）。
- 依赖清单是 `requirements.txt`。其中 `mujoco==3.3.7` 是刻意锁住的：plan 第 9 节写明 mujoco 4.x 与 gymnasium-robotics 的兼容性未验证，所以不用 4.x。
- `requirements.txt` 开头的安装说明是：激活该环境后执行 `pip install -r requirements.txt`。torch 行通过 `--extra-index-url https://download.pytorch.org/whl/cu130` 安装 CUDA 13.0 版。
- 装完后用冒烟脚本核对版本、CUDA、MuJoCo 和离屏视频：

```powershell
$env:PYTHONIOENCODING = "utf-8"
& D:\anaconda3\envs\LiteAlohaPAP\python.exe -u scripts\smoke_test.py
```

脚本会在 `runs/smoke/smoke.mp4` 写一段 1 秒的测试视频。

## 脚本用法

每条命令都可以直接复制。除特别注明的以外，先执行 `$env:PYTHONIOENCODING = "utf-8"`。

| 脚本 | 用途 | 命令 |
|---|---|---|
| `scripts/smoke_test.py` | 检查依赖版本、CUDA、能否加载 `models/push.xml` 并写出测试视频 | 见上一节 |
| `scripts/check_model.py info` | 打印模型摘要和 home 姿态下的 weld 残差 | 见下 |
| `scripts/check_model.py view` | 用 MuJoCo 窗口查看场景。这条会打开窗口 | 见下 |
| `scripts/check_model.py track` | 在裁剪盒网格上统计 pinch 跟踪误差 | 见下 |
| `scripts/check_model.py grasp` | 在 5 个默认 xy 上做下降、闭合、抬起并保持 | 见下 |
| `scripts/check_model.py video` | 把 grasp 过程用 `eval_cam` 录成 mp4 | 见下 |
| `scripts/eval_scripted.py` | 用脚本策略验证环境可解，并录视频 | 见下 |
| `scripts/train_tqc_pap.py` | TQC + HER 训练 | 见「正式训练」 |
| `scripts/eval_policy.py` | 加载训练好的策略，做独立回合评估 | 见「评估」 |
| `tests/test_pick_place_env.py` | 环境回归。直接用解释器运行，不依赖 pytest | 见下 |

```powershell
& D:\anaconda3\envs\LiteAlohaPAP\python.exe -u scripts\check_model.py info --seed 0
```

```powershell
& D:\anaconda3\envs\LiteAlohaPAP\python.exe -u scripts\check_model.py view --seed 0
```

```powershell
& D:\anaconda3\envs\LiteAlohaPAP\python.exe -u scripts\check_model.py track --seed 0
```

```powershell
& D:\anaconda3\envs\LiteAlohaPAP\python.exe -u scripts\check_model.py grasp --seed 0
```

```powershell
& D:\anaconda3\envs\LiteAlohaPAP\python.exe -u scripts\check_model.py video --seed 0
```

```powershell
& D:\anaconda3\envs\LiteAlohaPAP\python.exe -u scripts\eval_scripted.py --episodes 100 --seed 0 --videos 3 --out runs\m3
```

```powershell
& D:\anaconda3\envs\LiteAlohaPAP\python.exe -u tests\test_pick_place_env.py
```

M5 复跑回归的结果是 14 PASS、0 FAIL。

## 建模

场景分成两个文件（plan 第 3 节，M1 定稿见 `tasks/M1_report.md`）：

- `models/aloha_pap.xml`：从 `models/aloha.xml` 复制后改成 pick & place 用的左臂。删掉重复的夹爪 equality 和 6 个臂关节电机，只留夹爪执行器；臂部连杆加 `gravcomp="1"`。
- `models/pick_place.xml`：include 上面的臂，加上地面、mocap 刚体、weld、物块、目标 site 和 `eval_cam`。

控制方式是 mocap + weld。夹爪竖直朝下，姿态锁死，mocap 拖着 pinch 点走。

| 项 | 取值 | 出处 |
|---|---|---|
| pinch 点 | 垫片质心，gripper_base 坐标 `(0, 0.003636, 0.108806)` | plan 3.2，`tasks/M1_report.md` |
| weld | `body1="mocap" body2="left/pinch"`，单位 `relpose="0 0 0 1 0 0 0"`，`solref="0.01 1"`，`torquescale="5"` | plan 3.3 |
| 夹爪 equality | `solref="0.005 1"` | plan 3.2 |
| `GRASP_QUAT`（wxyz） | `(0, -0.7071067811865475, 0.7071067811865476, 0)` | plan 3.4；定义在 `scripts/check_model.py` 和 `aloha_envs/aloha_pick_place_env.py` |
| `HOME_QPOS` | `(0.00040629901495, -0.047747780177, 0.10111069065, -3.110639406e-07, 1.518228499, -0.00039001127121, 0.037, 0.037)` | 同上。顺序是 waist、shoulder、elbow、forearm_roll、wrist_angle、wrist_rotate、两指 |
| 裁剪盒（pinch） | x∈[0.15, 0.40]，y∈[−0.15, 0.15]，z∈[0.02, 0.30] | plan 3.5 |
| 物块 / 目标 xy 采样 | x∈[0.25, 0.40]，y∈[−0.15, 0.15] | plan 3.5 |
| home 时 pinch | (0.30, 0, 0.20) | plan 3.4 |

`pick_place.xml` 里的 keyframe 名是 `home`。物块是边长 4 cm 的立方体（半边长 0.02），质量 0.05 kg。

## 观测、动作与奖励

定义在 `aloha_envs/aloha_pick_place_env.py`。回合固定 50 步（`aloha_envs/__init__.py` 的 `max_episode_steps=50`），不因成功提前结束。仿真 `timestep=0.002`，`n_substeps=20`，所以控制周期 dt = 0.04 s，`render_fps` 为 25。

观测是字典。`achieved_goal` 和 `desired_goal` 都是 3 维物块 / 目标位置。`observation` 在环境里是 25 维，经 `TimeFeatureWrapper` 后为 26 维：

| 分量 | 维度 | 含义 |
|---|---|---|
| grip_pos | 3 | pinch 点世界坐标 |
| object_pos | 3 | `obj_site` 世界坐标 |
| object_rel_pos | 3 | object_pos − grip_pos |
| gripper_state | 2 | 左右指关节位置 |
| object_rot | 3 | 物块欧拉角 |
| object_velp | 3 | （物块线速度 − 夹爪线速度）× dt |
| object_velr | 3 | 物块角速度 × dt |
| grip_velp | 3 | pinch 点线速度 × dt |
| gripper_vel | 2 | 左右指关节速度 × dt |
| time（wrapper 追加） | 1 | 剩余时间比例 `1 - step / 50`，范围 [0, 1] |

动作是 4 维，范围 [−1, 1]：

- 前 3 维是 pinch 的位置增量，乘 `ACTION_SCALE = 0.05` m，加到当前 pinch 位置上，再裁进上面的裁剪盒。mocap 姿态固定为 `GRASP_QUAT`。
- 第 4 维是夹爪绝对开合：`0.002 + (a + 1) / 2 * (0.037 - 0.002)`。−1 全闭，+1 全开。

稀疏奖励（`AlohaPickAndPlaceSparse-v0`）：距离 `d = ‖achieved_goal − desired_goal‖`，`d > 0.05` 时奖励 −1，否则奖励 0。成功判定是 `d < 0.05`（正好等于 0.05 时奖励为 0，但不算成功）。`p_air` 默认 0.5：目标以这个概率放在空中，高度 = 物块静止高度 + U(0, 0.20)，否则贴地。

保持率是辅助指标：回合最后 10 步全部成功的回合占比。验收只看最后一步是否成功。

## 训练超参

下面是 `RUN/config.json` 里这次 1,000,000 步训练的实际取值，不是计划里的默认表。

| 项 | 取值 |
|---|---|
| 算法 / 策略 | TQC，`MultiInputPolicy` |
| 网络 | `net_arch = [512, 512, 512]`，`n_critics = 2` |
| learning_rate | 0.001 |
| buffer_size | 1,000,000 |
| batch_size | 2048 |
| gamma | 0.95 |
| tau | 0.05 |
| learning_starts | 10,000 |
| train_freq | 1 |
| gradient_steps | 8（命令行是 null，按 `n_envs` 解析，见 `resolved.gradient_steps`） |
| n_envs | 8，`SubprocVecEnv` |
| total_timesteps | 1,000,000 |
| seed | 0（训练环境种子 0..7） |
| device | cuda |
| 环境 | `AlohaPickAndPlaceSparse-v0`，`p_air = 0.5`，`distance_threshold = 0.05`，`max_episode_steps = 50` |
| TimeFeatureWrapper | 开启。`time_feature_max_steps = 50`。wrapper 顺序：`gym.make(TimeLimit)` → `TimeFeatureWrapper` → `Monitor` |
| HER | `HerReplayBuffer`，`goal_selection_strategy = "future"`，`n_sampled_goal = 4`，`handle_timeout_termination = true` |
| 训练中评估 | 每 20,000 环境步一次（`eval_freq = 2500` 次向量环境步），每次 20 回合，种子固定为 10000..10019 |
| checkpoint | 每 100,000 环境步（`save_freq = 12500`） |
| 视频 | 每次评估录 1 段，种子 20000，相机 `eval_cam`，320×240，25 fps |

## 正式训练（1M 步）

```powershell
$env:PYTHONIOENCODING = "utf-8"
Set-Location D:\AIControl\LiteFinalHomework
& D:\anaconda3\envs\LiteAlohaPAP\python.exe -u scripts\train_tqc_pap.py --total-timesteps 1000000
```

- 要训 3M 步，把 `--total-timesteps` 改成 `3000000`。
- 启动后第一行会打印 `RUN_DIR=...`。下文用 `<RUN>` 表示这个目录，用 `<RUN_NAME>` 表示它的目录名（形如 `tqc_AlohaPickAndPlaceSparse-v0_<时间戳>`）。
- 所有超参数都可以从命令行修改，查看方法：`python scripts\train_tqc_pap.py --help`。

### 查看 TensorBoard

```powershell
& D:\anaconda3\envs\LiteAlohaPAP\python.exe -m tensorboard.main --logdir <RUN>\tb --port 6006
```

浏览器打开 http://localhost:6006 ，主要看 `eval/success_rate`、`eval/hold_rate`、`rollout/success_rate` 三条曲线。

### 中断后续训

用最新的 `checkpoints\<步数>_steps.zip` 和 `checkpoints\replay_buffer_latest.pkl` 续训。注意：
- `--total-timesteps` 是**还要再训**的步数；
- `--run-name` 必须用原来的目录名，曲线才能接上；
- `--n-envs` 必须与原来一致（默认 8）。

```powershell
& D:\anaconda3\envs\LiteAlohaPAP\python.exe -u scripts\train_tqc_pap.py --resume-model <RUN>\checkpoints\<步数>_steps.zip --resume-buffer <RUN>\checkpoints\replay_buffer_latest.pkl --run-name <RUN_NAME> --total-timesteps <剩余步数> --n-envs 8
```

### 预计资源（RTX 4060 Laptop 实测估算）

| 项 | 1M 步 | 3M 步 |
|---|---|---|
| 耗时 | 约 5 小时 | 约 15 小时 |
| 磁盘 | 约 0.84 GB | 约 1.28 GB |
| 内存峰值 | 约 8 GB | 约 8 GB |
| 显存 | 约 230 MB | 约 230 MB |

瓶颈在 GPU 梯度更新（batch 2048，每个环境步一次更新，每个梯度步约 18 ms）。

### 注意事项
- 训练开始时的 “obs is not within the observation space” 警告来自 `TimeFeatureWrapper`，可以忽略。
- 在训练窗口按 Ctrl+C，会先保存 `final_model.zip` 和 replay buffer 再退出（退出码 130）。直接关掉窗口这类强制结束来不及保存，此时用最近的 checkpoint 续训。checkpoint 每 100k 步自动保存一次，最坏约丢 30 分钟。
- 训练中的 `eval/success_rate` 和 `best_success_model` 每次评估都用同一组 20 个种子，成功率偏乐观；最终以独立的 100 回合评估为准。
- `--gradient-steps 4` 吞吐量约为默认值的 1.9 倍，但每个样本的更新次数减半，第一次正式训练不建议使用。

### 输出目录

```
runs/<RUN_NAME>/
  ├── tb/              TensorBoard
  ├── best_model/      best_success_model.zip（按最后一步成功率选）、best_model.zip（按平均奖励选，仅供参考）
  ├── checkpoints/     <步数>_steps.zip、replay_buffer_latest.pkl
  ├── videos/          每次评估录 1 段：step_<N>_<success|fail>.mp4
  ├── evaluations.npz
  ├── config.json      本次全部参数、种子、软件版本、git commit
  └── final_model.zip
```

## 评估

`scripts/eval_policy.py` 按 `RUN/config.json` 构造和训练相同的 wrapper，用确定性动作跑独立种子。它只加载策略，不加载 replay buffer。回合 i 的 reset 种子是 `seed_base + i`。默认种子 30000–30099，与训练种子 0–7、训练中评估 10000–10019、视频种子 20000 都不重叠。输出写到 `RUN/eval_policy/<模型名>_seed<seed_base>/`。该目录已存在且非空时脚本会退出；换目录用 `--out-dir`，覆盖用 `--overwrite`。

验收评估（四个模型，各 100 回合，并各录前 4 个回合）：

```powershell
$env:PYTHONIOENCODING = "utf-8"
Set-Location D:\AIControl\LiteFinalHomework
& D:\anaconda3\envs\LiteAlohaPAP\python.exe -u scripts\eval_policy.py --model runs\tqc_AlohaPickAndPlaceSparse-v0_20261114_103005\best_model\best_success_model.zip runs\tqc_AlohaPickAndPlaceSparse-v0_20261114_103005\final_model.zip runs\tqc_AlohaPickAndPlaceSparse-v0_20261114_103005\checkpoints\900000_steps.zip runs\tqc_AlohaPickAndPlaceSparse-v0_20261114_103005\checkpoints\800000_steps.zip --n-episodes 100 --seed-base 30000 --n-videos 4
```

同一模型换一组种子再评 100 回合：

```powershell
& D:\anaconda3\envs\LiteAlohaPAP\python.exe -u scripts\eval_policy.py --model runs\tqc_AlohaPickAndPlaceSparse-v0_20261114_103005\best_model\best_success_model.zip --n-episodes 100 --seed-base 40000
```

`--render-human` 会打开 MuJoCo 窗口，只在有人看着屏幕时使用。

每个模型的成功率都 ≥ 80% 时进程退出码为 0，否则为 1。

## 最终结果

训练跑满 1,000,000 步，没有续训，git commit `1827be578cefdf22c1f6cfe034ba850ebabefa42`（`RUN/config.json`）。`RUN/perf.json`：学习阶段 `post_seconds = 14713.5`（约 4.09 小时），`post_sps_wall = 67.28`。`best_success_model.zip` 来自 940,000 步（`RUN/best_model/selection.json`：训练中评估成功率 1.0，平均奖励 −6.05）。

独立评估用最后一步 `is_success`。下表来自 `RUN/eval_policy/*/summary.json`。验收对象是 `best_success_model.zip`。

| 模型 | seed_base | 成功率 | 保持率 | 贴地 | 空中 | 平均最终距离 (m) |
|---|---:|---:|---:|---:|---:|---:|
| `best_success_model.zip` | 30000 | 98/100 (0.980) | 98/100 (0.980) | 46/46 (1.000) | 52/54 (0.963) | 0.021500 |
| `final_model.zip` | 30000 | 96/100 (0.960) | 95/100 (0.950) | 45/46 (0.978) | 51/54 (0.944) | 0.023113 |
| `checkpoints/900000_steps.zip` | 30000 | 99/100 (0.990) | 99/100 (0.990) | 46/46 (1.000) | 53/54 (0.981) | 0.025628 |
| `checkpoints/800000_steps.zip` | 30000 | 97/100 (0.970) | 96/100 (0.960) | 46/46 (1.000) | 51/54 (0.944) | 0.025839 |
| `best_success_model.zip` 复验 | 40000 | 97/100 (0.970) | 97/100 (0.970) | 50/50 (1.000) | 47/50 (0.940) | 0.022716 |

同一模型在 seed 30000 和 seed 40000 上的成功率相差 1 个百分点（0.980 与 0.970）。这组种子里，90 万步的 checkpoint 比 `best_success_model.zip` 高 1 个成功回合。验收仍以 `best_success_model.zip` 为准：它的独立成功率是 98%，高于 80% 的验收线。

按上一节的验收命令，在新开的 PowerShell 里再执行一次，并加上 `--out-dir` 把输出写到临时目录。四个模型的 `episodes.csv` 与第一次逐字节相同。

训练中的 `eval/success_rate`（`RUN/evaluations.npz`，每次 20 回合，种子始终是 10000–10019，其中空中目标 11 个、贴地 9 个，仅供参考）：最终值 1.00，最大值 1.00，第一次达到 1.00 是 360,000 步，共 20 次评估为 1.00。最后 10 次评估的成功率和保持率平均都是 0.995。独立 100 回合的 0.980 低于这条训练曲线上的 1.00。

本机路径（相对仓库根目录；`runs/` 不入库）：

- 验收模型：`runs/tqc_AlohaPickAndPlaceSparse-v0_20261114_103005/best_model/best_success_model.zip`
- 最终模型：`runs/tqc_AlohaPickAndPlaceSparse-v0_20261114_103005/final_model.zip`
- TensorBoard：`runs/tqc_AlohaPickAndPlaceSparse-v0_20261114_103005/tb`
- 训练视频：`runs/tqc_AlohaPickAndPlaceSparse-v0_20261114_103005/videos/`（`step_<N>_success.mp4` 或 `step_<N>_fail.mp4`）
- 独立评估视频：`runs/tqc_AlohaPickAndPlaceSparse-v0_20261114_103005/eval_policy/best_success_model_seed30000/videos/`，另外三个模型在同级的 `final_model_seed30000`、`900000_steps_seed30000`、`800000_steps_seed30000` 目录里

## 训练曲线

环境里没有 matplotlib，下表由 `RUN/evaluations.npz` 算出。每一行都是同一组 20 个种子。除了每 10 万步，还加了第一次达到 0.8 和 1.0 的评估点。

| 步数 | 标记 | success_rate | hold_rate | 贴地 | 空中 | final_distance (m) |
|---:|---|---:|---:|---:|---:|---:|
| 100000 | 每 10 万步 | 4/20 (0.200) | 4/20 (0.200) | 3/9 (0.333) | 1/11 (0.091) | 0.1407 |
| 200000 | 每 10 万步 | 15/20 (0.750) | 15/20 (0.750) | 7/9 (0.778) | 8/11 (0.727) | 0.0516 |
| 220000 | 首次 ≥ 0.8 | 16/20 (0.800) | 16/20 (0.800) | 7/9 (0.778) | 9/11 (0.818) | 0.0531 |
| 300000 | 每 10 万步 | 19/20 (0.950) | 19/20 (0.950) | 9/9 (1.000) | 10/11 (0.909) | 0.0395 |
| 360000 | 首次 ≥ 1.0 | 20/20 (1.000) | 20/20 (1.000) | 9/9 (1.000) | 11/11 (1.000) | 0.0179 |
| 400000 | 每 10 万步 | 18/20 (0.900) | 18/20 (0.900) | 8/9 (0.889) | 10/11 (0.909) | 0.0373 |
| 500000 | 每 10 万步 | 18/20 (0.900) | 18/20 (0.900) | 9/9 (1.000) | 9/11 (0.818) | 0.0667 |
| 600000 | 每 10 万步 | 20/20 (1.000) | 20/20 (1.000) | 9/9 (1.000) | 11/11 (1.000) | 0.0177 |
| 700000 | 每 10 万步 | 20/20 (1.000) | 20/20 (1.000) | 9/9 (1.000) | 11/11 (1.000) | 0.0183 |
| 800000 | 每 10 万步 | 20/20 (1.000) | 20/20 (1.000) | 9/9 (1.000) | 11/11 (1.000) | 0.0184 |
| 900000 | 每 10 万步 | 20/20 (1.000) | 20/20 (1.000) | 9/9 (1.000) | 11/11 (1.000) | 0.0223 |
| 1000000 | 每 10 万步 | 20/20 (1.000) | 20/20 (1.000) | 9/9 (1.000) | 11/11 (1.000) | 0.0149 |

查看完整曲线：

```powershell
& D:\anaconda3\envs\LiteAlohaPAP\python.exe -m tensorboard.main --logdir runs\tqc_AlohaPickAndPlaceSparse-v0_20261114_103005\tb --port 6006
```

浏览器打开 http://localhost:6006 。

## 已知局限

- 物块会被压进地面。mocap weld 对力没有上限，夹爪从上方压住物块时，物块陷进地面约 15 mm（地面受力约 147 N）。调接触参数无效。松开后物块不会弹飞。出处：plan 第 9 节。
- 物块在夹爪里会沿世界坐标 +x 滑。回合结束时相对抓取时的偏移，贴地中位约 5.5 mm（最大约 9.1 mm），空中中位约 7.8 mm（最大约 12.6 mm）。加大扭转摩擦也不消失。成功阈值是 5 cm，策略要按实时物块位置调整。出处：plan 第 9 节。
- `eval_cam` 看近处空中目标时，目标会被夹爪挡住一部分。home 姿态下，采样区空中角点 (0.25, 0.15, 0.22) 的红球被夹爪挡住了大半（`tasks/M1_review_r1.md` R2）。M1 把它推迟到 M4，M4 没有改相机。M5 按用户决定保持原相机，评估视频仍用 `eval_cam`。
- 训练中每次评估都用固定的 20 个种子（10000–10019）。曲线上的 1.00 不能代替独立 100 回合。
- 正式训练只跑了随机种子 0 这一次。
- 在控制台按 Ctrl+C 会保存模型和 replay buffer。直接结束进程（例如关闭窗口）来不及保存。checkpoint 每 100,000 步写一次，这种情况下最坏大约丢掉 30 分钟。出处：plan 第 9 节。
- 训练和评估开始时，`TimeFeatureWrapper` 会触发 “obs is not within the observation space” 警告。外层 26 维观测和观测空间是一致的，警告可以忽略（`tasks/M4_review_r1.md` N1）。

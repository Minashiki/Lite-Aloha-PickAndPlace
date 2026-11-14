# Aloha Pick & Place（TQC + HER）

用 MuJoCo 中的 Aloha 左臂（vx300s）抓起地面上的方块，搬到随机目标点（贴地或空中）；策略用 TQC + HER、稀疏奖励训练。

> 本文件目前只写了正式训练的用法，完整说明（建模、观测/动作/奖励、评估、结果）在 M5 补全。

## 环境
- conda 环境 `LiteAlohaPAP`（Python 3.12），依赖见 `requirements.txt`
- 以下命令都在 PowerShell 中、项目根目录下执行

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

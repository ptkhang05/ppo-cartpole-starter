# PPO CartPole Starter

Huấn luyện và đánh giá **Proximal Policy Optimization (PPO)** trên môi trường `CartPole-v1` bằng Stable-Baselines3.

## 1. Yêu cầu

- Python 3.11 đến 3.13
- Windows 10 hoặc Windows 11
- CPU là đủ cho cấu hình mặc định

## 2. Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Huấn luyện

```bash
python -m src.train --timesteps 100000 --seed 42
```

Chạy thử với số bước ngắn hơn:

```bash
python -m src.train --timesteps 5000 --seed 42
```

Đầu ra:

```text
models/ppo_cartpole.zip          mô hình cuối
models/best/best_model.zip       mô hình có mean return đánh giá cao nhất
logs/train/monitor.csv           return và độ dài từng episode train
logs/eval/evaluations.npz        kết quả đánh giá định kỳ
outputs/training_config.json     cấu hình huấn luyện
outputs/training_curve.png       đồ thị return
outputs/training_curve.csv       dữ liệu của đồ thị
```

## 4. Đánh giá và ghi video

```bash
python -m src.evaluate --model models/ppo_cartpole.zip --episodes 20 --video
```

Đầu ra:

```text
outputs/evaluation_summary.json  mean, standard deviation, min và max return
outputs/evaluation_episodes.csv  kết quả từng episode
videos/ppo-cartpole-episode-0.mp4 video một episode đánh giá
```

Nếu chỉ muốn lấy số liệu mà không ghi video:

```bash
python -m src.evaluate --model models/ppo_cartpole.zip --episodes 20
```

## 5. Chạy kiểm thử

```bash
python -m pytest -q
```

## 6. Cấu hình PPO

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `learning_rate` | `3e-4` | Kích thước bước cập nhật tham số |
| `n_steps` | `1024` | Số transition thu trước mỗi pha cập nhật |
| `batch_size` | `64` | Số mẫu trong một mini-batch |
| `n_epochs` | `10` | Số lượt tái sử dụng rollout |
| `gamma` | `0.99` | Hệ số chiết khấu |
| `gae_lambda` | `0.95` | Hệ số GAE |
| `clip_range` | `0.2` | Biên clipping của probability ratio |

## Nguồn kỹ thuật

- [PPO paper](https://arxiv.org/abs/1707.06347)
- [Stable-Baselines3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Gymnasium CartPole documentation](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

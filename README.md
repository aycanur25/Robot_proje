# Reinforcement Learning with Mujoco-Pytorch

Bu proje, **Mujoco** ortamlarında **Pekiştirmeli Öğrenme (Reinforcement Learning)** algoritmalarını kullanarak yapılan deneysel çalışmaları içermektedir. Kullanılan algoritmalar ve test edilen ortamlar şunlardır:

- **Proximal Policy Optimization (PPO)**
- **Q-Learning**
- **Soft Actor-Critic (SAC)**
- **Deep Deterministic Policy Gradient (DDPG)**

## 🧑‍💻 Kullanılan Algoritmalar

- **PPO (Proximal Policy Optimization)**: Proximal Policy Optimization, güvenli politika güncellemeleri sağlayarak daha istikrarlı öğrenme süreci sunar.
- **Q-Learning**: Q-Learning, değer tabanlı bir pekiştirmeli öğrenme algoritmasıdır. Bu projede **Mountain Car** ortamında uygulandı.
- **SAC (Soft Actor-Critic)**: SAC, entropi temelli bir algoritmadır ve keşif ile istikrarı dengesiz bir şekilde sağlar. **Mountain Car** ortamında kullanıldı.
- **DDPG (Deep Deterministic Policy Gradient)**: Sürekli aksiyon alanlarıyla çalışmak için kullanılan bir algoritmadır. **Mountain Car** ortamında uygulandı.

## 🏞️ Ortamlar

Projede test edilen ortamlar aşağıdaki gibidir:

- **Half Cheetah** (Mujoco)
- **Mountain Car** (OpenAI Gym)

## 📈 Performans

Algoritmaların her biri, ilgili ortamlar üzerinde eğitim ve test süreçlerinden geçirilmiştir. Eğitim sürecinin görselleştirilmesi sağlanmıştır ve performans sonuçları izlenebilir.

## 🚀 Başlangıç

### Gereksinimler

Bu projeyi çalıştırmak için aşağıdaki gereksinimlere ihtiyacınız olacak:

- Python 3.x
- gym 0.17.x+
- mujoco_py 2.0.x+
- pytorch 1.6.x+

### Kurulum

Projeyi çalıştırmaya başlamak için aşağıdaki adımları izleyin:

1. Repo'yu klonlayın:

    ```bash
    git clone https://github.com/<repo_adı>.git
    cd <repo_adı>
    ```

2. Gerekli kütüphaneleri yükleyin:

    ```bash
    pip install -r requirements.txt
    ```

### Mujoco Kurulumu

**Mujoco** ortamları için `mujoco_py` kütüphanesinin kurulması gerekmektedir. Bu, simülasyonları çalıştırmak için gereklidir. Kurulum talimatlarını [Mujoco](https://github.com/openai/mujoco) ve [mujoco_py](https://github.com/openai/mujoco-py) reposunda bulabilirsiniz.

Mujoco'yu kurduktan sonra aşağıdaki adımları izleyerek ortamları çalıştırabilirsiniz.

1. **Mujoco**'yu indirip kurun ve çevresel değişkenlerinizi ayarlayın:
   
   ```bash
   export MUJOCO_PY_MUJOCO_PATH=/path/to/mujoco
   export MUJOCO_PY_MJKEY_PATH=/path/to/mjkey.txt

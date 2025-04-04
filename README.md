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

# Parametreler:
--env_name: Kullanılacak ortam adı. (default: 'MountainCar-v0')

'Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'HumanoidStandup-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Walker2d-v2', 'Swimmer-v2', 'Reacher-v2'

--algorithm: Seçilen algoritma. ('PPO', 'Q-Learning', 'SAC', 'DDPG')

--train: Eğitim aşamasını başlatır. (default: True)

--render: Ortamı görselleştirir. (default: False)

--epochs: Eğitim süresi, toplam epoch sayısı. (default: 1000)

--entropy_coef: Entropi katsayısı. (default: 0.01)

--critic_coef: Critic katsayısı. (default: 0.5)

--learning_rate: Öğrenme oranı. (default: 0.0003)

--gamma: İndirim faktörü. (default: 0.99)

--lmbda: GAE'de kullanılan lambda katsayısı. (default: 0.95)

--eps_clip: Aktör ve kritik ağlarının clip aralığı. (default: 0.2)

--K_epoch: Eğitimde kullanılan epoch sayısı. (default: 64)

--T_horizon: Bir jenerasyonun eğitime başlamadan önceki zaman adımları. (default: 2048)

--hidden_dim: Aktör ve kritik ağlarının gizli katman boyutu. (default: 64)

--minibatch_size: Mini-batch boyutu. (default: 64)

--tensorboard: TensorBoard desteği. (default: False)

--load: Yüklenmesi gereken model ismi. (default: 'no')

--save_interval: Modelin kaydedilme sıklığı. (default: 100)

--print_interval: Eğitim sırasında yazdırma sıklığı. (default: 20)

--use_cuda: CUDA kullanım durumu. (default: True)

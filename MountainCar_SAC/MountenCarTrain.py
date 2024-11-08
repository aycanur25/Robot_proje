import gym
import time
import torch
from Sac_Network.Discrete_SAC_Agent import SACAgent

print(torch.cuda.is_available())

# Gym ortamını başlat
env = gym.make('MountainCar-v0')
sac_agent = SACAgent(env)

print('Observation Space:', env.observation_space)
print('Action Space:', env.action_space)

# Süre limiti ayarı
TIME_LIMIT = 500
env = gym.wrappers.TimeLimit(
    gym.envs.classic_control.MountainCarEnv(),
    max_episode_steps=TIME_LIMIT + 1,
)

# Değişkenler
last_score = -500
run_score = -500
best = -500

# Eğitim döngüsü
for episode in range(400):
    last_score = run_score
    if best < last_score:
        best = last_score
    run_score = 0
    state = env.reset()
    done = False
    evaluation_episode = episode % 20 == 1  # 20. bölümde değerlendirme yapma

    print(f'Episode: {episode + 1}/{400} | Last Score: {last_score} | Best: {best}', end=' ')

    for step in range(TIME_LIMIT):
        # Değerlendirme bölümü
        if evaluation_episode:
            action = sac_agent.get_next_action(state, evaluation_episode=True)
            result = env.step(action.item())  # Hangi aksiyon alınacak?

            # Dönüş değerini yazdır
            print(f"Result Length: {len(result)} | Result: {result}")

            # Dönüş değerine göre işlem yapalım
            if len(result) == 3:  # Eğer sadece 3 değer döndürülüyorsa
                next_state, reward, done = result
                info = {}  # info'yu boş bir dict olarak atıyoruz
            elif len(result) == 4:  # Eğer 4 değer döndürülüyorsa
                next_state, reward, done, info = result
            elif len(result) == 5:  # Eğer 5 değer döndürülüyorsa
                next_state, reward, done, info, extra = result  # 5. değeri alıyoruz
            else:
                raise ValueError(f"Unexpected result length: {len(result)} from env.step()")

            run_score += reward
            if done:
                break
            state = next_state
            env.render(mode='human')
            time.sleep(0.01)

        # Eğitim bölümü
        else:
            action = sac_agent.get_next_action(state, evaluation_episode=False)
            result = env.step(action)

            # Dönüş değerini yazdır
            print(f"Result Length: {len(result)} | Result: {result}")

            # Dönüş değerine göre işlem yapalım
            if len(result) == 3:  # Eğer sadece 3 değer döndürülüyorsa
                next_state, reward, done = result
                info = {}  # info'yu boş bir dict olarak atıyoruz
            elif len(result) == 4:  # Eğer 4 değer döndürülüyorsa
                next_state, reward, done, info = result
            elif len(result) == 5:  # Eğer 5 değer döndürülüyorsa
                next_state, reward, done, info, extra = result  # 5. değeri alıyoruz
            else:
                raise ValueError(f"Unexpected result length: {len(result)} from env.step()")

            sac_agent.train_on_transition(state, action, next_state, reward, done)
  

import numpy as np
import gym

env = gym.make("MountainCar-v0", render_mode="human")

# Hiperparametreler
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0  # Keşif oranı
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 5000
num_bins = 20  # Durum uzayını ayrıklaştırma için bölme sayısı

# Durum uzayını ayrıklaştırma
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bins = [np.linspace(b[0], b[1], num_bins) for b in state_bounds]

def discretize_state(state):
    """Durumu ayrık hale getir."""
    state_indices = [
        np.digitize(state[i], state_bins[i]) - 1
        for i in range(len(state))
    ]
    return tuple(state_indices)

# Q-Tablosu
q_table = np.random.uniform(low=-1, high=0, size=(num_bins, num_bins, env.action_space.n))

# Q-Learning Döngüsü
rewards = []
for episode in range(num_episodes):
    observation, _ = env.reset()  # Tuple dönerse yalnızca observation alınır
    state = discretize_state(observation)
    total_reward = 0

    done = False
    while not done:
        # Epsilon-greedy eylem seçimi
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # Rastgele eylem (keşif)
        else:
            action = np.argmax(q_table[state])  # En iyi eylem (sömürü)

        # Eylemi gerçekleştir
        next_observation, reward, done, truncated, _ = env.step(action)
        done = done or truncated  # Truncated durumunu ekleyin
        next_state = discretize_state(next_observation)

        # Q-Tablosunu güncelle
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + discount_factor * q_table[next_state][best_next_action]
        td_error = td_target - q_table[state][action]
        q_table[state][action] += learning_rate * td_error

        # Durum ve toplam ödülü güncelle
        state = next_state
        total_reward += reward

    # Epsilon'u azalt (keşfi azalt)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    rewards.append(total_reward)

    if episode % 100 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

# Eğitim sonrası test
observation, _ = env.reset()
state = discretize_state(observation)
done = False
while not done:
    action = np.argmax(q_table[state])
    next_observation, reward, done, truncated, _ = env.step(action)
    done = done or truncated
    next_state = discretize_state(next_observation)
    state = next_state
    env.render()

env.close()
import numpy as np
import gym

class ReplayBuffer:

    def __init__(self, environment, capacity=5000):
        transition_type_str = self.get_transition_type_str(environment)
        self.buffer = np.zeros(capacity, dtype=transition_type_str)
        self.weights = np.zeros(capacity)
        self.head_idx = 0
        self.count = 0
        self.capacity = capacity
        self.max_weight = 10**-2
        self.delta = 10**-4
        self.indices = None

    def get_transition_type_str(self, environment):
        # State boyutunu al
        state_dim = environment.observation_space.shape[0]

        # Action tipini kontrol et: Discrete mi, yoksa Box mı
        if isinstance(environment.action_space, gym.spaces.Discrete):
            # Discrete aksiyon alanı, sayısal bir değer
            action_dim = 1
        elif isinstance(environment.action_space, gym.spaces.Box):
            # Box aksiyon alanı, vektör tipi
            action_dim = environment.action_space.shape[0]
        else:
            raise ValueError("Beklenmedik action_space tipi")

        # Geçiş tipini belirle
        transition_type_str = np.dtype([
            ('state', np.float32, (state_dim,)),  # state
            ('action', np.float32, (action_dim,)),  # action
            ('reward', np.float32),  # reward
            ('next_state', np.float32, (state_dim,)),  # next_state
            ('done', bool)  # done (boolean) -> np.bool yerine bool kullanıldı
        ])

        return transition_type_str

    def add_transition(self, transition):
        # Geçiş verisini buffer'a ekle
        # Transition verisi doğru formatta olmalı: (state, action, reward, next_state, done)
        try:
            self.buffer[self.head_idx] = transition
        except Exception as e:
            print(f"Transition error: {e}")
            print(f"Transition details: {transition}")
        
        self.weights[self.head_idx] = self.max_weight

        self.head_idx = (self.head_idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample_minibatch(self, size=100):
        # Örnekleme için ağırlıkların normalize edilmesi
        set_weights = self.weights[:self.count] + self.delta
        probabilities = set_weights / sum(set_weights)
        self.indices = np.random.choice(range(self.count), size, p=probabilities, replace=False)
        return self.buffer[self.indices]

    def update_weights(self, prediction_errors):
        # Ağırlıkları güncelle
        max_error = max(prediction_errors)
        self.max_weight = max(self.max_weight, max_error)
        self.weights[self.indices] = prediction_errors

    def get_size(self):
        return self.count

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from server.fin_auditor_environment import FinAuditorEnvironment
from models import AuditorAction

class GymnasiumFinAuditorEnv(gym.Env):
    """Gymnasium wrapper for FinAuditorEnvironment."""
    def __init__(self):
        super().__init__()
        self.env = FinAuditorEnvironment()
        
        # Max anomalies to handle in a single step
        self.max_anomalies = 100
        
        # Action: List of decisions (0=Pass, 1=Investigate, 2=Flag) for anomalies.
        self.action_space = spaces.MultiDiscrete([3] * self.max_anomalies)
        
        # Observation: [time_elapsed, price_delta, missing_freq, risk_score]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.max_anomalies, 4), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        return self._process_obs(obs.features), {}

    def step(self, action_array):
        
        # Action array size must match max_anomalies
        # Convert to list for the environment
        action = AuditorAction(decisions=action_array.tolist())
        
        obs = self.env.step(action)
        
        # In this env, done is handleed via episode length or wrap-around
        # For training stability, we'll keep it simple
        done = getattr(obs, 'done', False)
        reward = getattr(obs, 'reward', 0.0)
        
        return self._process_obs(obs.features), float(reward), done, False, {}

    def _process_obs(self, features):
        if not features:
            return np.zeros((self.max_anomalies, 4), dtype=np.float32)
            
        feat_arr = np.array(features, dtype=np.float32)
        target = np.zeros((self.max_anomalies, 4), dtype=np.float32)
        
        n = min(len(feat_arr), self.max_anomalies)
        if n > 0:
            target[:n] = feat_arr[:n]
            
        return target

def main():
    # Force CPU for MlpPolicy as per SB3 recommendation for faster training without CNNs
    device = "cpu"
    
    print(f"Initializing wrapped environment on {device}...")
    def make_env():
        return GymnasiumFinAuditorEnv()
        
    vec_env = DummyVecEnv([make_env])
    
    # Initialize PPO agent
    # We use n_steps=512 to force more frequent synchronization and avoid deadlocks
    model = PPO("MlpPolicy", vec_env, verbose=1, device=device, batch_size=256, n_steps=512)
    
    # Add CheckpointCallback to save progress every 5,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=5000, 
        save_path="./logs/", 
        name_prefix="rl_model"
    )
    
    print("Starting training for 100,000 timesteps...")
    try:
        model.learn(
            total_timesteps=100000, 
            log_interval=1,
            callback=checkpoint_callback
        )
        
        # Save the model
        model.save("ppo_fin_auditor")
        print("Training complete. Model saved as ppo_fin_auditor.zip")
    except Exception as e:
        print(f"Training interrupted or failed: {e}")

if __name__ == "__main__":
    main()

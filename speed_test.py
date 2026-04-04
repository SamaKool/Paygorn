import time
import numpy as np
from train import GymnasiumFinAuditorEnv

env = GymnasiumFinAuditorEnv()
obs, _ = env.reset()

print("Environment initialized. Starting speed test...")
start_time = time.time()
steps = 1000
for i in range(steps):
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    if i % 100 == 0:
        print(f"Step {i}...")
end_time = time.time()

fps = steps / (end_time - start_time)
print(f"FPS: {fps:.2f}")

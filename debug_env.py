import time
import os
print("1. Importing modules...")
from server.fin_auditor_environment import FinAuditorEnvironment
try:
    print("2. Initializing C++ Engine...")
    env = FinAuditorEnvironment()
    print("3. Attempting env.reset()...")
    obs = env.reset()
    print("4. Reset successful. Obs shape:", len(obs.features))
    print("5. Attempting one env.step()...")
    # Simulate a dummy action
    from models import AuditorAction
    action = AuditorAction(decisions=[0]*100)
    obs = env.step(action)
    print("6. Step successful. Reward:", obs.reward)
    print("✅ ENVIRONMENT IS HEALTHY. The hang is inside Stable Baselines3.")
except Exception as e:
    print(f"❌ FAILED at stage {os.getpid()}: {e}")
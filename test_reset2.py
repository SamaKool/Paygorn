import sys
import time

print("Loading dependencies...")
from server.fin_auditor_environment import FinAuditorEnvironment

print("Dependencies loaded, initializing env...")
try:
    env = FinAuditorEnvironment()
    print("Env initialized.")
except Exception as e:
    print(f"Error during init: {e}")
    sys.exit(1)

print("Running reset()...")
try:
    obs = env.reset()
    print(f"Reset done, Shape Verification: {len(obs.features)} active trades")
except Exception as e:
    print(f"Error during reset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Running step()...")
try:
    env.step(None)
    print("Step execution successful")
except Exception as e:
    print(f"Error during step: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

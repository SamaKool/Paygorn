from server.fin_auditor_environment import FinAuditorEnvironment
try:
    print("Initializing environment...")
    env = FinAuditorEnvironment()
    print("Resetting environment...")
    obs = env.reset()
    print(f'Shape Verification: {len(obs.features)} active trades')
    print("Stepping environment...")
    env.step(None)
    print('Step execution successful')
except Exception as e:
    import traceback
    traceback.print_exc()

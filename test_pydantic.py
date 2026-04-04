from models import AuditorObservation
features = [[1.0, 2.0, 3.0, 4.0]] * 10000
print("Creating observation...")
try:
    obs = AuditorObservation(features=features)
    print("Done!")
except Exception as e:
    print(f"Error: {e}")

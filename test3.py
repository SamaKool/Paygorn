import time
from server.fin_auditor_environment import FinAuditorEnvironment
print("Init")
env = FinAuditorEnvironment()
print("Tick")
env.engine.tick(time.time_ns())
print("Ingest")
env._ingest_data_chunk()
print("Get Matrix")
mat = env.engine.get_observation_matrix()
print("ToList")
features = mat.tolist()
print("Len", len(features))
from server.fin_auditor_environment import FinAuditorObservation
print("Observation")
obs = FinAuditorObservation(
    features=features,
    message="Fin Auditor engine ready.",
)
print("Done")

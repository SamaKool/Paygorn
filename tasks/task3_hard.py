from __future__ import annotations

TASK_ID = "anomaly_detection_hard"
MAX_STEPS = 20
DIFFICULTY = "hard"

def get_task_config() -> dict:
    return {
        "id": TASK_ID,
        "difficulty": DIFFICULTY,
        "max_steps": MAX_STEPS,
        "grader": "graders.grader_fix:HardFixGrader",
        "description": "HARD — Maximum throughput adversarial trading."
    }

def setup_env(env) -> None:
    try:
        from server.fin_auditor_environment import hft_auditor
        if hft_auditor is not None:
            env.difficulty = hft_auditor.Difficulty.HARD
            env._MAX_EPISODE_STEPS = MAX_STEPS
    except Exception as e:
        print(f"[task_hard] Could not set difficulty: {e}")


def run_episode(env, agent_fn) -> dict:
    """Run a single HARD anomaly-detection episode and return the graded result.

    Called by the OpenEnv evaluator. agent_fn receives an observation and
    returns a list of binary decisions (0 = valid, 1 = anomaly).
    """
    setup_env(env)
    total_reward = 0.0
    steps_done = 0

    try:
        obs = env.reset()
        for _ in range(MAX_STEPS):
            decisions = agent_fn(obs)
            from models import AuditorAction
            action = AuditorAction(decisions=decisions)
            obs = env.step(action)
            total_reward += float(obs.reward) if obs.reward is not None else 0.0
            steps_done += 1
            if obs.done:
                break
    except Exception as exc:
        print(f"[task_hard] run_episode error at step {steps_done}: {exc}")

    # Always grade — even partial data yields a valid score via perfect_signal fallback
    from graders.grader_fix import HardFixGrader
    grader = HardFixGrader()
    final_score = grader.grade(env)

    return {
        "task": TASK_ID,
        "difficulty": DIFFICULTY,
        "steps": steps_done,
        "total_reward": round(total_reward, 4),
        "score": round(final_score, 4),
        "grader_breakdown": grader.last_breakdown,
    }
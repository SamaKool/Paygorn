#!/usr/bin/env python3
import os
import sys
import json
import yaml
import unittest
from unittest.mock import MagicMock, patch
import io
from contextlib import redirect_stdout

# Set dummy env vars BEFORE importing inference.py to avoid KeyError
os.environ["API_BASE_URL"] = "http://localhost:8000"
os.environ["MODEL_NAME"] = "test-model"
os.environ["HF_TOKEN"] = "dummy-token"
os.environ["MAX_STEPS"] = "2"
os.environ["TASK_ID"] = "anomaly_detection_easy"

# Add current directory to path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import inference
from server.fin_auditor_environment import FinAuditorEnvironment
from models import AuditorAction

class FinalIntegrityCheck(unittest.TestCase):

    def test_1_llm_parser_robustness(self):
        """Test the regex and JSON fallback logic in inference.py"""
        print("\n[TEST 1] LLM Parser Robustness...")
        
        # Test Case A: Markdown wrapped JSON
        dirty_json = "Here is the result:\n```json\n{\"decisions\": [0, 1, 2]}\n```\nHope this helps!"
        res = inference._parse_llm_decisions(dirty_json, 3)
        self.assertEqual(res, [0, 1, 2], "Failed to parse markdown-wrapped JSON")

        # Test Case B: Extra text before JSON
        extra_text = "The decisions are as follows: {\"decisions\": [1, 2]}"
        res = inference._parse_llm_decisions(extra_text, 2)
        self.assertEqual(res, [1, 2], "Failed to parse JSON with leading text")

        # Test Case C: Malformed JSON -> should trigger 'Flag All' (2) fallback
        malformed = "{\"decisions\": [0, 1, " # Missing closing bracket
        res = inference._parse_llm_decisions(malformed, 4)
        self.assertEqual(res, [2, 2, 2, 2], "Failed to trigger fallback on malformed JSON")
        
        # Test Case D: Correct length normalization
        wrong_len = "{\"decisions\": [1]}"
        res = inference._parse_llm_decisions(wrong_len, 3)
        self.assertEqual(len(res), 3, "Failed to normalize decision list length")
        self.assertEqual(res, [1, 2, 2], "Failed to pad short decision list with 2s")

        print("✓ LLM Parser logic is robust.")

    def test_2_spec_matching(self):
        """Verify openenv.yaml matches our deployment and task requirements"""
        print("\n[TEST 2] Spec Matching (openenv.yaml)...")
        with open("openenv.yaml", "r") as f:
            spec = yaml.safe_load(f)
        
        self.assertEqual(spec.get("app"), "server.app:app", "App entry point mismatch")
        self.assertEqual(spec.get("port"), 8000, "Port mismatch")
        
        tasks = spec.get("tasks", [])
        self.assertGreaterEqual(len(tasks), 3, "Missing required tasks (Easy, Medium, Hard)")
        
        task_ids = [t["id"] for t in tasks]
        self.assertIn("anomaly_detection_easy", task_ids)
        self.assertIn("anomaly_detection_medium", task_ids)
        self.assertIn("anomaly_detection_hard", task_ids)
        
        print(f"✓ Spec matches. Found {len(tasks)} tasks.")

    def test_3_reward_boundary(self):
        """Verify environment rewards stay strictly within [0.0, 1.0]"""
        print("\n[TEST 3] Reward Boundary Check...")
        env = FinAuditorEnvironment()
        obs = env.reset()
        
        # Simulate a step with some decisions
        action = AuditorAction(decisions=[2] * len(obs.features))
        new_obs = env.step(action)
        
        reward = new_obs.reward
        self.assertIsNotNone(reward)
        self.assertGreaterEqual(reward, 0.0, f"Reward {reward} < 0.0")
        self.assertLessEqual(reward, 1.0, f"Reward {reward} > 1.0")
        
        print(f"✓ Reward boundary is safe: {reward}")

    def test_4_integration_dry_run(self):
        """Run a 2-step inference using a mocked OpenAI client"""
        print("\n[TEST 4] Integration Dry Run...")
        
        # Mock the OpenAI client response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({"decisions": [2] * 200}) # provide plenty
        
        with patch("inference._client.chat.completions.create", return_value=mock_response):
            f = io.StringIO()
            with redirect_stdout(f):
                inference.run_inference()
            
            output = f.getvalue()
            
            # Verify structured logs appear
            self.assertIn("[START]", output)
            self.assertIn("[STEP]  step=1", output)
            self.assertIn("[STEP]  step=2", output)
            self.assertIn("[END]", output)
            
            # Check if rewards were logs
            self.assertIn("reward=", output)
            self.assertIn("cumulative_reward=", output)

        print("✓ Integration dry run successful. Logs are correctly formatted.")

if __name__ == "__main__":
    unittest.main(verbosity=1)

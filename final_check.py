#!/usr/bin/env python3
import os
import sys
import re
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
        dirty_json = "Here is the result:\n```json\n{\"decisions\": [0, 1, 1]}\n```\nHope this helps!"
        res = inference._parse_llm_decisions(dirty_json, 3)
        self.assertEqual(res, [0, 1, 1], "Failed to parse markdown-wrapped JSON")

        # Test Case B: Extra text before JSON
        extra_text = "The decisions are as follows: {\"decisions\": [1, 1]}"
        res = inference._parse_llm_decisions(extra_text, 2)
        self.assertEqual(res, [1, 1], "Failed to parse JSON with leading text")

        # Test Case C: Malformed JSON -> should trigger 'Flag All' (1) fallback
        malformed = "{\"decisions\": [0, 1, " # Missing closing bracket
        res = inference._parse_llm_decisions(malformed, 4)
        self.assertEqual(res, [1, 1, 1, 1], "Failed to trigger fallback on malformed JSON")
        
        # Test Case D: Correct length normalization
        wrong_len = "{\"decisions\": [1]}"
        res = inference._parse_llm_decisions(wrong_len, 3)
        self.assertEqual(len(res), 3, "Failed to normalize decision list length")
        self.assertEqual(res, [1, 1, 1], "Failed to pad short decision list with 1s")

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
        """Verify environment rewards stay strictly within (0.0, 1.0)"""
        print("\n[TEST 3] Reward Boundary Check...")
        env = FinAuditorEnvironment()
        obs = env.reset()
        
        # Reset should return features now (not empty)
        self.assertGreater(len(obs.features), 0, "Reset should return features for step 1")
        
        # Simulate a step with some decisions
        action = AuditorAction(decisions=[1] * len(obs.features))
        new_obs = env.step(action)

        reward = new_obs.reward
        self.assertIsNotNone(reward)
        self.assertGreater(reward, 0.0, f"Reward {reward} must be > 0.0 (not exact boundary)")
        self.assertLess(reward, 1.0, f"Reward {reward} must be < 1.0 (not exact boundary)")
        
        print(f"✓ Reward boundary is safe: {reward}")

    def test_4_reward_varies_by_action(self):
        """Verify rewards differ between optimal and random agents"""
        print("\n[TEST 4] Reward Variation Check...")
        
        # Run with all-1 decisions (flag everything)
        env1 = FinAuditorEnvironment()
        obs1 = env1.reset()
        action1 = AuditorAction(decisions=[1] * len(obs1.features))
        result1 = env1.step(action1)
        reward1 = result1.reward

        # Run with all-0 decisions (pass everything)
        env2 = FinAuditorEnvironment()
        obs2 = env2.reset()
        action2 = AuditorAction(decisions=[0] * len(obs2.features))
        result2 = env2.step(action2)
        reward2 = result2.reward

        print(f"  All-flag reward: {reward1:.4f}")
        print(f"  All-pass reward: {reward2:.4f}")

        # In EASY mode (100% anomalies), flagging everything should score higher
        self.assertNotEqual(reward1, reward2, "Rewards must differ between flag-all and pass-all")
        
        print("✓ Rewards vary based on agent decisions.")

    def test_5_stdout_format(self):
        """Run a 2-step inference and verify stdout matches hackathon regex"""
        print("\n[TEST 5] Stdout Format Compliance...")
        
        # Mock the OpenAI client response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({"reasoning": "test", "decisions": [1] * 200})
        
        with patch("inference._client.chat.completions.create", return_value=mock_response):
            f = io.StringIO()
            with redirect_stdout(f):
                inference.run_inference()
            
            output = f.getvalue()
            lines = [l for l in output.strip().split("\n") if l.strip()]
            
            # Verify START tag format
            start_line = lines[0]
            start_match = re.match(r'^\[START\] task=\S+ env=\S+ model=\S+$', start_line)
            self.assertIsNotNone(start_match, f"START line doesn't match regex: {start_line}")
            
            # Verify STEP tag format
            step_lines = [l for l in lines if l.startswith("[STEP]")]
            self.assertTrue(len(step_lines) >= 1, "No STEP lines found")
            for sl in step_lines:
                step_match = re.match(
                    r'^\[STEP\] step=\d+ action=\S+ reward=\d+\.\d{2} done=(true|false) error=\S+$',
                    sl
                )
                self.assertIsNotNone(step_match, f"STEP line doesn't match regex: {sl}")
            
            # Verify END tag format
            end_line = lines[-1]
            end_match = re.match(
                r'^\[END\] success=(true|false) steps=\d+ rewards=[\d.,]+$',
                end_line
            )
            self.assertIsNotNone(end_match, f"END line doesn't match regex: {end_line}")
            
            # Verify NO JSON on stdout
            self.assertNotIn("{", output, "Stdout must not contain JSON braces")
            self.assertNotIn("}", output, "Stdout must not contain JSON braces")

        print("✓ Stdout format is compliant with hackathon regex rules.")

if __name__ == "__main__":
    unittest.main(verbosity=1)
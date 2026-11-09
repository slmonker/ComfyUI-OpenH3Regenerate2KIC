import importlib.util
import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_nodes():
    spec = importlib.util.spec_from_file_location(
        "open_h3_ic_nodes", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["open_h3_ic_nodes"] = module
    spec.loader.exec_module(module)
    return module.NODE_CLASS_MAPPINGS


class NodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes = load_nodes()

    def test_sigmas(self):
        sigmas, report = self.nodes["OpenH3ICSigmas"]().build("up3", "")
        self.assertEqual(sigmas.tolist(), [0.8349419832229614, 0.6968520283699036, 0.46875301003456116, 0.0])
        self.assertEqual(json.loads(report)["steps"], 3)

    def test_draft_audio_duration_uses_source_rate(self):
        audio = {"waveform": torch.randn(1, 1, 48000 * 3), "sample_rate": 48000}
        frames, out_audio, report = self.nodes["OpenH3ICDraft"]().prepare(
            torch.rand(150, 36, 64, 3), 30.0, 64, 64, 124, 24, "cover", "trim_pad", audio
        )
        self.assertEqual(tuple(frames.shape), (124, 64, 64, 3))
        self.assertEqual(out_audio["waveform"].shape, (1, 2, 248000))
        self.assertAlmostEqual(out_audio["waveform"].shape[-1] / out_audio["sample_rate"], 124 / 24, places=6)
        self.assertEqual(json.loads(report)["real_frames_at_24fps"], 120)

    def test_invalid_frame_count(self):
        with self.assertRaises(ValueError):
            self.nodes["OpenH3ICDraft"]().prepare(
                torch.rand(150, 36, 64, 3), 30.0, 64, 64, 125, 24, "cover", "trim_pad", None
            )


if __name__ == "__main__":
    unittest.main()

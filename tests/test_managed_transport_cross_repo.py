from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest

from backend.continuum import resolved_chunk_prompt, serialize_timeline


def _load_managed_transport(source_root: Path):
    package_name = "_h3_continuum_managed_contract"
    package = types.ModuleType(package_name)
    package.__path__ = [str(source_root)]
    sys.modules[package_name] = package

    v2_name = f"{package_name}.v2"
    v2_package = types.ModuleType(v2_name)
    v2_package.__path__ = [str(source_root / "v2")]
    sys.modules[v2_name] = v2_package

    modules = {}
    for short_name, path in (
        ("constants", source_root / "constants.py"),
        ("version", source_root / "version.py"),
        ("v2.prompts", source_root / "v2" / "prompts.py"),
        ("v2.physical_prompts", source_root / "v2" / "physical_prompts.py"),
        ("v2.prompt_transport", source_root / "v2" / "prompt_transport.py"),
    ):
        module_name = f"{package_name}.{short_name}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        modules[short_name] = module
    return modules["v2.prompts"], modules["v2.prompt_transport"]


def _sidecar(text: str, *, chunks: int, chunk_seconds: str, fmt: str = "timeline") -> str:
    prompt_document = {"schema_version": 1, "format": fmt}
    if fmt == "timeline":
        prompt_document.update({
            "routing": "logical_chunks",
            "geometry": {
                "chunks": chunks,
                "chunk_seconds": chunk_seconds,
            },
        })
    return json.dumps({
        "magic": "DSM_H3_PROMPT_SOURCE",
        "schema_version": 1,
        "text": text,
        "prompt_document": prompt_document,
        "raw_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "library_revision": 9,
        "binding": {
            "manager_node": "3",
            "text_node": "1",
            "impact_node": "2",
            "role": "positive",
            "slot": "default",
        },
        "queue_contract": "ordered-impact-v1",
    })


@unittest.skipUnless(
    os.environ.get("H3_CONTINUUM_MANAGED_SOURCE"),
    "H3_CONTINUUM_MANAGED_SOURCE is required for managed transport cross-repo CI",
)
class ManagedContinuumTransportContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_root = Path(os.environ["H3_CONTINUUM_MANAGED_SOURCE"]).resolve()
        cls.prompts, cls.transport = _load_managed_transport(cls.source_root)

    def test_reviewed_provider_advertises_current_execution_geometry(self):
        provider = self.transport.PROMPT_TRANSPORT_PROVIDER_V1
        self.assertEqual(provider["provider_version"], 1)
        self.assertEqual(provider["chunks"], {"min": 1, "max": 16})
        self.assertEqual(provider["chunk_seconds"], {"min": 4.0, "max": 15.0})
        self.assertIn("logical_chunks", provider["timeline_routings"])

    def test_writer_document_enters_reviewed_continuum_as_verified_logical_timeline(self):
        preamble = "Global identity and environment persist."
        bodies = ["One.", "Two."]
        script = serialize_timeline(preamble, bodies, 7)
        plan = self.prompts.build_sampler_prompt_plan(
            prompt_mode="Auto",
            prompt_script="legacy",
            sequence_prompt=script,
            prompt_plan=None,
            chunks=2,
            chunk_seconds=7.0,
            managed_prompt_source_json=_sidecar(
                script,
                chunks=2,
                chunk_seconds="7",
            ),
        )

        self.assertEqual(plan["mode"], self.prompts.PROMPT_MODE_TIMELINE)
        self.assertEqual(
            plan["prompts"],
            [resolved_chunk_prompt(preamble, body) for body in bodies],
        )
        receipt = plan["managed_prompt_transport"]
        self.assertEqual(receipt["status"], "verified_sequence")
        self.assertTrue(receipt["geometry_match"])
        self.assertTrue(receipt["skeleton_match"])
        self.assertTrue(receipt["sequence_verified"])

    def test_explicit_fixed_document_remains_opaque_with_header_like_text(self):
        text = "[0-5s]\nLiteral text that must remain Fixed."
        plan = self.prompts.build_sampler_prompt_plan(
            prompt_mode="Auto",
            prompt_script="legacy",
            sequence_prompt=text,
            prompt_plan=None,
            chunks=2,
            chunk_seconds=5.0,
            managed_prompt_source_json=_sidecar(
                text,
                chunks=2,
                chunk_seconds="5",
                fmt="fixed",
            ),
        )

        self.assertEqual(plan["mode"], self.prompts.PROMPT_MODE_FIXED)
        self.assertEqual(plan["prompts"], [text, text])
        self.assertEqual(
            plan["managed_prompt_transport"]["status"],
            "document_applied",
        )


if __name__ == "__main__":
    unittest.main()

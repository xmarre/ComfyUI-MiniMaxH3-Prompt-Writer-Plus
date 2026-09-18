from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from backend.continuum import ContinuumError, parse_sequence_plan
from backend.continuum_schema import continuum_plan_json_schema, planner_response_metadata
from backend.models.contract import ModelError
from backend.models.api_provider_backend import (
    ApiConnection,
    ApiProviderBackend,
    _ApiChatHandler,
    _lm_studio_continuum_response_format,
)


def _settings(chunks: int = 2) -> dict[str, object]:
    return {
        "schema_version": 2,
        "chunks": chunks,
        "chunk_seconds": 5.0,
        "total_seconds": chunks * 5.0,
    }


def _assembled(stage: str | None, *, chunks: int = 2) -> dict[str, object]:
    request_input: dict[str, object] = {
        "generation_target": "continuum",
        "continuum": _settings(chunks),
    }
    if stage is not None:
        request_input["continuum_stage"] = stage
    return {"input": request_input}


def _plan(chunks: int = 2) -> dict[str, object]:
    return {
        "schema_version": 2,
        "global": {
            "sequence_preamble": "The same subject, location, wardrobe, lighting, and camera language persist.",
            "continuity_anchors": "Maintain the established spatial relationship.",
            "persistent_constraints": "Do not change wardrobe.",
            "subject_anchors": [],
        },
        "chunks": [
            {
                "continuity": "initial" if index == 1 else "continuous",
                "transition": "",
                "start_state": f"Start state {index}.",
                "action": f"Action {index}.",
                "end_state": f"End state {index}.",
            }
            for index in range(1, chunks + 1)
        ],
    }


class ContinuumStructuredOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = ApiProviderBackend()

    @staticmethod
    def connection(*, profile: str = "lm_studio") -> ApiConnection:
        return ApiConnection(
            id="test-connection",
            preset="custom",
            base_url="http://127.0.0.1:1234/v1",
            api_key="",
            custom_images=False,
            custom_context_tokens=15360,
            compatibility_profile=profile,
        )

    def test_plan_and_repair_use_the_same_lm_studio_json_schema(self) -> None:
        connection = self.connection()
        plan_format = _lm_studio_continuum_response_format(connection, _assembled("plan"))
        repair_format = _lm_studio_continuum_response_format(connection, _assembled("plan_repair"))

        self.assertEqual(plan_format, repair_format)
        self.assertEqual(plan_format["type"], "json_schema")
        self.assertTrue(plan_format["json_schema"]["strict"])
        schema = plan_format["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["schema_version"]["const"], 2)
        self.assertEqual(schema["properties"]["chunks"]["minItems"], 2)
        self.assertEqual(schema["properties"]["chunks"]["maxItems"], 2)
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(schema["properties"]["global"]["additionalProperties"])
        self.assertFalse(schema["properties"]["chunks"]["items"]["additionalProperties"])

    def test_schema_tracks_requested_chunk_count_without_encoding_position_semantics(self) -> None:
        schema = continuum_plan_json_schema(_settings(4))
        chunks = schema["properties"]["chunks"]
        self.assertEqual(chunks["minItems"], 4)
        self.assertEqual(chunks["maxItems"], 4)
        self.assertEqual(
            chunks["items"]["properties"]["continuity"]["enum"],
            ["initial", "continuous", "intentional_break"],
        )

    def test_non_planner_and_generic_custom_requests_do_not_get_lm_studio_schema(self) -> None:
        lm_studio = self.connection()
        for stage in (None, "chunk", "refine_chunk"):
            with self.subTest(stage=stage):
                self.assertIsNone(_lm_studio_continuum_response_format(lm_studio, _assembled(stage)))

        single = {"input": {"generation_target": "single"}}
        self.assertIsNone(_lm_studio_continuum_response_format(lm_studio, single))
        self.assertIsNone(
            _lm_studio_continuum_response_format(
                self.connection(profile="generic"),
                _assembled("plan"),
            )
        )

    def test_lm_studio_planner_preflight_uses_non_thinking_8192_budget(self) -> None:
        model = self.backend._model_info(self.connection(), "qwen-model")
        plan = self.backend.preflight(
            model,
            _assembled("plan"),
            context_profile="auto",
            kv_cache="auto",
            thinking=True,
        )
        self.assertFalse(plan["thinking"])
        self.assertEqual(plan["context_tokens"], 15360)
        self.assertEqual(plan["max_output_tokens"], 8192)

        repair = self.backend.preflight(
            model,
            _assembled("plan_repair"),
            context_profile="auto",
            kv_cache="auto",
            thinking=True,
        )
        self.assertFalse(repair["thinking"])
        self.assertEqual(repair["max_output_tokens"], 8192)

    def test_non_planner_lm_studio_and_generic_planner_keep_thinking_validation(self) -> None:
        lm_studio_model = self.backend._model_info(self.connection(), "qwen-model")
        with self.assertRaises(ModelError) as normal:
            self.backend.preflight(
                lm_studio_model,
                {"input": {"generation_target": "single"}, "messages": [], "media_inputs": []},
                context_profile="auto",
                kv_cache="auto",
                thinking=True,
            )
        self.assertEqual(normal.exception.code, "API_THINKING_UNAVAILABLE")

        generic_model = self.backend._model_info(self.connection(profile="generic"), "qwen-model")
        with self.assertRaises(ModelError) as generic:
            self.backend.preflight(
                generic_model,
                _assembled("plan"),
                context_profile="auto",
                kv_cache="auto",
                thinking=True,
            )
        self.assertEqual(generic.exception.code, "API_THINKING_UNAVAILABLE")

    def test_lm_studio_structured_output_rejection_is_distinct_and_content_free(self) -> None:
        class Response:
            status = 400

            @staticmethod
            def getheaders():
                return [("X-Request-ID", "structured-request")]

            @staticmethod
            def read():
                return json.dumps({
                    "error": {
                        "message": "unsupported response_format",
                        "type": "invalid_request_error",
                    }
                }).encode("utf-8")

        class HttpConnection:
            def request(self, *_args, **_kwargs):
                return None

            @staticmethod
            def getresponse():
                return Response()

            @staticmethod
            def close():
                return None

        response_format = _lm_studio_continuum_response_format(self.connection(), _assembled("plan"))
        payload = {
            "model": "qwen-model",
            "messages": [{"role": "user", "content": "private creative brief"}],
            "stream": True,
            "response_format": response_format,
        }
        with (
            patch.object(self.backend, "_http_connection", return_value=HttpConnection()),
            self.assertRaises(ModelError) as raised,
        ):
            self.backend._request_chat_completion_stream(self.connection(), payload)
        self.assertEqual(raised.exception.code, "API_STRUCTURED_OUTPUT_REJECTED")
        serialized = json.dumps(raised.exception.details)
        self.assertNotIn("private creative brief", serialized)
        self.assertNotIn("properties", serialized)
        self.assertIn("structured-request", serialized)

    def test_lm_studio_structured_planner_promotes_valid_json_from_reasoning_stream(self) -> None:
        response_format = _lm_studio_continuum_response_format(self.connection(), _assembled("plan"))

        class Response:
            status = 200

            @staticmethod
            def getheaders():
                return []

            def __init__(self, reasoning_field: str, *, content: str = "") -> None:
                self.lines = iter([
                    ("data: " + json.dumps({
                        "choices": [{
                            "delta": {
                                reasoning_field: '{"schema_version":2}',
                                "content": content,
                            },
                            "finish_reason": "stop",
                        }]
                    }) + "\n").encode("utf-8"),
                    b"data: [DONE]\n",
                ])

            def readline(self):
                return next(self.lines, b"")

        class HttpConnection:
            def __init__(self, response):
                self.response = response

            def request(self, *_args, **_kwargs):
                return None

            def getresponse(self):
                return self.response

            def close(self):
                return None

        for reasoning_field in ("reasoning_content", "reasoning"):
            with self.subTest(reasoning_field=reasoning_field):
                http = HttpConnection(Response(reasoning_field))
                payload = {
                    "model": "qwen-model",
                    "messages": [],
                    "stream": True,
                    "response_format": response_format,
                }
                with patch.object(self.backend, "_http_connection", return_value=http):
                    result = self.backend._request_chat_completion_stream(self.connection(), payload)
                self.assertEqual(result["choices"][0]["message"]["content"], '{"schema_version":2}')

    def test_lm_studio_reasoning_fallback_is_scoped_and_never_overrides_content(self) -> None:
        response_format = _lm_studio_continuum_response_format(self.connection(), _assembled("plan"))

        class Response:
            status = 200

            @staticmethod
            def getheaders():
                return []

            def __init__(self, *, content: str, reasoning: str) -> None:
                self.lines = iter([
                    ("data: " + json.dumps({
                        "choices": [{
                            "delta": {"content": content, "reasoning_content": reasoning},
                            "finish_reason": "stop",
                        }]
                    }) + "\n").encode("utf-8"),
                    b"data: [DONE]\n",
                ])

            def readline(self):
                return next(self.lines, b"")

        class HttpConnection:
            def __init__(self, response):
                self.response = response

            def request(self, *_args, **_kwargs):
                return None

            def getresponse(self):
                return self.response

            def close(self):
                return None

        cases = [
            (self.connection(), "plain-content", '{"schema_version":2}', response_format, "plain-content"),
            (self.connection(), "", "reasoning prose", response_format, ""),
            (self.connection(profile="generic"), "", '{"schema_version":2}', response_format, ""),
        ]
        for connection, content, reasoning, fmt, expected in cases:
            with self.subTest(profile=connection.compatibility_profile, content=content, reasoning=reasoning):
                http = HttpConnection(Response(content=content, reasoning=reasoning))
                payload = {
                    "model": "qwen-model",
                    "messages": [],
                    "stream": True,
                    "response_format": fmt,
                }
                with patch.object(self.backend, "_http_connection", return_value=http):
                    result = self.backend._request_chat_completion_stream(connection, payload)
                self.assertEqual(result["choices"][0]["message"]["content"], expected)

    def test_handler_sends_schema_and_disables_lm_studio_reasoning_for_constrained_planner(self) -> None:
        connection = self.connection()
        response_format = _lm_studio_continuum_response_format(connection, _assembled("plan"))
        captured: dict[str, object] = {}

        def request(_connection, payload):
            captured["payload"] = payload
            return {
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {},
            }

        with patch.object(self.backend, "_request_chat_completion_stream", side_effect=request):
            _ApiChatHandler(self.backend, connection, "qwen-model")(
                messages=[{"role": "user", "content": "brief"}],
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                max_tokens=8192,
                seed=None,
                thinking=True,
                response_format=response_format,
            )

        payload = captured["payload"]
        self.assertEqual(payload["response_format"], response_format)
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertTrue(payload["stream"])

    def test_normal_lm_studio_generation_keeps_existing_reasoning_behavior(self) -> None:
        connection = self.connection()
        captured: dict[str, object] = {}

        def request(_connection, payload):
            captured["payload"] = payload
            return {
                "choices": [{"message": {"content": "plain prompt"}, "finish_reason": "stop"}],
                "usage": {},
            }

        with patch.object(self.backend, "_request_chat_completion_stream", side_effect=request):
            _ApiChatHandler(self.backend, connection, "qwen-model")(
                messages=[{"role": "user", "content": "brief"}],
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                max_tokens=1536,
                seed=None,
                thinking=True,
            )

        payload = captured["payload"]
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["reasoning_effort"], "low")

    def test_structural_schema_does_not_bypass_continuum_semantic_validation(self) -> None:
        invalid = _plan()
        invalid["chunks"][0]["continuity"] = "continuous"
        with self.assertRaises(ContinuumError) as raised:
            parse_sequence_plan(
                json.dumps(invalid),
                _settings(),
                expected_references=set(),
                persistent_references=set(),
                chunk_reference_scopes=[set(), set()],
            )
        self.assertEqual(raised.exception.code, "INVALID_CONTINUUM_PLAN_CONTINUITY")

    def test_planner_response_metadata_is_structural_and_does_not_expose_output(self) -> None:
        secret = "<think>private creative brief\n{broken"
        metadata = planner_response_metadata(
            {"prompt": secret, "primary_finish_reason": "stop"}
        )
        self.assertEqual(metadata["chars"], len(secret))
        self.assertEqual(metadata["finish_reason"], "stop")
        self.assertTrue(metadata["starts_with_think"])
        self.assertTrue(metadata["contains_object_open"])
        self.assertFalse(metadata["contains_object_close"])
        serialized = json.dumps(metadata)
        self.assertNotIn("private creative brief", serialized)
        self.assertNotIn("broken", serialized)


if __name__ == "__main__":
    unittest.main()

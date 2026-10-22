# API providers

API providers use one OpenAI-compatible Chat Completions backend with presets for Gemini, OpenAI, OpenRouter, and Custom endpoints.

![API provider settings](assets/v0.3/api-providers.png)

## Choose a preset

- **Gemini** uses Google's OpenAI-compatible endpoint and exposes Gemini Thinking levels.
- **OpenAI** uses the OpenAI API preset.
- **OpenRouter** sends requests through OpenRouter to an upstream model provider.
- **Custom** accepts a generic OpenAI-compatible URL, including local LM Studio.

Gemini was validated live. The shared Custom transport was validated live with LM Studio. OpenAI and OpenRouter contract tests cover request serialization, model listing, streaming, cancellation, errors, and secret handling, but no credentialed live smoke was run for those two services.

The API provider path is not tied to Gemma 4. You can choose another multimodal model when the provider accepts image inputs in a format Writer supports. A successful connection shows that Writer can reach the model, but it does not guarantee a good H3 prompt.

## Connect

1. Open **H3 Prompt Writer > Settings > API providers**.
2. Choose a preset.
3. Paste the API key for the current Writer session. Custom endpoints may be used without a key.
4. Optionally enter an exact Model ID before connecting.
5. Select **Connect & test**, then choose a vision-capable model from the provider list or enter its exact ID.

Model availability, pricing, quotas, rate limits, and input support belong to the provider and selected model. Check the policy and pricing links shown in Settings.

## Gemini Thinking

Gemini supports these Writer levels:

- Minimal
- Low
- Medium
- High

Gemini manages its reasoning and output budget. Higher levels can consume more tokens and take longer. The general Thinking switch on Generate is hidden for API providers so it does not conflict with the provider-specific setting.

## Custom and LM Studio

For LM Studio, use a base URL such as:

```text
http://localhost:1234/v1
```

Load a vision model in LM Studio, connect the Custom preset, and choose the model. Writer can read LM Studio's local capability metadata when the standard `/v1/models` response does not identify vision support.

For Ollama through the Custom OpenAI-compatible preset, use a base URL such as `http://192.168.1.25:11434/v1`. The `/v1` prefix is required for Ollama's OpenAI-compatible routes.

For another Custom endpoint, enable **Endpoint accepts image_url inputs** only when the server and model really accept OpenAI-style image content. You can also provide a known context size. Loopback and private LAN IP addresses may use HTTP; public remote endpoints must use HTTPS.

Custom is a transport contract, not a claim that every OpenAI-compatible server or model is supported.

## Keys and saved settings

The key is sent once to the local H3 backend and held only in the current Writer backend process's memory. It is not read from environment variables and is not written to browser storage, model settings, developer notes, or request content. The backend uses it only to authenticate provider requests. Disconnecting removes the in-memory connection. The browser may save the preset, base URL, model ID, Gemini Thinking level, and Custom capability settings.

## What leaves this computer

For a remote provider, Writer sends:

- your Creative Brief;
- H3 and system instructions;
- prepared images in the current mode's manifest;
- one derived contact sheet for each video in that manifest.

Writer does not upload original video bytes or audio bytes. Audio references remain textual entries in the request manifest, so describe their intended role in the brief.

The provider can retain or process requests according to its own policy. OpenRouter can forward the request to an upstream model provider with a separate policy. Review the provider's data policy before sending private media.

## Lifecycle and errors

API providers show **Cancel** for the current Writer request. Writer cannot unload or stop a remote service.

Authentication, billing, rate-limit, safety, and quota errors are returned by the provider. A response that reaches its length limit is rejected rather than displayed as a successful but truncated H3 prompt.

Comfy Cloud has not been validated for v0.3. The existence of an API provider does not establish that the extension, outbound networking, or session-key handling works there.

## Versioned scene observations

Timeline Studio integration can negotiate `visual_observations_v1` using
`GET /h3studio/scene-analysis/capabilities`. It reuses connected API vision models
and their existing transport. `POST /h3studio/scene-analysis` accepts version 1,
`api_provider` connection/model IDs, a `request_token`, `context_budget` (2048–49152),
and one to four inline PNG/JPEG evidence objects (`id`, `pts`, `data_url`). No
arbitrary filesystem or remote image URL is read. Images are limited to 4 MP and
4 MiB each, and conservatively estimated visual tokens must fit the selected model's context. Qwen3-VL uses approximately 32×32 pixels per compressed visual token; the 28-pixel budget is deliberately conservative rather than a model-exact estimator.

For local LM Studio connections detected through the Custom provider, this route requests strict `response_format: json_schema` output. The LM Studio-specific grammar uses **string-only** observation fields (empty string when unsupported), because some LM Studio/Outlines versions reject nullable `type` arrays or collapse unions. Empty fields are converted back to JSON null before the unchanged strict evidence/markup validation and persistence; external provider records retain their nullable schema. The actual user LM Studio Qwen-VL log from 2026-10-22 confirms that the request reached model generation, but its generated JSON was not included in the log; therefore the failing field cannot be identified from that log alone.

**Single-still contract correction:** A single frame cannot establish camera motion. The LM Studio schema now forces `camera_movement` to the empty string (and constrains evidence-id strings to the identities actually submitted). If a local model/runtime nevertheless emits nonempty motion text for a single image, the parser deterministically drops **only that unsupported field**, converting it to `null`; all other fields remain subject to strict evidence-identity, type, content, markup, subject-limit and confidence checks. A multi-image comparison retains valid measured/qualified motion descriptions. Invalid records return the rejected *field/category* without echoing model text or private evidence IDs, making otherwise opaque Studio failures actionable. Model output is not treated as factual truth: review observations before using descriptions to condition H3. Live successful Qwen3.8-27B interoperability remains to be verified. Some local models cannot honor constrained generation; unsupported structured-output errors must be handled by selecting a compatible model, rather than presenting unvalidated text as a successful observation. A single complete ```json fence and a complete leading `<think>...</think>` reasoning wrapper are tolerated; neither contributes to observations, and the remaining JSON must still pass strict schema and evidence validation. Unfinished reasoning wrappers are rejected.

The capabilities response marks each vision model as `remote` or local using the connected endpoint address. Timeline Studio confirms before sending extracted images to remote endpoints, including hosted Custom servers. The corresponding provider's privacy and retention terms apply. No full original video file is transmitted.

The response is a validated observation object, not an H3 prompt. It contains
version/evidence IDs, anonymous subjects, confidence, and nullable visual fields:
environment, shot type, framing, posture, gaze, gestures, expression, lighting,
props, continuity, acting and camera movement. Single-still motion, unexpected
fields, dialogue/reference markup, unbound evidence, malformed JSON and truncated
or unfinished streams are rejected. Pixel-grounding is not proved by schema
validation; the receiving editor must present observations for human review.

`POST /h3studio/scene-analysis/cancel` cancels the matching `request_token`,
including cancellation that arrives before request ownership is established.
This path shares Writer's active-request lease and media-mutation exclusions;
ordinary prompt generation/refinement is unchanged. ComfyUI must be idle before
interpretation; a queue/disconnect watcher cancels ongoing requests. Timeline's
normal queue handoff waits for cancellation completion. Other clients can race
the queue watcher. LM Studio remains provider-managed and is not unloaded by
ComfyUI or by cancelling an API request. Secrets remain in the existing in-memory
connection store and are not included in capability responses or project data.

The companion Continuum Plus scene workspace is PR #83. This API layer is
independent of the State Manager contribution stack; keep that stack's existing
order and apply this layer afterward when using both integrations.


### Visually grounded subject descriptors

The scene-observation vision instruction explicitly asks for **visible** complexion/skin tone, hairstyle, hair texture and color, facial hair, approximate age range, wardrobe and position where supported by the screenshot. A terse generic `person with short hair` is insufficient when distinguishing features are clearly visible. The model must not infer ethnic ancestry, nationality, religion or the identity of the actor from a still frame. Detailed subject observations remain reviewable evidence; Continuum's H3 composer adds only brief, user-approved shot/Picture cues, not full subject paragraphs or a replacement of the identity references.

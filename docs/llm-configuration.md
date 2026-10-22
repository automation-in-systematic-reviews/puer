# LLM configuration

This is the canonical reference for server-side model and reasoning configuration in PUER.
[Environment variables](environment-variables.md) remains canonical for API keys, safe secret setup, environment loading and precedence, restart behavior, and Docker Compose forwarding.
The source of truth for the defaults is `inference-api/app/resources/globals.py`.
The defaults below are code defaults at this revision, not a guarantee that the named models are available through a provider account.

## Prompt-based endpoint settings

`POST /data_extraction/title_abstract/` and `POST /study_screening/predict/` use the same screening configuration.

### OPENAI_STUDY_SCREENING_MODEL

- Type: string.
- Default: `gpt-5.6-terra`.
- Scope: Both prompt-based endpoints.
- Purpose: Selects the provider model used for title and abstract extraction and prediction.
- Constraints: The selected provider path must support the OpenAI Responses API call and the requested structured-output schema.

### OPENAI_STUDY_SCREENING_REASONING_EFFORT

- Type: one of `none`, `minimal`, `low`, `medium`, `high`, or `xhigh`.
- Default: `medium`.
- Scope: Both prompt-based endpoints.
- Purpose: Selects the reasoning effort sent to the screening provider.
- Constraints: The service validates this setting locally before the provider call and sends `reasoning={"effort": ...}`, including when the value is `none`.

Changing either screening variable changes both prompt-based endpoints after the API process is restarted.
The endpoint request and response contracts are documented in [API endpoints](api-endpoints.md#prompt-based-title-and-abstract-screening).

## Risk-of-bias endpoint settings

`POST /risk_of_bias/assess` has an independent model and reasoning configuration.

### OPENAI_MODEL

- Type: string.
- Default: `gpt-5.2`.
- Scope: The risk-of-bias provider calls made for `/risk_of_bias/assess`.
- Purpose: Selects the provider model used for PDF detail extraction and domain assessment.
- Constraints: The selected provider path must support the OpenAI Responses API and the requested structured-output schemas.

### OPENAI_REASONING_EFFORT

- Type: string.
- Default: `medium`.
- Scope: The risk-of-bias provider calls made for `/risk_of_bias/assess`.
- Purpose: Selects the reasoning effort for risk-of-bias provider calls.
- Constraints: The service omits the reasoning argument when the value is `none` and passes other configured values to the provider without local enum validation. Choose a value supported by the selected model and provider.

These settings do not fall back to, or override, the screening-specific variables.
The shared server-side credential is `OPENAI_API_KEY`; use [Environment variables](environment-variables.md) for its setup and handling.
The endpoint contract is documented in [API endpoints](api-endpoints.md#post-risk_of_biasassess).

## Server-owned settings

- Owner: Server configuration controls model, reasoning, prompt version, schema version, and provider credentials; no per-request overrides.
- Extraction fields: Title and optional abstract.
- Prediction fields: Title, optional abstract, review topic, criteria, and characteristics.
- Unknown fields: Rejected, including caller-supplied model, reasoning, or provenance fields.
- Provenance: Created by the server after provider output validation.

See [API endpoints](api-endpoints.md) for complete request and response schemas rather than duplicating them here.

## Structured output and provider compatibility

- API: OpenAI Responses API, `responses.parse`, Pydantic schema via `text_format`.
- Compatibility: Model and provider must support the API call, requested reasoning setting, and structured-output schema together.
- Fallback: No unstructured-JSON fallback or alternate provider adapter.
- Output: Parsed output matching the strict Pydantic schema; undeclared fields forbidden.
- Validation: Evidence, title-only behavior, conflicts, and decision invariants.
- Failure: Incompatible provider or SDK behavior can return `503 Service Unavailable`; see [operational errors](api-endpoints.md#operational-errors).

## Lazy initialization and configuration failures

- Configuration read: On import of `app/resources/globals.py`.
- Client creation: At the LLM provider boundary or during an explicit extraction diagnostic check, not API startup or prompt-service import.
- Prompt configuration errors (`503`): Missing key, blank or invalid screening setting, missing SDK support, client-construction failure, or missing `responses.parse` capability.
- Prompt provider errors (`503`): Transport, request, and parsed-output failures, using the documented error categories.
- Contract diagnostics: Server warnings identify the stage and a bounded failure phase, such as `provider_parse`, `parsed_output`, `schema_validation`, or `evidence_validation`.
  For `provider_parse`, one warning also reports only the first validation error's allowlisted `error_type`, schema `field_path`, and `invariant` labels.
  Array indices become `*`; unknown paths or error types become `other`, and a non-validation `ValueError` has `field_path=unavailable`.
  They omit exception messages, tracebacks, error input and context, arbitrary location tokens, document text, provider bodies, and credentials; the public error response stays unchanged.
  SDK parsing executes Pydantic model validators as well as JSON and field validation, so `provider_parse error_type=value_error field_path=response` can indicate a prediction decision-invariant failure before evidence grounding.
  For a root `value_error`, exact matches to the four constant validator messages identify `invariant=excluded_low_confidence`, `excluded_without_evidenced_not_met`, `unclear_without_included_low`, or `material_conflict_without_included_low`.
  The latter two mean that unclear criteria or a material conflict did not yield included with low confidence; the former two mean excluded with low confidence or excluded without an evidenced `not_met` criterion.
  Unknown root messages and non-validation `ValueError`s yield `invariant=unknown`; other error types or field paths yield `invariant=not_applicable`.
  No substring matching is used, and message text is never emitted; SDK/Pydantic message-format changes safely yield `unknown` rather than changing validation.
  Only the first failure is identified, so simultaneous violations can remain hidden, and these labels cannot establish the cause of a previously uninstrumented failure.
- Risk-of-bias errors (`503`): Missing credentials or SDK/provider failures; client creation occurs only when `/risk_of_bias/assess` is invoked.

## Diagnose extraction configuration

Use authenticated [`GET /check/extraction`](api-endpoints.md#get-checkextraction) to inspect local provider readiness without making a paid request.
Unlike `/check/auth`, this checks the API process's provider configuration and SDK interface, not just the notebook's application key.
It returns only safe flags and diagnostic codes, not credential values or raw exception messages.

- `missing_provider_key`: configure the server-side `OPENAI_API_KEY`; it is separate from `WCRF_API_KEY`.
- `missing_model`: set a nonblank screening model identifier.
- `invalid_reasoning_effort`: choose an allowed screening reasoning value listed above.
- `client_initialization_failed`: check the installed OpenAI SDK and client setup; the response includes the underlying exception class name when available.
- `structured_parse_unavailable`: use an SDK with callable `responses.parse` support.
- `client_cleanup_failed`: the temporary client's cleanup failed; check the SDK and local client lifecycle.

Restart the API after changing configuration, then repeat the diagnostic check.
A successful local check does not establish remote credential validity, model availability, quota, or provider compatibility with a real request.

## Practical nonsecret overrides

Set nonsecret overrides in the shell before starting the API, and restart the process after changing them.
Use a provider-supported model identifier and reasoning value rather than assuming that the source default is enabled for your account.

```bash
export OPENAI_STUDY_SCREENING_MODEL=provider-model-id
export OPENAI_STUDY_SCREENING_REASONING_EFFORT=high
export OPENAI_MODEL=provider-model-id
export OPENAI_REASONING_EFFORT=high
```

Set only the variables for the endpoint you intend to change.
Keep API keys out of command text, tracked files, and documentation; follow [Environment variables](environment-variables.md#example-configuration-file) for the `.env.example` template, private local setup, and Docker behavior.

## Response provenance and analysis alignment

- Response fields: Effective model, reasoning effort, prompt version, and schema version.
- Current prompts: Extraction remains `title-abstract-extraction-v1`; screening uses `title-abstract-screening-v2`.
  Screening v2 makes the existing final aggregation priority explicit: assess all criteria and conflicts, then let any unclear criterion or material conflict override even an evidenced exclusion and require included/low.
  Decision and evidence validators, provider output field order, model/reasoning configuration, and schema version `1` are unchanged.
- Historical artifacts: Stage-aware response validation still accepts screening-v1 for prediction and extraction-v1 for extraction without rewriting saved provenance; unknown or wrong-stage versions fail.
- Source: Server-loaded `OPENAI_STUDY_SCREENING_MODEL` and `OPENAI_STUDY_SCREENING_REASONING_EFFORT`, not caller or provider-response values.
- Client declaration: Expected provenance in the analysis run manifest.
- Acceptance and reuse: Returned provenance must match the declared model, reasoning effort, endpoint-specific prompt version, and schema version.
- Overrides: Match analysis-side expected-provenance options to server changes; mismatches are recorded as provider-contract errors.

See the [analysis screening workflow](../../analysis/README.md#guarded-screening-run) for those client-side controls and [API endpoints](api-endpoints.md#provenance) for the public response contract.

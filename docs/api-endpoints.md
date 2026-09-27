# API endpoints

This document describes the API endpoints implemented by the FastAPI service in `inference-api/app`.
When running with the default Docker Compose configuration, the base URL is `http://localhost:12306`.
The interactive OpenAPI documentation is available at `/docs` while the service is running.

## Schema notation

The typed blocks below describe the public request and response shapes without replacing the executable JSON examples.

- `field?: T` means the field may be omitted.
- `field: T | null` means the field is required but its value may be `null`.
- `field?: T | null` means the field may be omitted or explicitly set to `null`.
- `Array<T>` means a JSON array whose items have type `T`.
- An alias defines a reusable type, and `|` separates allowed alternatives.
- Object fields shown without `?` are required unless a constraint rule says otherwise.

## Authentication

`POST /study_screening/encode`, `POST /study_screening/predict/`, `POST /data_extraction/title_abstract/`, and `POST /risk_of_bias/assess` require an API key in the `X-API-Key` header.
The accepted key is configured with `WCRF_API_KEY`; see [Environment variables](environment-variables.md) for the canonical configuration reference.
Requests without the header return `403 Forbidden` with `{"detail":"Not authenticated"}`.
Requests with an incorrect key return `401 Unauthorized` with `{"detail":"The API key is not correct"}`.
Other project-defined endpoints do not currently require authentication.

Request and error types:

```text
AuthenticatedHeaders = {
  "X-API-Key": string
}

AuthenticationError = {
  detail: string
}
```

## GET /

Returns a simple string response that can be used as a minimal liveness check.

Response type:

```text
string
```

```json
"hello world"
```

## GET /check

Checks whether all configured model and data paths exist.
The paths are defined in `inference-api/app/resources/globals.py`.
It returns `true` only when all configured paths exist and otherwise returns `false`.

Response type:

```text
boolean
```

```json
true
```

## POST /debug/encode

Runs the debug ALBERT sentiment classifier over a small batch of text records.
This endpoint is intended for debugging model loading and inference plumbing.
It uses `models/albert-base-v2-imdb` and does not currently require authentication.

### Request body

Request type:

```text
ExampleInputRecord = {
  text: string
}

DebugEncodeRequest = {
  example_record: Array<ExampleInputRecord>
  param1: string | null
  param2: string | null
}
```

Constraint rules:

- `example_record`: Required array with at most two records.
- `param1`: Required nullable string; it has no default, so callers should include it.
- `param2`: Required nullable string; it has no default, so callers should include it.

```json
{
  "example_record": [
    {
      "text": "This study is useful."
    }
  ],
  "param1": null,
  "param2": null
}
```

### Response

Returns a list of predicted class labels from the ALBERT model.
The exact labels depend on the loaded model configuration.

Response type:

```text
DebugEncodeResponse = Array<string>
```

```json
[
  "LABEL_1"
]
```

## Prompt-based title and abstract screening

The prompt-based workflow has two authenticated JSON endpoints with exact trailing-slash paths: `POST /data_extraction/title_abstract/` and `POST /study_screening/predict/`.
Both use `X-API-Key` authentication and reject missing or invalid keys as described in [Authentication](#authentication).
Both accept only their declared fields, because their request and response models forbid undeclared fields.
Invalid request bodies, including undeclared fields, blank required strings, invalid enum values, and invalid nested types, return FastAPI or Pydantic `422 Unprocessable Entity` validation errors.
The server trims title and abstract boundary whitespace without changing internal content.
`title` must remain non-empty after trimming.
`abstract` may be omitted, `null`, empty, or whitespace-only, which are all canonicalized as title-only input.
The canonical document used for screening is `Title: {title}\nAbstract: {abstract}`.

### POST /data_extraction/title_abstract/

Extracts strict study characteristics from a title and optional abstract.
The extraction response is review-independent and can be supplied as the `characteristics` field of a later prediction request.

#### Complete request and response

Request type:

```text
TitleAbstractExtractionRequest = {
  title: string
  abstract?: string | null
}
```

Response type:

```text
TitleAbstractExtractionResponse = {
  characteristics: Characteristics
  provenance: ScreeningProvenance
}
```

```json
{
  "title": "Physical activity and colorectal cancer incidence",
  "abstract": "A prospective cohort examined physical activity and colorectal cancer using a hazard ratio."
}
```

```json
{
  "characteristics": {
    "human_study": "yes",
    "publication_type": "primary_research",
    "study_design": "prospective_cohort",
    "population": "Adults in a cohort",
    "setting": "United Kingdom",
    "sample_size": 12500,
    "exposures": ["physical activity"],
    "comparators": ["lower physical activity"],
    "outcomes": ["incident colorectal cancer"],
    "outcome_types": ["incidence"],
    "follow_up": "10 years",
    "effect_measures": ["hazard ratio"],
    "confidence_intervals_reported": "yes",
    "analysis_methods": ["Cox regression"],
    "evidence": [
      {
        "characteristic": "study_design",
        "source": "abstract",
        "quote": "prospective cohort"
      }
    ],
    "limitations": []
  },
  "provenance": {
    "model": "gpt-5.6-terra",
    "reasoning_effort": "medium",
    "prompt_version": "title-abstract-extraction-v1",
    "schema_version": "1"
  }
}
```

#### Title-only request and response behavior

```json
{
  "title": "Physical activity and colorectal cancer incidence",
  "abstract": null
}
```

Constraint rules:

- `title`: Required string that is trimmed at both boundaries and must remain non-empty.
- `abstract`: Optional nullable string. Empty and whitespace-only values are canonicalized to `null`.
- Title-only response: Extraction adds `No abstract was available; extraction is based on the title only.` to `characteristics.limitations` if it is not already present.

#### Characteristics schema

Type definitions:

```text
HumanStudy = "yes" | "no" | "unclear"

PublicationType =
  | "primary_research"
  | "systematic_review"
  | "meta_analysis"
  | "review"
  | "protocol"
  | "editorial"
  | "commentary"
  | "letter"
  | "conference_abstract"
  | "case_report"
  | "case_series"
  | "other"
  | "unclear"

StudyDesign =
  | "prospective_cohort"
  | "retrospective_cohort"
  | "case_cohort"
  | "nested_case_control"
  | "pooled_cohort_analysis"
  | "randomized_controlled_trial"
  | "pooled_randomized_trial_analysis"
  | "case_control"
  | "ecological"
  | "cross_sectional"
  | "non_randomized_controlled_trial"
  | "case_only"
  | "other"
  | "unclear"

OutcomeType =
  | "incidence"
  | "mortality"
  | "recurrence"
  | "survival"
  | "prevalence"
  | "other"
  | "unclear"

EvidenceSource = "title" | "abstract"

EvidenceQuote = {
  source: EvidenceSource
  quote: string
}

CharacteristicEvidence = {
  characteristic: string
  source: EvidenceSource
  quote: string
}

Characteristics = {
  human_study: HumanStudy
  publication_type: PublicationType
  study_design: StudyDesign
  population: string | null
  setting: string | null
  sample_size: integer > 0 | null
  exposures: Array<string>
  comparators: Array<string>
  outcomes: Array<string>
  outcome_types: Array<OutcomeType>
  follow_up: string | null
  effect_measures: Array<string>
  confidence_intervals_reported: HumanStudy
  analysis_methods: Array<string>
  evidence: Array<CharacteristicEvidence>
  limitations: Array<string>
}
```

Constraint rules:

- Enum fields: `human_study` and `confidence_intervals_reported` use `HumanStudy`.
- Enum fields: `publication_type`, `study_design`, and each `outcome_types` item use the alternatives shown above.
- Nullable fields: `population`, `setting`, `sample_size`, and `follow_up` are required fields whose values may be `null`.
- Positive integer: `sample_size` is positive only for one unambiguous enrolled or analyzed total; otherwise it is `null`.
- Arrays: `exposures`, `comparators`, `outcomes`, `outcome_types`, `effect_measures`, `analysis_methods`, `evidence`, and `limitations` are required arrays and may be empty.
- Evidence: Each extraction evidence item has non-empty `characteristic`, `source` of `title` or `abstract`, and a non-empty, non-blank `quote`.
- Evidence source: Each quote must be a verbatim substring of its named raw field after boundary trimming. The server validates extraction evidence quotes against the raw title or abstract.
- Inference: The service does not infer unsupported facts such as peer-review status, an unstated comparator, or an unstated effect measure.

### POST /study_screening/predict/

Performs conservative binary screening against a review topic and fully resolved criteria.
The supplied `characteristics` are advisory context, while the raw title and abstract are authoritative if they conflict.
`review_topic` and `criteria` must be non-empty after trimming.
`criteria` must not contain `[INSERT CANCER OUTCOME]`.

#### Complete request and response

Request type:

```text
StudyScreeningPredictionRequest = {
  title: string
  abstract?: string | null
  review_topic: string
  criteria: string
  characteristics: Characteristics
}
```

Response type:

```text
StudyScreeningPredictionResponse = {
  decision: "included" | "excluded"
  confidence: "high" | "moderate" | "low"
  rationale: string
  characteristic_conflicts: Array<CharacteristicConflict>
  criterion_assessments: Array<CriterionAssessment>
  provenance: ScreeningProvenance
}
```

```json
{
  "title": "Physical activity and colorectal cancer incidence",
  "abstract": "A prospective cohort examined physical activity and colorectal cancer using a hazard ratio.",
  "review_topic": "physical activity",
  "criteria": "Include cohort studies reporting colorectal cancer.",
  "characteristics": {
    "human_study": "yes",
    "publication_type": "primary_research",
    "study_design": "prospective_cohort",
    "population": "Adults in a cohort",
    "setting": "United Kingdom",
    "sample_size": 12500,
    "exposures": ["physical activity"],
    "comparators": ["lower physical activity"],
    "outcomes": ["incident colorectal cancer"],
    "outcome_types": ["incidence"],
    "follow_up": "10 years",
    "effect_measures": ["hazard ratio"],
    "confidence_intervals_reported": "yes",
    "analysis_methods": ["Cox regression"],
    "evidence": [],
    "limitations": []
  }
}
```

```json
{
  "decision": "included",
  "confidence": "low",
  "rationale": "The criterion is unclear from the available evidence.",
  "characteristic_conflicts": [],
  "criterion_assessments": [
    {
      "criterion": "Eligible cancer outcome",
      "status": "unclear",
      "rationale": "The abstract is incomplete.",
      "evidence": []
    }
  ],
  "provenance": {
    "model": "gpt-5.6-terra",
    "reasoning_effort": "medium",
    "prompt_version": "title-abstract-screening-v1",
    "schema_version": "1"
  }
}
```

#### Title-only request behavior

```json
{
  "title": "Physical activity and colorectal cancer incidence",
  "review_topic": "physical activity",
  "criteria": "Include cohort studies reporting colorectal cancer.",
  "characteristics": {
    "human_study": "unclear",
    "publication_type": "unclear",
    "study_design": "unclear",
    "population": null,
    "setting": null,
    "sample_size": null,
    "exposures": [],
    "comparators": [],
    "outcomes": [],
    "outcome_types": [],
    "follow_up": null,
    "effect_measures": [],
    "confidence_intervals_reported": "unclear",
    "analysis_methods": [],
    "evidence": [],
    "limitations": ["No abstract was available; extraction is based on the title only."]
  }
}
```

Title-only predictions cannot have `high` confidence.

Constraint rules:

- `title`: Required string, trimmed at both boundaries, and non-empty after trimming.
- `abstract`: Optional nullable string. Omitted, `null`, empty, and whitespace-only values are canonicalized as title-only input.
- `review_topic`: Required non-empty string after trimming.
- `criteria`: Required non-empty string after trimming and must not contain `[INSERT CANCER OUTCOME]`.
- `characteristics`: Required `Characteristics` object used as advisory context; raw title and abstract remain authoritative.
- Unknown fields: Request and response models forbid undeclared fields.
- Validation: Undeclared fields, blank required strings, invalid enum values, and invalid nested types return `422 Unprocessable Entity`; the unresolved criteria token is reported as the documented `503` provider-contract error.

#### Prediction schema and conservative invariants

Type definitions:

```text
ScreeningDecision = "included" | "excluded"
ScreeningConfidence = "high" | "moderate" | "low"
CriterionStatus = "met" | "not_met" | "unclear"

CharacteristicConflict = {
  characteristic: string
  supplied_value: string
  document_value: string
  material: boolean
  rationale: string
  evidence: Array<EvidenceQuote>
}

CriterionAssessment = {
  criterion: string
  status: CriterionStatus
  rationale: string
  evidence: Array<EvidenceQuote>
}
```

Constraint rules:

- `decision`: `included` or `excluded`.
- `confidence`: `high`, `moderate`, or `low`, describing evidence sufficiency and consistency rather than a calibrated probability.
- `criterion_assessments`: Each assessment has non-empty `criterion` and `rationale`, a `status` of `met`, `not_met`, or `unclear`, and an evidence array that may be empty.
- `characteristic_conflicts`: Always an array and may be empty.
- Conflict: Each conflict has non-empty `characteristic` and `rationale`, string `supplied_value` and `document_value`, boolean `material`, and a non-empty evidence array.
- Exclusion: An `excluded` decision requires at least one `not_met` criterion with evidence.
- Confidence: An `excluded` decision cannot have `low` confidence.
- Unclear criteria: If any criterion is `unclear`, the result must be `included` with `low` confidence.
- Material conflicts: If any disclosed conflict is material, the result must be `included` with `low` confidence.
- Evidence interpretation: A criterion that cannot be assessed from the title and abstract is `unclear`, not `not_met`. Missing, conflicting, or insufficient evidence is not evidence of ineligibility.
- Evidence quotes: Every quote is non-empty and must be a verbatim substring of its named raw `title` or `abstract` source after boundary trimming. The server validates quotes in prediction criterion and conflict evidence.
- Provider obligation: The provider is instructed to assess every operative criterion and disclose every detected conflict, but exhaustive semantic criterion or conflict detection in free text is a provider obligation rather than deterministic proof by the application.

#### Provenance

Type:

```text
ScreeningProvenance = {
  model: string
  reasoning_effort: "none" | "minimal" | "low" | "medium" | "high" | "xhigh"
  prompt_version: "title-abstract-extraction-v1" | "title-abstract-screening-v1"
  schema_version: "1"
}
```

Constraint rules:

- Ownership: The server creates `provenance` and does not accept it from callers or providers.
- `model` and `reasoning_effort`: Report the configured screening provider settings described in [LLM configuration](llm-configuration.md).
- `prompt_version`: Extraction uses `title-abstract-extraction-v1`; prediction uses `title-abstract-screening-v1`.
- `schema_version`: Both responses use `1`.

#### Operational errors

Provider configuration, provider failures, parsed-response failures, invalid evidence, and invariant failures return `503 Service Unavailable` rather than a screening decision or extraction response.
The response body has this exact shape.

Response type:

```text
ScreeningErrorCode =
  | "configuration_error"
  | "provider_transient"
  | "provider_request_error"
  | "provider_contract_error"
ScreeningStage = "extraction" | "prediction"

ScreeningErrorResponse = {
  detail: {
    code: ScreeningErrorCode
    stage: ScreeningStage
    message: string
  }
}
```

```json
{
  "detail": {
    "code": "provider_contract_error",
    "stage": "prediction",
    "message": "The prediction provider response did not satisfy the response contract."
  }
}
```

Constraint rules:

- `code`: `configuration_error`, `provider_transient`, `provider_request_error`, or `provider_contract_error`.
- `stage`: `extraction` or `prediction`.
- `configuration_error`: Missing or invalid configuration and unavailable required SDK support.
- `provider_transient`: Transport failures, timeouts, rate limits, and provider 5xx failures.
- `provider_request_error`: Non-transient provider request rejection.
- `provider_contract_error`: Unresolved criteria, absent or invalid parsed output, schema failure, invalid evidence, and invariant failure.
- `message`: Stable non-secret summary for the stated code and stage. It does not include provider responses, prompts, credentials, or request content.

## POST /study_screening/encode

Scores one study against a review topic and optional eligibility criteria.
The endpoint builds one query string and one study string, encodes both with the study-screening SentenceTransformer model, calculates cosine similarity, selects a threshold, and returns an inclusion decision.
This legacy embedding and threshold endpoint is retained without a contract change.

### Request headers

```http
X-API-Key: <api-key>
Accept: application/json
```

### Request body

Request type:

```text
StudyScreeningEncodeRequest = {
  review_topic: string
  criteria?: string | null
  study_title: string
  study_abstract: string
  strategy: string
}
```

Constraint rules:

- `criteria`: Optional nullable string; omission defaults to `null`.
- `strategy`: Selects the decision threshold.
- Threshold lookup: Uses the first word of `review_topic` as the topic key.

```json
{
  "review_topic": "cancer",
  "criteria": "Adults with dietary exposure data.",
  "study_title": "Example study title",
  "study_abstract": "Example study abstract.",
  "strategy": "best_recall"
}
```

### Response

Returns a two-item JSON array.
The first item is the cosine similarity score.
The second item is the decision label, currently `included` or `excluded`.

Response type:

```text
StudyScreeningEncodeResponse = [number, "included" | "excluded"]
```

```json
[
  0.82,
  "included"
]
```

### Error responses

Response type:

```text
AuthenticationError = {
  detail: string
}
```

Missing authentication header:

```json
{
  "detail": "Not authenticated"
}
```

Incorrect API key:

```json
{
  "detail": "The API key is not correct"
}
```

Invalid request bodies return FastAPI validation errors with status code `422`.

## POST /risk_of_bias/assess

Runs a modified RoB-NObs risk-of-bias assessment for one nutrition observational study PDF using the OpenAI API.
The endpoint follows the CUP cancer-incidence prompt protocol and returns domain-level judgements only.
It does not return an overall risk-of-bias judgement.
This endpoint requires `X-API-Key` authentication and server-side `OPENAI_API_KEY` configuration.
See [LLM configuration](llm-configuration.md) for the independent model, reasoning, provider compatibility, and lazy-initialization behavior.
See [Environment variables](environment-variables.md) for API-key setup and environment loading.

### Request headers

```http
X-API-Key: <api-key>
Accept: application/json
```

### Request body

The request uses `multipart/form-data`.
Only PDF uploads are supported in the current implementation.
`file` and `study_id` are required.
`exposure_timepoint`, `outcome_timepoint`, and repeated `domains` form fields are optional.

Request type:

```text
PDFUpload = uploaded file whose bytes start with "%PDF-"

RiskOfBiasAssessForm = {
  file: PDFUpload
  study_id: string
  exposure_timepoint?: string | null
  outcome_timepoint?: string | null
  domains?: Array<string> | null
}
```

Constraint rules:

- `file`: Required PDF upload. Empty uploads and uploads without the `%PDF-` signature return `400 Bad Request`.
- `study_id`: Required study identifier.
- `exposure_timepoint`: Optional nullable target exposure description.
- `outcome_timepoint`: Optional nullable target outcome description.
- `domains`: Optional nullable repeated form field for domain names. Omit it to assess all domains. An empty array is also treated as no selection and assesses all domains.
- Domain names: Unknown names return `422 Unprocessable Entity`; supplied names are matched after lowercasing and replacing non-alphanumeric runs with spaces.

Example:

```sh
curl -X POST "http://localhost:12306/risk_of_bias/assess" \
  -H "X-API-Key: <api-key>" \
  -F "study_id=Smith 2020" \
  -F "exposure_timepoint=baseline dietary fibre intake" \
  -F "outcome_timepoint=incident colorectal cancer" \
  -F "file=@study.pdf;type=application/pdf"
```

### Response

Returns a strict JSON object containing extracted study details and domain assessments.

Response type:

```text
RobAnswer = "Y" | "PY" | "PN" | "N" | "NI" | "NA"
RobJudgement = "Low" | "Moderate" | "Serious" | "Critical" | "No information"

SignallingQuestion = {
  question_id: string
  question: string
  answer: RobAnswer
  justification: string
}

KeyExtractedDetails = {
  population_sample?: string | null
  setting_country?: string | null
  study_design?: string | null
  exposure_definition?: string | null
  exposure_measurement?: string | null
  comparator?: string | null
  outcomes?: string | null
  outcome_measurement?: string | null
  follow_up_time?: string | null
  start_of_follow_up_relative_to_exposure_assessment?: string | null
  repeated_exposure_measurement?: string | null
  missing_data_summary?: string | null
  target_cancer_site_for_confounder_guidance?: string | null
  key_confounders_covariates?: string | null
  main_statistical_methods?: string | null
  inclusion_exclusion?: string | null
  protocol_or_analysis_plan_mentioned?: string | null
  notes?: string | null
}

DomainAssessment = {
  domain: string
  judgement: RobJudgement
  rationale: string
  signalling_questions?: Array<SignallingQuestion>
}

RiskOfBiasAssessment = {
  study_id: string
  study_design_guess: string
  key_extracted_details?: KeyExtractedDetails
  domains: Array<DomainAssessment>
}
```

Constraint rules:

- Object contract: The response models reject undeclared fields.
- Nullable details: Every `KeyExtractedDetails` value may be `null`; its fields are optional nullable fields, and absent provider details are returned as `null`.
- Signalling answers: Each `answer` is `Y`, `PY`, `PN`, `N`, `NI`, or `NA`.
- Domain judgements: Each `judgement` is `Low`, `Moderate`, `Serious`, `Critical`, or `No information`.
- Domain questions: `signalling_questions` is optional and defaults to an empty array when no questions are returned.
- Response defaults: `key_extracted_details` is optional in the model and defaults to an empty details object when omitted; the service response emits the detail keys with null values when no details are available.
- Domains: The response contains domain-level judgements only and does not contain an overall risk-of-bias judgement.

```json
{
  "study_id": "Smith 2020",
  "study_design_guess": "prospective cohort",
  "key_extracted_details": {
    "population_sample": null,
    "setting_country": null,
    "study_design": "cohort",
    "exposure_definition": null,
    "exposure_measurement": null,
    "comparator": null,
    "outcomes": null,
    "outcome_measurement": null,
    "follow_up_time": null,
    "start_of_follow_up_relative_to_exposure_assessment": null,
    "repeated_exposure_measurement": null,
    "missing_data_summary": null,
    "target_cancer_site_for_confounder_guidance": null,
    "key_confounders_covariates": null,
    "main_statistical_methods": null,
    "inclusion_exclusion": null,
    "protocol_or_analysis_plan_mentioned": null,
    "notes": null
  },
  "domains": [
    {
      "domain": "Bias due to confounding",
      "judgement": "Moderate",
      "rationale": "...",
      "signalling_questions": [
        {
          "question_id": "1.1",
          "question": "...",
          "answer": "PY",
          "justification": "..."
        }
      ]
    }
  ]
}
```

### Error responses

Response type for the documented HTTP errors:

```text
ErrorResponse = {
  detail: string
}
```

Non-PDF uploads return `400 Bad Request`.
Unknown domain names return `422 Unprocessable Entity`.
Missing OpenAI configuration or missing OpenAI SDK returns `503 Service Unavailable`.
Tests for this endpoint mock `assess_pdf_document` and do not call the live OpenAI API.

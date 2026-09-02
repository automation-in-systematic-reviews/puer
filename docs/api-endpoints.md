# API endpoints

This document describes the API endpoints implemented by the FastAPI service in `inference-api/app`.
When running with the default Docker Compose configuration, the base URL is `http://localhost:12306`.
The interactive OpenAPI documentation is available at `/docs` while the service is running.

## Authentication

`POST /study_screening/encode`, `POST /study_screening/predict/`, `POST /data_extraction/title_abstract/`, and `POST /risk_of_bias/assess` require an API key in the `X-API-Key` header.
The accepted key is loaded from the `WCRF_API_KEY` environment variable.
Requests without the header return `403 Forbidden` with `{"detail":"Not authenticated"}`.
Requests with an incorrect key return `401 Unauthorized` with `{"detail":"The API key is not correct"}`.
Other project-defined endpoints do not currently require authentication.

## GET /

Returns a simple string response that can be used as a minimal liveness check.

```json
"hello world"
```

## GET /check

Checks whether all configured model and data paths exist.
The paths are defined in `inference-api/app/resources/globals.py`.
It returns `true` only when all configured paths exist and otherwise returns `false`.

```json
true
```

## POST /debug/encode

Runs the debug ALBERT sentiment classifier over a small batch of text records.
This endpoint is intended for debugging model loading and inference plumbing.
It uses `models/albert-base-v2-imdb` and does not currently require authentication.

### Request body

`example_record` is limited to at most two records.
`param1` and `param2` are nullable strings but have no defaults, so callers should include them.

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

For title-only input, extraction adds `No abstract was available; extraction is based on the title only.` to `characteristics.limitations` if it is not already present.

#### Characteristics schema

`human_study` and `confidence_intervals_reported` are `yes`, `no`, or `unclear`.
`publication_type` is one of `primary_research`, `systematic_review`, `meta_analysis`, `review`, `protocol`, `editorial`, `commentary`, `letter`, `conference_abstract`, `case_report`, `case_series`, `other`, or `unclear`.
`study_design` is one of `prospective_cohort`, `retrospective_cohort`, `case_cohort`, `nested_case_control`, `pooled_cohort_analysis`, `randomized_controlled_trial`, `pooled_randomized_trial_analysis`, `case_control`, `ecological`, `cross_sectional`, `non_randomized_controlled_trial`, `case_only`, `other`, or `unclear`.
Each `outcome_types` item is one of `incidence`, `mortality`, `recurrence`, `survival`, `prevalence`, `other`, or `unclear`.
`population`, `setting`, `sample_size`, and `follow_up` are nullable.
`sample_size` is a positive integer only for an unambiguous enrolled or analyzed total and is otherwise `null`.
`exposures`, `comparators`, `outcomes`, `outcome_types`, `effect_measures`, `analysis_methods`, `evidence`, and `limitations` are arrays and may be empty.
Each extraction evidence item has non-empty `characteristic`, `source`, and `quote` fields, where `source` is `title` or `abstract`.
The service does not infer unsupported facts such as peer-review status, an unstated comparator, or an unstated effect measure.

### POST /study_screening/predict/

Performs conservative binary screening against a review topic and fully resolved criteria.
The supplied `characteristics` are advisory context, while the raw title and abstract are authoritative if they conflict.
`review_topic` and `criteria` must be non-empty after trimming.
`criteria` must not contain `[INSERT CANCER OUTCOME]`.

#### Complete request and response

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

#### Prediction schema and conservative invariants

`decision` is `included` or `excluded`.
`confidence` is `high`, `moderate`, or `low`, and describes evidence sufficiency and consistency rather than a calibrated probability.
Each criterion assessment has a non-empty `criterion` and `rationale`, a `status` of `met`, `not_met`, or `unclear`, and an `evidence` array that may be empty.
`characteristic_conflicts` is always an array and may be empty.
Each conflict has non-empty `characteristic` and `rationale`, string `supplied_value` and `document_value`, boolean `material`, and a non-empty evidence array.

An `excluded` decision requires at least one `not_met` criterion with evidence.
An `excluded` decision cannot have `low` confidence.
If any criterion is `unclear`, the result must be `included` with `low` confidence.
If any disclosed conflict is material, the result must be `included` with `low` confidence.
A criterion that cannot be assessed from the title and abstract is `unclear`, not `not_met`.
Missing, conflicting, or insufficient evidence is not evidence of ineligibility.

Every evidence quote is non-empty and must be a verbatim substring of its named raw `title` or `abstract` source after boundary trimming.
The server validates quotes in extraction evidence and prediction criterion and conflict evidence.
The provider is instructed to assess every operative criterion and disclose every detected conflict, but exhaustive semantic criterion or conflict detection in free text is a provider obligation rather than deterministic proof by the application.

#### Provenance

The server owns the `provenance` object and does not accept it from callers or providers.
`model` and `reasoning_effort` report the configured screening provider settings.
The default values are `gpt-5.6-terra` and `medium`.
Extraction uses `prompt_version` `title-abstract-extraction-v1`.
Prediction uses `prompt_version` `title-abstract-screening-v1`.
Both responses use `schema_version` `1`.

#### Operational errors

Provider configuration, provider failures, parsed-response failures, invalid evidence, and invariant failures return `503 Service Unavailable` rather than a screening decision or extraction response.
The response body has this exact shape.

```json
{
  "detail": {
    "code": "provider_contract_error",
    "stage": "prediction",
    "message": "The prediction provider response did not satisfy the response contract."
  }
}
```

`code` is `configuration_error`, `provider_transient`, `provider_request_error`, or `provider_contract_error`.
`stage` is `extraction` or `prediction`.
`configuration_error` covers missing or invalid configuration and unavailable required SDK support.
`provider_transient` covers transport failures, timeouts, rate limits, and provider 5xx failures.
`provider_request_error` covers non-transient provider request rejection.
`provider_contract_error` covers unresolved criteria, absent or invalid parsed output, schema failure, invalid evidence, and invariant failure.
`message` is a stable non-secret summary for the stated code and stage and does not include provider responses, prompts, credentials, or request content.

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

`criteria` is optional and defaults to `null` when omitted.
`strategy` is used to select the decision threshold.
The threshold lookup uses the first word of `review_topic` as the topic key.

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

```json
[
  0.82,
  "included"
]
```

### Error responses

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
This endpoint requires `X-API-Key` authentication.
The server also requires `OPENAI_API_KEY`.
`OPENAI_MODEL` defaults to `gpt-5.2`, and `OPENAI_REASONING_EFFORT` defaults to `medium`.

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

Fields:

- `file`: required PDF file.
- `study_id`: required study identifier.
- `exposure_timepoint`: optional target exposure description.
- `outcome_timepoint`: optional target outcome description.
- `domains`: optional repeated form field for domain names; omit to assess all domains.

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

Non-PDF uploads return `400 Bad Request`.
Unknown domain names return `422 Unprocessable Entity`.
Missing OpenAI configuration or missing OpenAI SDK returns `503 Service Unavailable`.
Tests for this endpoint mock `assess_pdf_document` and do not call the live OpenAI API.

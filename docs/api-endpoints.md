# API endpoints

This document describes the API endpoints implemented by the FastAPI service in `inference-api/app`.
When running with the default Docker Compose configuration, the base URL is `http://localhost:12306`.

The interactive OpenAPI documentation is available at `/docs` while the service is running.

## Authentication

`POST /study_screening/encode` and `POST /risk_of_bias/assess` require an API key in the `X-API-Key` header.
The accepted key is loaded from the `WCRF_API_KEY` environment variable.

Requests without the header return `403 Forbidden`.
Requests with an incorrect key return `401 Unauthorized`.

Other project-defined endpoints do not currently require authentication.

## GET /

Returns a simple string response that can be used as a minimal liveness check.

### Response

```json
"hello world"
```

## GET /check

Checks whether all configured model and data paths exist.
The paths are defined in `inference-api/app/resources/globals.py`.

### Response

Returns `true` only when all configured paths exist.
Returns `false` if one or more paths are missing.

```json
true
```

## POST /debug/encode

Runs the debug ALBERT sentiment classifier over a small batch of text records.
This endpoint is intended for debugging model loading and inference plumbing.

The endpoint uses `models/albert-base-v2-imdb`.
It does not currently require authentication.

### Request body

`example_record` is limited to at most two records.
`param1` and `param2` are nullable strings in the current model, but they do not have defaults and should be included in the request body.

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

```json
[
  "LABEL_1"
]
```

The exact labels depend on the loaded model configuration.

## POST /study_screening/encode

Scores one study against a review topic and optional eligibility criteria.
The endpoint builds one query string and one study string, encodes both with the study-screening SentenceTransformer model, calculates cosine similarity, selects a threshold, and returns an inclusion decision.

This endpoint requires `X-API-Key` authentication.

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

Tests for this endpoint should mock `assess_pdf_document` and must not call the live OpenAI API.

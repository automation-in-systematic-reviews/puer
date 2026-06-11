# API endpoints

This document describes the API endpoints implemented by the FastAPI service in `inference-api/app`.
When running with the default Docker Compose configuration, the base URL is `http://localhost:12306`.

The interactive OpenAPI documentation is available at `/docs` while the service is running.

## Authentication

`POST /study_screening/encode` requires an API key in the `X-API-Key` header.
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

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.funcs import risk_of_bias
from app.main import app
from app.resources import globals


API_ENDPOINT = "/risk_of_bias/assess"


def _empty_key_details(**overrides):
    details = {
        "population_sample": None,
        "setting_country": None,
        "study_design": None,
        "exposure_definition": None,
        "exposure_measurement": None,
        "comparator": None,
        "outcomes": None,
        "outcome_measurement": None,
        "follow_up_time": None,
        "start_of_follow_up_relative_to_exposure_assessment": None,
        "repeated_exposure_measurement": None,
        "missing_data_summary": None,
        "target_cancer_site_for_confounder_guidance": None,
        "key_confounders_covariates": None,
        "main_statistical_methods": None,
        "inclusion_exclusion": None,
        "protocol_or_analysis_plan_mentioned": None,
        "notes": None,
    }
    details.update(overrides)
    return details


def test_assess_pdf_returns_domain_assessments():
    with TestClient(app) as client:
        headers = {
            "Accept": "application/json",
            "x-api-key": globals.api_keys[0],
        }
        files = {
            "file": (
                "study.pdf",
                b"%PDF-1.4\n% test pdf bytes\n",
                "application/pdf",
            )
        }
        data = {
            "study_id": "Smith 2020",
            "exposure_timepoint": "baseline diet",
            "outcome_timepoint": "incident colorectal cancer",
        }
        expected = {
            "study_id": "Smith 2020",
            "study_design_guess": "prospective cohort",
            "key_extracted_details": _empty_key_details(study_design="cohort"),
            "domains": [
                {
                    "domain": "Bias due to confounding",
                    "judgement": "Moderate",
                    "rationale": "Adjusted for key confounders.",
                    "signalling_questions": [],
                }
            ],
        }

        with patch(
            "app.apis.risk_of_bias.assess_pdf_document",
            new=AsyncMock(return_value=expected),
        ) as mock_assess:
            response = client.post(
                API_ENDPOINT,
                headers=headers,
                files=files,
                data=data,
            )

        assert response.status_code == 200
        assert response.json() == expected
        mock_assess.assert_awaited_once()
        call_kwargs = mock_assess.await_args.kwargs
        assert call_kwargs["study_id"] == "Smith 2020"
        assert call_kwargs["filename"] == "study.pdf"
        assert call_kwargs["pdf_bytes"] == b"%PDF-1.4\n% test pdf bytes\n"


def test_assess_pdf_rejects_non_pdf_upload():
    with TestClient(app) as client:
        headers = {
            "Accept": "application/json",
            "x-api-key": globals.api_keys[0],
        }
        files = {
            "file": (
                "study.txt",
                b"not a pdf",
                "text/plain",
            )
        }
        data = {"study_id": "Smith 2020"}

        response = client.post(
            API_ENDPOINT,
            headers=headers,
            files=files,
            data=data,
        )

        assert response.status_code == 400
        assert response.json() == {"detail": "Only PDF uploads are supported"}


def test_assess_pdf_requires_api_key():
    with TestClient(app) as client:
        files = {
            "file": (
                "study.pdf",
                b"%PDF-1.4\n% test pdf bytes\n",
                "application/pdf",
            )
        }
        data = {"study_id": "Smith 2020"}

        response = client.post(API_ENDPOINT, files=files, data=data)

        assert response.status_code == 403


def test_assess_pdf_maps_service_error_to_503():
    with TestClient(app) as client:
        headers = {
            "Accept": "application/json",
            "x-api-key": globals.api_keys[0],
        }
        files = {
            "file": (
                "study.pdf",
                b"%PDF-1.4\n% test pdf bytes\n",
                "application/pdf",
            )
        }
        data = {"study_id": "Smith 2020"}

        with patch(
            "app.apis.risk_of_bias.assess_pdf_document",
            new=AsyncMock(
                side_effect=risk_of_bias.RiskOfBiasServiceError(
                    "OpenAI study-detail extraction failed"
                )
            ),
        ):
            response = client.post(
                API_ENDPOINT,
                headers=headers,
                files=files,
                data=data,
            )

        assert response.status_code == 503
        assert response.json() == {
            "detail": "OpenAI study-detail extraction failed"
        }


def test_select_domains_rejects_unknown_domain():
    with pytest.raises(ValueError):
        risk_of_bias.select_domains(["not a real domain"])


def test_assessment_deletes_uploaded_pdf_after_success():
    details = risk_of_bias.StudyDetails(
        study_id="Smith 2020",
        study_design_guess="prospective cohort",
        key_extracted_details=risk_of_bias.KeyExtractedDetails(
            study_design="cohort"
        ),
    )
    domain = risk_of_bias.DomainAssessment(
        domain="Bias due to confounding",
        judgement="Moderate",
        rationale="Adjusted for key confounders.",
    )

    with patch("app.funcs.risk_of_bias._get_openai_client") as mock_client, patch(
        "app.funcs.risk_of_bias._upload_pdf", return_value="file-123"
    ), patch(
        "app.funcs.risk_of_bias._extract_details", return_value=details
    ), patch(
        "app.funcs.risk_of_bias._assess_domain", return_value=domain
    ), patch(
        "app.funcs.risk_of_bias._delete_uploaded_pdf"
    ) as mock_delete:
        result = risk_of_bias._assess_pdf_document_sync(
            filename="study.pdf",
            pdf_bytes=b"%PDF-1.4\n% test pdf bytes\n",
            study_id="Smith 2020",
            domains=["Bias due to confounding"],
        )

    assert result.domains == [domain]
    mock_delete.assert_called_once_with(
        client=mock_client.return_value,
        file_id="file-123",
    )


def test_assessment_deletes_uploaded_pdf_after_failure():
    with patch("app.funcs.risk_of_bias._get_openai_client") as mock_client, patch(
        "app.funcs.risk_of_bias._upload_pdf", return_value="file-123"
    ), patch(
        "app.funcs.risk_of_bias._extract_details",
        side_effect=risk_of_bias.RiskOfBiasServiceError("OpenAI failed"),
    ), patch("app.funcs.risk_of_bias._delete_uploaded_pdf") as mock_delete:
        with pytest.raises(risk_of_bias.RiskOfBiasServiceError):
            risk_of_bias._assess_pdf_document_sync(
                filename="study.pdf",
                pdf_bytes=b"%PDF-1.4\n% test pdf bytes\n",
                study_id="Smith 2020",
                domains=["Bias due to confounding"],
            )

    mock_delete.assert_called_once_with(
        client=mock_client.return_value,
        file_id="file-123",
    )

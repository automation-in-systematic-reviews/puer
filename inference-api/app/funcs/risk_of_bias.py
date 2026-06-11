from __future__ import annotations

import asyncio
import io
import json
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.resources import globals


RobAnswer = Literal["Y", "PY", "PN", "N", "NI", "NA"]
RobJudgement = Literal[
    "Low", "Moderate", "Serious", "Critical", "No information"
]


class RiskOfBiasServiceError(RuntimeError):
    pass


class UnknownRiskOfBiasDomainError(ValueError):
    pass


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SignallingQuestion(StrictBaseModel):
    question_id: str
    question: str
    answer: RobAnswer
    justification: str


class DomainAssessment(StrictBaseModel):
    domain: str
    judgement: RobJudgement
    rationale: str
    signalling_questions: List[SignallingQuestion] = Field(default_factory=list)


class KeyExtractedDetails(StrictBaseModel):
    population_sample: Optional[str] = None
    setting_country: Optional[str] = None
    study_design: Optional[str] = None
    exposure_definition: Optional[str] = None
    exposure_measurement: Optional[str] = None
    comparator: Optional[str] = None
    outcomes: Optional[str] = None
    outcome_measurement: Optional[str] = None
    follow_up_time: Optional[str] = None
    start_of_follow_up_relative_to_exposure_assessment: Optional[str] = None
    repeated_exposure_measurement: Optional[str] = None
    missing_data_summary: Optional[str] = None
    target_cancer_site_for_confounder_guidance: Optional[str] = None
    key_confounders_covariates: Optional[str] = None
    main_statistical_methods: Optional[str] = None
    inclusion_exclusion: Optional[str] = None
    protocol_or_analysis_plan_mentioned: Optional[str] = None
    notes: Optional[str] = None


class StudyDetails(StrictBaseModel):
    study_id: str
    study_design_guess: str
    key_extracted_details: KeyExtractedDetails = Field(
        default_factory=KeyExtractedDetails
    )


class RiskOfBiasAssessment(StrictBaseModel):
    study_id: str
    study_design_guess: str
    key_extracted_details: KeyExtractedDetails = Field(
        default_factory=KeyExtractedDetails
    )
    domains: List[DomainAssessment]


SYSTEM_PROMPT = """You are an expert systematic reviewer and epidemiologist assessing risk of bias in nutrition observational studies using a MODIFIED RoB-NObs protocol tailored for cancer incidence studies.

Core rules:
- Use ONLY the information in the provided study PDF. If information is missing, answer NI and explain why.
- Return JSON that matches the requested schema exactly (no extra keys, no prose outside JSON).
- Be evidence-based in justifications; cite page/section or quote short snippets when possible.
- Use only these signalling-question responses: Y, PY, PN, N, NI, NA.
- Use only these domain judgements: Low, Moderate, Serious, Critical, No information.
- There is NO overall risk-of-bias judgement in this modified protocol. Assess domains separately only.
- Use the domain judgement algorithms below as decision aids after answering the signalling questions.

Protocol structure:
- Publication-level domains: 1 Confounding, 2 Selection of participants, 5 Missing data, 7 Selection of reported result.
- Exposure-outcome-specific domains: 3 Classification of exposures, 4 Departures from intended exposures, 6 Measurement of outcomes.

Protocol-specific operational rules:
- In observational nutrition studies, Q1.1 should almost always be PY.
- If Q1.2 asks whether exposure/follow-up was analysed as time-varying, answer strictly based on whether repeated exposure assessment with time-varying exposure/confounder analysis was done.
- In these studies, Q1.3 should usually be PY when Q1.2 is Y/PY.
- In these studies, Q4.1 and Q4.2 should usually be PY unless the paper clearly supports a different answer.
- For selection bias, follow-up and exposure assessment should coincide; >3 months difference suggests concern.
- For missing data, <10% missingness is generally acceptable when clearly reported.
- For exposure validity, acceptable guidance includes correlation >=0.4 for nutritional exposures and >=0.7 for BMI/self-reported anthropometric measures, when such statistics are reported.
- Cancer incidence and mortality outcomes ascertained by registries, hospital records, pathology, or laboratory confirmation are usually objective and less vulnerable to measurement bias.

Reasoning rules:
- Always follow skip logic from the signalling questions.
- Extract evidence first, then answer the signalling question.
- When the protocol gives a default/expected answer, use it unless the paper clearly supports a different answer.
- Do not invent unavailable details.
"""


ROB_NOBS_QUESTIONS_BY_DOMAIN: List[Dict[str, Any]] = [
    {
        "domain": "Bias due to confounding",
        "assessment_level": "publication",
        "protocol_notes": [
            "Q1.1 should almost always be PY in observational nutrition studies.",
            "If Q1.2 is Y/PY, Q1.3 should usually be PY in this review context.",
            "Evaluate adjustment against cancer-specific required/desirable confounders.",
            "Assess whether mediators or colliders were inappropriately adjusted for.",
        ],
        "judgement_algorithm": [
            "- Moderate if:",
            "  * baseline path: Q1.4 is Y/PY, Q1.5 is Y/PY, and Q1.6 is Y/PY/NI; or",
            "* time-varying path: Q1.7 is Y/PY and Q1.8 is Y/PY.",
            "- Serious if:",
            "  * baseline path: Q1.4 is Y/PY, Q1.5 is N/PN/NI, and Q1.6 is Y/PY/NI; or",
            "  * time-varying path: Q1.7 is Y/PY and Q1.8 is N/PN/NI.",
            "- Critical otherwise when the required conditions for Moderate or Serious are not met.",
            "- Leave blank if the prerequisite signalling questions for the chosen path are unanswered.",
        ],
        "questions": [
            (
                "1.1",
                "Is there potential for confounding of the effect of exposure in this study?",
            ),
            (
                "1.2",
                "If Y or PY to 1.1: Was the analysis based on splitting follow-up time according to exposure received?",
            ),
            (
                "1.3",
                "If Y or PY to 1.2: Were exposure discontinuations or switches likely to be related to factors that are prognostic for the outcome?",
            ),
            (
                "1.4",
                "If N or PN to 1.2 or 1.3: Did the authors use an appropriate analysis method that adjusted for all critically important baseline confounding variables?",
            ),
            (
                "1.5",
                "If N or PN to 1.2 or 1.3: Were adjusted confounders measured validly and reliably?",
            ),
            (
                "1.6",
                "If N or PN to 1.2 or 1.3: Did the authors avoid adjusting for post-exposure variables?",
            ),
            (
                "1.7",
                "If Y or PY to 1.3: Did the authors use an appropriate method that adjusted for baseline and time-varying confounding?",
            ),
            (
                "1.8",
                "If Y or PY to 1.3 and Y or PY to 1.7: Were adjusted confounders measured validly and reliably?",
            ),
        ],
    },
    {
        "domain": "Bias in selection of participants into the study",
        "assessment_level": "publication",
        "protocol_notes": [
            "Selection bias may occur when participation/contact/consent is required.",
            "Self-selection is often not strongly linked to future cancer outcomes; assess carefully.",
            "Start of follow-up should coincide with exposure assessment; >3 months difference suggests concern.",
        ],
        "judgement_algorithm": [
            "- Low if either:",
            "  * Q2.1 is N/PN and Q2.4 is Y/PY; or",
            "  * Q2.1 is Y/PY, both Q2.2 and Q2.3 are N, and Q2.4 is Y/PY.",
            "- Moderate if any of the following:",
            "  * Q2.1 is N/PN and Q2.4 is NI;",
            "  * Q2.1 is Y/PY and either (Q2.2 is N/PN and Q2.3 is Y/PY) or (Q2.2 is Y/PY and Q2.3 is N/PN), with Q2.4 = NI;",
            "  * Q2.1 is Y/PY, at least one of Q2.2/Q2.3 is PN, neither is N, and Q2.4 is Y/PY;",
            "  * Q2.1 is Y/PY and Q2.5 is Y/PY.",
            "- Serious if any of the following:",
            "  * Q2.1 is Y/PY, at least one of Q2.2/Q2.3 is PY, there are not two definite Y answers across Q2.2-Q2.3, and Q2.5 is N/PN;",
            "  * Q2.1 is Y/PY, Q2.2 = Y and Q2.3 = Y, Q2.4 is N/PN, and Q2.5 is Y/PY;",
            "  * at most one of Q2.2/Q2.3 is Y and Q2.4 is N/PN and Q2.5 is N/PN.",
            "- Critical if Q2.1 is Y/PY, Q2.2 = Y, Q2.3 = Y, and Q2.5 is N/PN.",
        ],
        "questions": [
            (
                "2.1",
                "Was selection of participants into the study or analysis based on participant characteristics observed after the start of exposure assessment?",
            ),
            (
                "2.2",
                "If Y or PY to 2.1: Were post-exposure variables that influenced selection associated with exposure?",
            ),
            (
                "2.3",
                "If Y or PY to 2.1: Were post-exposure variables that influenced selection associated with the outcome?",
            ),
            (
                "2.4",
                "Do start of follow-up and start of exposure assessment coincide for most participants?",
            ),
            (
                "2.5",
                "If Y or PY to 2.2 and 2.3, or N or PN to 2.4: Were adjustment techniques likely to correct for selection biases used?",
            ),
        ],
    },
    {
        "domain": "Bias in classification of exposures",
        "assessment_level": "exposure_outcome",
        "protocol_notes": [
            "Assess for the specific exposure-outcome association.",
            "Use protocol thresholds where available: correlation around 0.4 for nutritional exposures and around 0.7 for BMI/anthropometric self-report.",
            "Consider objective measures, biomarkers, repeated measures, and validation studies.",
        ],
        "judgement_algorithm": [
            "- Low if all of the following:",
            "  * Q3.1 = Y, Q3.2 = Y, Q3.3 = Y, Q3.6 = Y,",
            "  * Q3.4 = Y/PY, Q3.5 = Y/PY, Q3.7 = Y/PY,",
            "  * Q3.8 = N/PN.",
            "- Moderate if any of the following:",
            "  * Q3.1/Q3.2/Q3.3/Q3.6 are all Y/PY with at least one PY among Q3.1/Q3.2/Q3.3/Q3.6, and [Q3.4/Q3.5/Q3.7 are all Y/PY or Q3.8 is N/PN];",
            "  * all of Q3.1, Q3.2, Q3.3, Q3.6 are Y and Q3.4, Q3.5, Q3.7 are Y/PY and Q3.8 is Y/PY;",
            "  * one or more of Q3.1/Q3.2/Q3.3/Q3.5/Q3.7 is N/PN, with Q3.4 = Y/PY, Q3.6 = N/PN, and Q3.8 = N/PN.",
            "- Serious if any of the following:",
            "  * there is a mix of favourable and unfavourable answers across Q3.1-Q3.3 and Q3.6, plus some concern from Q3.4/Q3.5/Q3.7/Q3.8 but not enough for Critical;",
            "  * Q3.1 = Y, Q3.2 = Y, Q3.3 = Y, Q3.4 = N/PN, Q3.5 = Y, Q3.6 = Y/PY, Q3.7 = Y/PY, Q3.8 = Y/PY;",
            "  * one or more of Q3.1/Q3.2/Q3.3/Q3.5 is N/PN, Q3.6 = N/PN, and exactly one strong concern is present among Q3.4 = N/PN, Q3.7 = Y/PY, or Q3.8 = Y/PY.",
            "- Critical if:",
            "  * Q3.1 = N/PN, Q3.2 = N/PN, Q3.3 = N/PN, Q3.5 = N/PN,",
            "  * Q3.6 = N/PN,",
            "  * and at least two strong concerns are present among Q3.4 = N/PN, Q3.7 = N/PN, and Q3.8 = Y/PY.",
            "- Leave blank if any required signalling question is unanswered.",
        ],
        "questions": [
            ("3.1", "Is the exposure that was assessed clearly defined?"),
            (
                "3.2",
                "Does the exposure that was assessed represent the exposure of interest?",
            ),
            (
                "3.3",
                "Were the methods used to assess the exposure clearly described?",
            ),
            (
                "3.4",
                "Were the methods used to measure the exposure valid and/or reliable?",
            ),
            (
                "3.5",
                "Were the same methods used to assess exposure status for all participants/groups?",
            ),
            (
                "3.6",
                "Were the methods used to define exposure status for participants/groups clearly described?",
            ),
            (
                "3.7",
                "Were the methods used to define exposure status likely to result in minimal random or systematic exposure misclassification?",
            ),
            (
                "3.8",
                "Could classification of exposure status have been affected by the presence of the outcome, knowledge of the outcome, or risk of the outcome?",
            ),
        ],
    },
    {
        "domain": "Bias due to departures from intended exposures",
        "assessment_level": "exposure_outcome",
        "protocol_notes": [
            "Q4.1 should usually be PY in observational nutrition studies.",
            "Q4.2 should usually be PY in observational nutrition studies.",
            "Repeated exposure/confounder measurement, time-varying models, or sensitivity analyses may reduce concern.",
        ],
        "judgement_algorithm": [
            "Moderate if Q4.3 is Y or PY.",
            "Serious if Q4.3 is PN.",
            "Critical if Q4.3 is N.",
        ],
        "questions": [
            (
                "4.1",
                "Is there concern that changes in exposure status occurred among participants that were unbalanced across groups and likely to impact the outcome?",
            ),
            (
                "4.2",
                "Were any critical co-exposures unbalanced between exposure groups and likely to impact the outcome?",
            ),
            (
                "4.3",
                "If Y or PY to 4.1 or 4.2: Were adjustment techniques likely to correct for these issues used?",
            ),
        ],
    },
    {
        "domain": "Bias due to missing data",
        "assessment_level": "publication",
        "protocol_notes": [
            "<10% missingness is generally acceptable when clearly reported.",
            "Look for complete-case analysis, multiple imputation, IPW, g-formula, or marginal structural models.",
            "Assess whether missingness differs across exposure groups and whether reasons differ.",
        ],
        "judgement_algorithm": [
            "- No information if Q5.1-Q5.3 contain no Y/PY answers and at least one NI.",
            "- Low if either:",
            "  * all of Q5.1, Q5.2, Q5.3 are N/PN; or",
            "  * at least one of Q5.1-Q5.3 is Y/PY and [Q5.4 = Y or (Q5.4 = PY/NI and Q5.5 = Y/PY)].",
            "- Moderate if at least one of Q5.1-Q5.3 is Y/PY and one of the following:",
            "  * Q5.4 = PY and Q5.5 = N/PN; or",
            "  * Q5.4 = PN and Q5.5 = Y/PY; or",
            "  * Q5.4 = NI and Q5.5 = PN.",
            "- Serious if at least one of Q5.1-Q5.3 is Y/PY and one of the following:",
            "  * Q5.4 = PN and Q5.5 = N/PN; or",
            "  * Q5.4 = N and Q5.5 = Y/PY; or",
            "  * Q5.4 = NI and Q5.5 = N.",
            "- Critical if at least one of Q5.1-Q5.3 is Y/PY and Q5.4 = N and Q5.5 = N/PN.",
        ],
        "questions": [
            ("5.1", "Were there missing outcome data?"),
            (
                "5.2",
                "Were participants excluded due to missing data on exposure status?",
            ),
            (
                "5.3",
                "Were participants excluded due to missing data on other variables needed for the analysis?",
            ),
            (
                "5.4",
                "If Y or PY to 5.1, 5.2, or 5.3: Are the proportion of participants and reasons for missing data similar across exposure groups?",
            ),
            (
                "5.5",
                "If Y or PY to 5.1, 5.2, or 5.3: Were appropriate statistical methods used to account for missing data?",
            ),
        ],
    },
    {
        "domain": "Bias in measurement of outcomes",
        "assessment_level": "exposure_outcome",
        "protocol_notes": [
            "Assess for the specific outcome under review.",
            "Cancer incidence/mortality from registries, hospital records, pathology, or laboratory confirmation is usually objective.",
            "Self-reported outcomes are more vulnerable to bias, especially without verification.",
        ],
        "judgement_algorithm": [
            "- Low if Q6.1 = N/PN, Q6.2 = N/PN, Q6.3 = Y/PY, and Q6.4 = N/PN.",
            "- Moderate if:",
            "  * [Q6.1 = N and Q6.2 = Y/PY/NI, or Q6.1 = Y/PY and Q6.2 = N/PN/NI],",
            "  * and Q6.3 = Y/PY and Q6.4 = N/PN.",
            "- Serious if there is subjective or differential outcome measurement concern that does not reach Critical, typically when Q6.3 = N/PN or Q6.4 = Y/PY with mixed answers on Q6.1/Q6.2.",
            "- Critical if Q6.1 = Y/PY, Q6.2 = Y/PY, Q6.3 = N/PN, and Q6.4 = Y/PY.",
            "- Leave blank if required signalling questions are unanswered.",
        ],
        "questions": [
            (
                "6.1",
                "Could the outcome measure have been influenced by knowledge of the exposure received?",
            ),
            (
                "6.2",
                "Were outcome assessors aware of the exposure received by study participants?",
            ),
            (
                "6.3",
                "Were the methods of outcome assessment the same across exposure groups?",
            ),
            (
                "6.4",
                "Were any systematic errors during measurement of the outcome related to exposure received?",
            ),
        ],
    },
    {
        "domain": "Bias in selection of reported result",
        "assessment_level": "publication",
        "protocol_notes": [
            "There is often no accessible pre-registered protocol in observational studies.",
            "Compare methods against results for selective outcome, analysis, or subgroup reporting.",
        ],
        "judgement_algorithm": [
            "- Low if Q7.1, Q7.2, and Q7.3 are all N/PN and the protocol/analysis-plan support flag is Y.",
            "- Moderate if Q7.1, Q7.2, and Q7.3 are all N/PN and the protocol/analysis-plan support flag is not Y.",
            "- Serious if exactly one of Q7.1-Q7.3 is Y/PY.",
            "- Critical if more than one of Q7.1-Q7.3 is Y/PY.",
            "- Leave blank if any required signalling question is unanswered, or if all of Q7.1-Q7.3 are N/PN but the protocol/analysis-plan support flag is missing.",
        ],
        "questions": [
            (
                "7.1",
                "Is the reported effect estimate likely to be selected on the basis of multiple outcome measurements within the outcome domain?",
            ),
            (
                "7.2",
                "Is the reported effect estimate likely to be selected on the basis of multiple analyses of the exposure-outcome relationship?",
            ),
            (
                "7.3",
                "Is the reported effect estimate likely to be selected on the basis of different subgroups?",
            ),
            (
                "7.4",
                "If 7.1, 7.2, and 7.3 are N/PN: Were findings reported based on a pre-registered protocol or statistical analysis plan?",
            ),
        ],
    },
]


CANCER_CONFOUNDER_GUIDANCE_TEXT = """- Bladder: Required = Age, sex, smoking; Desirable = Body fatness
- Brain and spinal cord: Required = Age, sex; Desirable = None specified
- Colorectal: Required = Age, sex, body fatness; Desirable = Smoking, alcohol, physical activity, red/processed meat
- Gallbladder/bile ducts: Required = Age, sex, smoking, body fatness; Desirable = Gallstones/gallbladder disease/cholecystitis
- Kidney: Required = Age, sex, body fatness, smoking, alcohol; Desirable = Physical activity
- Liver: Required = Age, sex, alcohol, body fatness; Desirable = Smoking, race, physical activity, one of: HBV, HCV, liver cirrhosis
- Lung: Required = Age, sex, smoking; Desirable = Physical activity, race, environmental tobacco smoke
- Lymphoid and hematopoietic: Required = Age, sex; Desirable = None specified
- Mouth, pharynx, and larynx: Required = Age, sex, smoking, alcohol; Desirable = Body fatness, race, environmental tobacco smoke
- Nasopharynx: Required = Age, sex, smoking; Desirable = Body fatness
- Oesophageal adenocarcinoma: Required = Age, sex, body fatness; Desirable = Smoking, alcohol, race, physical activity
- Oesophageal squamous cell carcinoma: Required = Age, sex, smoking, alcohol; Desirable = Body fatness, race, physical activity
- Pancreas: Required = Age, sex, smoking, Body fatness; Desirable = Pancreatitis, alcohol
- Melanoma: Required = Age, sex, at least one of the following: UV light -sun exposure/ indoor tanning, sunscreen use, moles, fair skin, red/blond hair, blue/green eyes, freckles; Desirable = Race
- Basal cell skin cancer: Required = Age, sex, at least one of the following: UV light - sun exposure/ indoor tanning, sunscreen use, radiation therapy, immune-suppressing drugs; Desirable = Race
- Small intestine: Required = Age, sex; Desirable = Inflammatory Bowel Disease
- Stomach: Required = Age, sex, smoking, alcohol, body fatness; Desirable = Salt/preserved foods, race, Helicobacter Pylori
- Thyroid: Required = Age, sex; Desirable = One of: Benign thyroid nodules, goiter, thyroiditis
- Breast - premenopausal: Required = Age, body fatness, reproductive factor(s); Desirable = Alcohol, smoking, HRT/ oral contraceptive use, physical activity
- Breast - postmenopausal: Required = Age, body fatness, reproductive factor(s); Desirable = Alcohol, smoking, HRT, physical activity
- Cervix: Required = Age, HPV, smoking; Desirable = Body fatness, physical activity
- Endometrium: Required = Age, race, body fatness; Desirable = HRT/ oral contraceptive use, physical activity
- Ovary: Required = Age, body fatness, reproductive factor(s); Desirable = HRT/ oral contraceptive use
- Prostate: Required = Age, race; Desirable = Body fatness, physical activity, smoking

Protocol footnotes:
- Smoking-related cancers ideally require smoking status and duration/intensity where available.
- Reproductive factors may include age at menarche, age at menopause, parity, or lactation.
- Physical activity is required when investigating sedentary behaviour and cancer.
- Body fatness should generally be treated as a mediator for sugary sweetened drinks, juices, ultra-processed foods, and sedentary behaviour.
"""


def format_exposure_outcome_target(
    exposure_timepoint: Optional[str] = None,
    outcome_timepoint: Optional[str] = None,
) -> str:
    """Format the target exposure/outcome block for prompts."""
    parts: List[str] = []
    if exposure_timepoint:
        parts.append(f"- Exposure: {exposure_timepoint}")
    if outcome_timepoint:
        parts.append(f"- Outcome: {outcome_timepoint}")
    if not parts:
        return ""
    res = (
        "Target exposure-outcome association for this RoB assessment:\n"
        + "\n".join(parts)
        + "\nUse this specified exposure/outcome pair as the target association."
    )
    return res


def build_details_user_prompt(
    study_id: str,
    exposure_timepoint: Optional[str] = None,
    outcome_timepoint: Optional[str] = None,
) -> str:
    """Build the study-details extraction prompt."""
    target = format_exposure_outcome_target(
        exposure_timepoint=exposure_timepoint,
        outcome_timepoint=outcome_timepoint,
    )
    target_block = "\n\n" + target if target else ""
    res = f"""Study ID: {study_id}{target_block}

Task: Extract key study details ONLY (do not assess risk of bias yet).
Return JSON matching the StudyDetails schema.

This modified review concerns nutrition observational studies and cancer incidence outcomes.
Extract the following if available, otherwise null:
- population_sample
- setting_country
- study_design
- exposure_definition
- exposure_measurement
- comparator
- outcomes
- outcome_measurement
- follow_up_time
- start_of_follow_up_relative_to_exposure_assessment
- repeated_exposure_measurement
- missing_data_summary
- target_cancer_site_for_confounder_guidance
- key_confounders_covariates
- main_statistical_methods
- inclusion_exclusion
- protocol_or_analysis_plan_mentioned
- notes

Capture details that help answer whether exposure was time-varying, whether follow-up coincided with exposure assessment, which confounders were adjusted for, and whether missing data methods were applied.
"""
    return res


def _normalise_domain(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def select_domains(domains: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Return requested domain definitions in protocol order."""
    if not domains:
        return ROB_NOBS_QUESTIONS_BY_DOMAIN
    requested = {_normalise_domain(domain) for domain in domains}
    selected = [
        domain
        for domain in ROB_NOBS_QUESTIONS_BY_DOMAIN
        if _normalise_domain(domain["domain"]) in requested
    ]
    if len(selected) != len(requested):
        known = {domain["domain"] for domain in ROB_NOBS_QUESTIONS_BY_DOMAIN}
        raise UnknownRiskOfBiasDomainError(
            f"Unknown risk-of-bias domain. Known domains: {known}"
        )
    return selected


def build_domain_user_prompt(
    study_id: str,
    domain: str,
    questions: List[Tuple[str, str]],
    details: StudyDetails,
    exposure_timepoint: Optional[str] = None,
    outcome_timepoint: Optional[str] = None,
) -> str:
    """Build a single-domain RoB assessment prompt."""
    details_json = details.key_extracted_details.model_dump()
    domain_meta = next(
        (d for d in ROB_NOBS_QUESTIONS_BY_DOMAIN if d["domain"] == domain), None
    )
    assessment_level = "unspecified"
    protocol_notes: List[str] = []
    judgement_algorithm: List[str] = []
    if domain_meta:
        assessment_level = domain_meta.get("assessment_level", "unspecified")
        protocol_notes = domain_meta.get("protocol_notes", [])
        judgement_algorithm = domain_meta.get("judgement_algorithm", [])

    lines: List[str] = [f"Study ID: {study_id}"]
    target = format_exposure_outcome_target(
        exposure_timepoint=exposure_timepoint,
        outcome_timepoint=outcome_timepoint,
    )
    if target:
        lines.extend(["", target])
    lines.extend(
        [
            "",
            "Context (extracted summary; may be incomplete):",
            f"- Design guess: {details.study_design_guess}",
            f"- Key details: {json.dumps(details_json, ensure_ascii=False)}",
            "",
            "Task: Assess ONLY the following modified RoB-NObs domain for this study.",
            f"DOMAIN: {domain}",
            f"ASSESSMENT LEVEL: {assessment_level}",
            "",
            "Use the customised cancer-incidence protocol, not generic ROBINS-I or standard RoB-NObs.",
            "There is NO overall risk-of-bias judgement in this protocol.",
            "",
            "Answer every signalling question with: Y, PY, PN, N, NI, or NA.",
            "Provide evidence-based justifications from the PDF when possible.",
            "Apply skip logic exactly; skipped questions should use NA.",
            "",
        ]
    )
    if protocol_notes:
        lines.append("Protocol-specific notes for this domain:")
        lines.extend(f"- {note}" for note in protocol_notes)
        lines.append("")
    if domain == "Bias due to confounding":
        lines.extend(
            [
                "Cancer-site confounder guidance:",
                CANCER_CONFOUNDER_GUIDANCE_TEXT,
                "Use the target cancer site from the PDF or extracted details.",
                "",
            ]
        )
    if domain == "Bias in classification of exposures":
        lines.extend(
            [
                "Mention whether exposure was objective, self-reported, validated, or repeated over time.",
                "",
            ]
        )
    if domain == "Bias due to missing data":
        lines.extend(
            [
                "When percentages are reported, use them explicitly and note whether the <10% threshold is met.",
                "",
            ]
        )
    if domain == "Bias in measurement of outcomes":
        lines.extend(
            [
                "Note whether outcome ascertainment came from registry linkage, hospital records, pathology/lab confirmation, or self-report.",
                "",
            ]
        )
    lines.append("Signalling questions for this domain:")
    lines.extend(f"- {question_id} {question}" for question_id, question in questions)
    lines.append("")
    if judgement_algorithm:
        lines.append("After answering all signalling questions, apply this domain judgement algorithm:")
        lines.extend(f"- {step}" for step in judgement_algorithm)
        lines.append("")
    lines.append("Output: Return JSON matching the DomainAssessment schema exactly.")
    lines.append("Return only this domain's answers, justifications, and judgement.")
    res = "\n".join(lines)
    return res


def _get_openai_client():
    """Create an OpenAI client lazily so tests can mock without the SDK."""
    if not globals.openai_api_key:
        raise RiskOfBiasServiceError("OPENAI_API_KEY is not configured")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RiskOfBiasServiceError("The openai package is not installed") from exc
    res = OpenAI(api_key=globals.openai_api_key)
    return res


def _openai_error_types() -> tuple[type[Exception], ...]:
    """Return SDK exception types when the OpenAI package is installed."""
    try:
        from openai import OpenAIError
    except ImportError:
        return ()
    return (OpenAIError,)


def _parse_error_types() -> tuple[type[Exception], ...]:
    return _openai_error_types() + (ValidationError, ValueError)


def _reasoning_options() -> Optional[Dict[str, str]]:
    effort = globals.openai_reasoning_effort
    if effort == "none":
        return None
    return {"effort": effort, "summary": "auto"}


def _response_kwargs(schema: type[BaseModel]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"text_format": schema}
    reasoning = _reasoning_options()
    if reasoning:
        kwargs["reasoning"] = reasoning
    return kwargs


def _upload_pdf(client: Any, filename: str, pdf_bytes: bytes) -> str:
    pdf_file = io.BytesIO(pdf_bytes)
    pdf_file.name = filename
    try:
        uploaded = client.files.create(file=pdf_file, purpose="user_data")
    except _openai_error_types() as exc:
        raise RiskOfBiasServiceError("OpenAI PDF upload failed") from exc
    return uploaded.id


def _delete_uploaded_pdf(client: Any, file_id: str) -> None:
    try:
        client.files.delete(file_id)
    except _openai_error_types() as exc:
        logger.warning(f"OpenAI PDF cleanup failed for {file_id}: {exc}")


def _extract_details(
    client: Any,
    file_id: str,
    study_id: str,
    exposure_timepoint: Optional[str],
    outcome_timepoint: Optional[str],
) -> StudyDetails:
    user_prompt = build_details_user_prompt(
        study_id=study_id,
        exposure_timepoint=exposure_timepoint,
        outcome_timepoint=outcome_timepoint,
    )
    try:
        resp = client.responses.parse(
            model=globals.openai_model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                        {"type": "input_file", "file_id": file_id},
                    ],
                },
            ],
            **_response_kwargs(StudyDetails),
        )
    except _parse_error_types() as exc:
        raise RiskOfBiasServiceError(
            "OpenAI study-detail extraction failed"
        ) from exc
    if resp.output_parsed is None:
        raise RiskOfBiasServiceError(
            "OpenAI study-detail extraction returned no parsed output"
        )
    return resp.output_parsed


def _assess_domain(
    client: Any,
    file_id: str,
    study_id: str,
    domain_def: Dict[str, Any],
    details: StudyDetails,
    exposure_timepoint: Optional[str],
    outcome_timepoint: Optional[str],
) -> DomainAssessment:
    user_prompt = build_domain_user_prompt(
        study_id=study_id,
        domain=domain_def["domain"],
        questions=domain_def["questions"],
        details=details,
        exposure_timepoint=exposure_timepoint,
        outcome_timepoint=outcome_timepoint,
    )
    try:
        resp = client.responses.parse(
            model=globals.openai_model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                        {"type": "input_file", "file_id": file_id},
                    ],
                },
            ],
            **_response_kwargs(DomainAssessment),
        )
    except _parse_error_types() as exc:
        domain = domain_def["domain"]
        raise RiskOfBiasServiceError(
            f"OpenAI domain assessment failed: {domain}"
        ) from exc
    if resp.output_parsed is None:
        domain = domain_def["domain"]
        raise RiskOfBiasServiceError(
            f"OpenAI domain assessment returned no parsed output: {domain}"
        )
    return resp.output_parsed


def _assess_pdf_document_sync(
    filename: str,
    pdf_bytes: bytes,
    study_id: str,
    exposure_timepoint: Optional[str] = None,
    outcome_timepoint: Optional[str] = None,
    domains: Optional[List[str]] = None,
) -> RiskOfBiasAssessment:
    client = _get_openai_client()
    selected_domains = select_domains(domains)
    file_id = _upload_pdf(client, filename=filename, pdf_bytes=pdf_bytes)
    try:
        details = _extract_details(
            client=client,
            file_id=file_id,
            study_id=study_id,
            exposure_timepoint=exposure_timepoint,
            outcome_timepoint=outcome_timepoint,
        )
        domain_results = [
            _assess_domain(
                client=client,
                file_id=file_id,
                study_id=study_id,
                domain_def=domain_def,
                details=details,
                exposure_timepoint=exposure_timepoint,
                outcome_timepoint=outcome_timepoint,
            )
            for domain_def in selected_domains
        ]
        res = RiskOfBiasAssessment(
            study_id=study_id,
            study_design_guess=details.study_design_guess,
            key_extracted_details=details.key_extracted_details,
            domains=domain_results,
        )
    finally:
        _delete_uploaded_pdf(client=client, file_id=file_id)
    return res


async def assess_pdf_document(
    filename: str,
    pdf_bytes: bytes,
    study_id: str,
    exposure_timepoint: Optional[str] = None,
    outcome_timepoint: Optional[str] = None,
    domains: Optional[List[str]] = None,
) -> RiskOfBiasAssessment:
    """Assess a PDF with OpenAI and return domain-level RoB results."""
    res = await asyncio.to_thread(
        _assess_pdf_document_sync,
        filename=filename,
        pdf_bytes=pdf_bytes,
        study_id=study_id,
        exposure_timepoint=exposure_timepoint,
        outcome_timepoint=outcome_timepoint,
        domains=domains,
    )
    return res

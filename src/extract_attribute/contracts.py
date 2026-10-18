"""두 추출 방식의 JSON 형식과 결과 검증을 정의합니다."""

import logging

SENTIMENTS = ["positive", "negative", "mixed", "neutral", "unknown"]
FIELDS = {
    "attribute_units": ["raw_attribute", "sentiment"],
    "opinion_units": ["raw_aspect", "raw_status", "excerpt", "opinion", "sentiment"],
}
logger = logging.getLogger(__name__)


def output_schema(task: str) -> dict:
    properties = {}
    for field in FIELDS[task]:
        if field == "sentiment":
            properties[field] = {"type": "string", "enum": SENTIMENTS}
        elif field == "raw_status":
            properties[field] = {"type": ["string", "null"]}
        else:
            properties[field] = {"type": "string"}
    return {
        "type": "object",
        "properties": {
            task: {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": FIELDS[task],
                    "additionalProperties": False,
                },
            }
        },
        "required": [task],
        "additionalProperties": False,
    }


def parse_units(task: str, review: dict, response: dict) -> list[dict]:
    """응답 형식 오류는 실패시키고, 내용이 유효하지 않은 개별 단위만 제외합니다."""
    if not isinstance(response, dict) or set(response) != {task}:
        raise ValueError(f"{task}: 최상위 응답 필드가 올바르지 않습니다.")
    if not isinstance(response[task], list):
        raise ValueError(f"{task}: 결과는 배열이어야 합니다.")
    outputs = []
    for unit in response[task]:
        if not isinstance(unit, dict) or set(unit) != set(FIELDS[task]):
            raise ValueError(f"{task}: 추출 단위의 필드가 올바르지 않습니다.")
        valid = unit["sentiment"] in SENTIMENTS
        for field, value in unit.items():
            if field == "raw_status" and value is None:
                continue
            valid = valid and isinstance(value, str) and bool(value.strip())
        if task == "opinion_units":
            excerpt = unit["excerpt"]
            valid = valid and isinstance(excerpt, str) and excerpt in review["review"]
        if not valid:
            logger.warning("%s review_idx=%s: 유효하지 않은 단위 제외", task, review["idx"])
            continue
        # 입력 리뷰도 공백이 정리된 상태이므로 excerpt의 원문 일치를 유지합니다.
        cleaned = {
            field: " ".join(value.split()) if isinstance(value, str) else value
            for field, value in unit.items()
        }
        outputs.append({"review_idx": int(review["idx"]), **cleaned})
    return outputs

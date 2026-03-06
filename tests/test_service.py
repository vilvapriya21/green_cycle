from app.services.waste_audit_service import WasteAuditService


class FakeClassifier:
    def __init__(self, label="Recyclable", confidence=0.93, should_raise=False):
        self.label = label
        self.confidence = confidence
        self.should_raise = should_raise

    def predict(self, text: str) -> dict:
        if self.should_raise:
            raise RuntimeError("classifier failure")
        return {"label": self.label, "confidence": self.confidence}


class FakeLLMClient:
    def __init__(self, response=None):
        self.response = response
        self.last_prompt = None

    def generate(self, prompt: str):
        self.last_prompt = prompt
        return self.response


def test_generate_disposal_plan_returns_uncertain_for_empty_input():
    service = WasteAuditService(
        classifier=FakeClassifier(),
        llm_client=FakeLLMClient(),
    )

    result = service.generate_disposal_plan("")

    assert result["category"] == "Uncertain"
    assert result["confidence"] == 0.0
    assert "more specific description" in result["disposal_plan"].lower()


def test_generate_disposal_plan_returns_uncertain_for_low_confidence():
    service = WasteAuditService(
        classifier=FakeClassifier(label="Compost", confidence=0.30),
        llm_client=FakeLLMClient(response="Place it in the compost bin."),
    )

    result = service.generate_disposal_plan("banana peel")

    assert result["category"] == "Uncertain"
    assert result["confidence"] == 0.30


def test_generate_disposal_plan_uses_safe_llm_response():
    service = WasteAuditService(
        classifier=FakeClassifier(label="Hazardous", confidence=0.97),
        llm_client=FakeLLMClient(
            response="Seal the batteries in a safe container and take them to the hazardous waste facility."
        ),
    )

    result = service.generate_disposal_plan("used batteries")

    assert result["category"] == "Hazardous"
    assert result["confidence"] == 0.97
    assert "hazardous waste facility" in result["disposal_plan"].lower()


def test_generate_disposal_plan_falls_back_when_llm_returns_none():
    service = WasteAuditService(
        classifier=FakeClassifier(label="Recyclable", confidence=0.95),
        llm_client=FakeLLMClient(response=None),
    )

    result = service.generate_disposal_plan("plastic bottle")

    assert result["category"] == "Recyclable"
    assert "blue recycling bin" in result["disposal_plan"].lower() or "recycling" in result["disposal_plan"].lower()


def test_generate_disposal_plan_rejects_unsafe_llm_response():
    service = WasteAuditService(
        classifier=FakeClassifier(label="Hazardous", confidence=0.96),
        llm_client=FakeLLMClient(response="Burn the battery at home and throw the ash away."),
    )

    result = service.generate_disposal_plan("used batteries")

    assert result["category"] == "Hazardous"
    assert "leak-proof" in result["disposal_plan"].lower() or "hazardous waste facility" in result["disposal_plan"].lower()


def test_classify_returns_uncertain_when_classifier_fails():
    service = WasteAuditService(
        classifier=FakeClassifier(should_raise=True),
        llm_client=FakeLLMClient(),
    )

    result = service.classify("old paint can")

    assert result["label"] == "Uncertain"
    assert result["confidence"] == 0.0


def test_long_input_is_truncated_but_processed():
    service = WasteAuditService(
        classifier=FakeClassifier(label="Compost", confidence=0.92),
        llm_client=FakeLLMClient(response="Place the item in the green compost bin."),
    )

    text = "banana peel " * 100
    result = service.generate_disposal_plan(text)

    assert result["category"] == "Compost"
    assert "compost" in result["disposal_plan"].lower()

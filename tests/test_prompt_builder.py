from app.agent.prompt_builder import PromptBuilder


def test_prompt_builder_includes_inputs():
    prompt = PromptBuilder.build_prompt(
        description="used batteries",
        category="Hazardous",
        policy="Take it to a hazardous waste drop off.",
    )

    assert "used batteries" in prompt
    assert "Hazardous" in prompt
    assert "hazardous waste drop off" in prompt

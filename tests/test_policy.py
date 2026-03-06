from app.agent.policy import CityPolicyService


def test_get_policy_known_category():
    result = CityPolicyService.get_policy("Hazardous")
    assert "Hazardous Waste Facility" in result


def test_get_policy_unknown_category():
    result = CityPolicyService.get_policy("Unknown")
    assert "general municipal waste guidelines" in result.lower()

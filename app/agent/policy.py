class CityPolicyService:
    """
    Simulated city policy rules.
    """

    POLICIES = {
        "Recyclable": "Rinse the item and place it in the blue recycling bin.",
        "Compost": "Place the item in the green compost bin.",
        "Hazardous": "Seal the item in a leak-proof container and take it to the Hazardous Waste Facility."
    }

    @classmethod
    def get_policy(cls, category: str) -> str:
        return cls.POLICIES.get(category, "Follow general municipal waste guidelines.")

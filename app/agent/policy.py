from app.config import settings


class CityPolicyService:
    """
    Provides city waste disposal policies.
    Policies are loaded from application configuration.
    """

    @classmethod
    def get_policy(cls, category: str) -> str:
        """
        Retrieve disposal policy for a category.
        """

        policies = settings.CITY_POLICIES

        return policies.get(
            category,
            "Follow general municipal waste guidelines."
        )

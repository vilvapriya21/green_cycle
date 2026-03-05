class PromptBuilder:
    """
    Builds prompts for LLM disposal plan generation.
    """

    TEMPLATE = """
You are a waste disposal assistant.

Your task is to provide a safe and correct disposal plan based on the waste description and category.

Follow the provided city policy when generating the plan.

Examples:

Description: empty plastic bottle
Category: Recyclable
City Policy: Plastic bottles must be rinsed and placed in the blue bin.
Disposal Plan: Rinse the bottle and put it in the blue recycling bin.

Description: used batteries
Category: Hazardous
City Policy: Batteries must be taken to a hazardous waste drop off.
Disposal Plan: Take the batteries to the hazardous waste collection site.

Now generate the disposal plan.

Description: {description}
Category: {category}
City Policy: {policy}

Disposal Plan:
"""

    @classmethod
    def build_prompt(cls, description: str, category: str, policy: str) -> str:
        """
        Builds a formatted prompt for the LLM.
        """

        return cls.TEMPLATE.format(
            description=description,
            category=category,
            policy=policy
        )

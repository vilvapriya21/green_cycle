class PromptBuilder:
    """
    Builds few-shot prompts for LLM disposal plan generation.
    """

    @staticmethod
    def build_prompt(description: str, category: str, policy: str) -> str:
        return f"""
You are a waste disposal assistant. Given a waste description and its category,
provide a disposal plan according to City Policy.

Examples:

Description: empty plastic bottle
Category: Recyclable
City Policy: Plastic bottles must be rinsed and placed in the blue bin.
Disposal Plan: Rinse the bottle and put it in the blue recycling bin.

Description: used batteries
Category: Hazardous
City Policy: Batteries must be taken to a hazardous waste drop off.
Disposal Plan: Take the batteries to the hazardous waste collection site.

Now process:

Description: {description}
Category: {category}
City Policy: {policy}
Disposal Plan:
"""

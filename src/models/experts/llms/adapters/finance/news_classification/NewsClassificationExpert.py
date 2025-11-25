CATEGORIES = [
    "Finance",
    "Tax & Accounting",
    "Government & Controls",
    "Technology",
    "Industry",
    "Business & Management",
]


class NewsClassificationExpert:
    """
    Cleans and normalizes LLM output for news-category classification.

    The cleaner:
      1. Converts text to lowercase
      2. Checks for exact matches with known categories
      3. Checks for partial substring matches
      4. Returns "unknown" if no match found
    """
    def clean_output(self, raw: str) -> str:
        raw = raw.lower().strip()

        # Exact match
        for category in CATEGORIES:
            if category.lower() == raw:
                return category

        # Partial match
        for category in CATEGORIES:
            cat_l = category.lower()
            if raw in cat_l or cat_l in raw:
                return category

        return "unknown"

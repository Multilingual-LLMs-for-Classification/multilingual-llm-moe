import re

class SentimentAnalysisExpert:
    """
    Cleanup utility for sentiment-analysis outputs.

    This expert extracts a valid star-rating (1-5) from
    a raw model-generated string.
    """

    def clean_output(self, raw: str) -> str:
        """
        Extract a single digit between 1 and 5 from the raw model output.

        The method searches the text for an isolated digit 1-5 using a
        regular expression. If found, it returns the digit as a string.
        If no valid rating is found, an empty string is returned.

        Parameters
        ----------
        raw : str
            The raw output text produced by the LLM.

        Returns
        -------
        str
            A single character ("1"…"5") if detected, otherwise "".
        """
        m = re.search(r"\b([1-5])\b", raw)
        return m.group(1) if m else ""

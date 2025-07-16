import re


class TextNormalizer:
    """Normalize text for more accurate comparison."""

    def normalize(self, text):
        """Apply standard text normalization techniques."""
        if not text:
            return ""

        text = re.sub(r"[^A-Za-z0-9]", " ", text)
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()

        return text

import re

class BaseAnswerExtractor:
    """Base class for extracting answers from model responses."""
    
    def extract(self, response, extraction_params=None):
        """Extract answer from model response."""
        raise NotImplementedError("Subclasses must implement extract()")

class RegexAnswerExtractor(BaseAnswerExtractor):
    """Extract answers using regular expressions."""

    def __init__(self, pattern='(?<=: )(.+)'):
        self.pattern = pattern
    
    def extract(self, response, extraction_params=None):
        """Extract answer from response using regex pattern."""
        if not response:
            return ""
            
        try:
            match = re.search(self.pattern, response)
            return match.group(1).strip() if match and match.groups() else response.strip()
        except Exception as e:
            print(f"Regex extraction error: {e}")
            return response.strip()

class NumericAnswerExtractor(BaseAnswerExtractor):
    """Extract numeric answers, first from 'Answer: X' format, then last number in text."""
    
    def extract(self, response, extraction_params=None):
        """Extract numeric answer from response."""
        if not response:
            return ""
            
        try:
            answer_match = re.search(r'Answer:\s*(\d[\d,\.]*)', response)
            if answer_match:
                return self._clean_number(answer_match.group(1))
            numbers = re.findall(r'\b(\d[\d,\.]*)\b', response)
            return self._clean_number(numbers[-1]) if numbers else ""
        except Exception:
            return ""
    
    def _clean_number(self, number):
        """Clean up a number string to standard US format."""
        return re.sub(r'[^\d\.]', '', number.replace(",", "")).strip()

    @staticmethod
    def is_numeric_method(extraction_method):
        """Check if the extraction method is numeric-based."""
        return (extraction_method == 'numeric' or 
                (extraction_method and extraction_method.startswith('regex_') and 
                 "(?!.*\\d)" in extraction_method))

class MultipleChoiceExtractor(BaseAnswerExtractor):
    """Extract multiple choice answers (A, B, C, D)."""
    
    def extract(self, response, extraction_params={}):
        """Extract multiple choice answer from response."""
        pattern = extraction_params.get('pattern', '([A-D])')
        match = re.search(pattern, response)
        return match.group(1) if match else response.strip()

class RawTextExtractor(BaseAnswerExtractor):
    """Use the full text as the answer (e.g., for summarization)."""
    
    def extract(self, response, extraction_params=None):
        """Return the raw response text."""
        return response.strip() if response else ""

EXTRACTORS = {
    'raw': RawTextExtractor(),
    'multiple_choice': MultipleChoiceExtractor(),
    'regex': RegexAnswerExtractor(),
    'numeric': NumericAnswerExtractor(),
}

def get_extractor(extraction_method="raw"):
    """Get the appropriate extractor based on method name."""

    if extraction_method == 'numeric':
        return NumericAnswerExtractor()
    if extraction_method.startswith('regex_') and "(?!.*\\d)" in extraction_method:
        return NumericAnswerExtractor()
    if extraction_method.startswith('regex_'):
        pattern = extraction_method[6:]
        return RegexAnswerExtractor(pattern)

    return EXTRACTORS.get(extraction_method, RawTextExtractor())
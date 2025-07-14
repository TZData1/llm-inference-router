class BaseFeatureExtractor:
    """Base class for all feature extractors"""
    
    def __init__(self, config=None):
        self.config = config or {}
    
    def extract(self, query_text, metadata=None):
        """Extract feature from the query text and metadata"""
        raise NotImplementedError("Subclasses must implement extract()")
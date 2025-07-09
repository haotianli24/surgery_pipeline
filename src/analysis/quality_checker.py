from typing import Dict, Any, List
import re
import logging

class TranscriptQualityChecker:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.medical_terms = self._load_medical_terms()
    
    def _load_medical_terms(self) -> List[str]:
        """Load medical terminology for relevance checking"""
        # TODO: Load from medical terms file or database
        return ["surgery", "procedure", "anesthesia", "incision", "suture"]
    
    def assess_quality(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive quality assessment of transcript"""
        text = transcript.get("text", "")
        confidence = transcript.get("confidence", 0.0)
        
        # Basic quality metrics
        length = len(text)
        word_count = len(text.split())
        
        # Medical relevance
        medical_check = self.check_medical_relevance(text)
        
        # Confidence metrics
        confidence_metrics = self.calculate_confidence_metrics(transcript)
        
        return {
            "length": length,
            "word_count": word_count,
            "confidence_score": confidence,
            "medical_relevance": medical_check,
            "confidence_metrics": confidence_metrics,
            "meets_length_threshold": length >= self.config.min_transcript_length,
            "meets_confidence_threshold": confidence >= self.config.min_confidence_score,
            "overall_quality_score": min(confidence * 100, 100)  # Simple scoring
        }
        
    def check_medical_relevance(self, text: str) -> Dict[str, Any]:
        """Check if transcript contains medical/surgical content"""
        text_lower = text.lower()
        found_terms = []
        
        for term in self.medical_terms:
            if term.lower() in text_lower:
                found_terms.append(term)
        
        return {
            "found_terms": found_terms,
            "term_count": len(found_terms),
            "is_medical": len(found_terms) >= self.config.medical_terms_threshold,
            "relevance_score": min(len(found_terms) / 10.0, 1.0)  # Score 0-1
        }
        
    def calculate_confidence_metrics(self, transcript: Dict[str, Any]) -> Dict[str, float]:
        """Calculate various confidence metrics"""
        confidence = transcript.get("confidence", 0.0)
        text = transcript.get("text", "")
        
        # Simple metrics based on available data
        return {
            "overall_confidence": confidence,
            "text_quality": min(len(text) / 1000.0, 1.0),  # Normalize by expected length
            "completeness": 1.0 if len(text) > 50 else len(text) / 50.0
        }
        
    def is_valuable_for_annotation(self, quality_report: Dict[str, Any]) -> bool:
        """Determine if transcript meets criteria for AI annotation"""
        return (
            quality_report.get("meets_length_threshold", False) and
            quality_report.get("meets_confidence_threshold", False) and
            quality_report.get("medical_relevance", {}).get("is_medical", False)
        )
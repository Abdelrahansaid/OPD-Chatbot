"""
app/utils/kpi_similarity.py — Semantic KPI name matching
Finds best-matching KPI from user free text using sentence embeddings.
Example: "doctor performance score" → "Doctor PMS %"
"""
import logging
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)


class KPISimilarity:
    """
    Semantic KPI matching using multilingual embeddings.
    
    Features:
    ✓ Supports Arabic + English
    ✓ Fuzzy matching for typos (lakage → leakage)
    ✓ Caches embeddings for performance
    """
    
    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
    
    def __init__(self, kpi_list: List[str]):
        """
        Initialize with list of canonical KPI names.
        
        Args:
            kpi_list: List of KPI names (e.g., ["Total Revenue", "Doctor PMS %", ...])
        """
        try:
            from sentence_transformers import SentenceTransformer, util
            self._st_util = util
            self.model = SentenceTransformer(self.MODEL_NAME)
            self.kpi_list = kpi_list
            # Pre-compute embeddings for all KPIs (cached locally)
            self.kpi_embeddings = self.model.encode(kpi_list, convert_to_tensor=True)
            logger.info(f"✅ KPISimilarity initialized with {len(kpi_list)} KPIs")
        except Exception as e:
            logger.error(f"❌ KPISimilarity init failed: {e}")
            self.kpi_list = []
            self.kpi_embeddings = None
            self.model = None
            self._st_util = None
    
    def find_best_match(self, query: str, threshold: float = 0.5) -> Tuple[Optional[str], float]:
        """
        Find best-matching KPI from user query.
        
        Args:
            query: Free-text question (e.g., "What is doctor performance?")
            threshold: Minimum similarity score (0-1). Below this returns None.
        
        Returns:
            Tuple of (best_kpi_name, similarity_score) or (None, 0.0) if no match
        
        Examples:
            >>> sim = KPISimilarity(["Total Revenue", "Doctor PMS %", "No-Show %"])
            >>> sim.find_best_match("doctor performance score", threshold=0.5)
            ('Doctor PMS %', 0.78)
            
            >>> sim.find_best_match("random gibberish", threshold=0.5)
            (None, 0.0)
        """
        if not self.model or not self.kpi_list or not self._st_util:
            return None, 0.0
        
        try:
            # Encode the query
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            
            # Compute cosine similarity with all KPIs
            scores = self._st_util.cos_sim(query_embedding, self.kpi_embeddings)[0]
            
            # Find best match
            best_idx = scores.argmax().item()
            best_score = scores[best_idx].item()
            
            if best_score >= threshold:
                best_kpi = self.kpi_list[best_idx]
                logger.debug(f"KPI match: '{query}' → '{best_kpi}' (score: {best_score:.2f})")
                return best_kpi, best_score
            
            logger.debug(f"No KPI match for '{query}' (best score: {best_score:.2f} < threshold {threshold})")
            return None, 0.0
            
        except Exception as e:
            logger.error(f"❌ KPI matching failed for '{query}': {e}")
            return None, 0.0
    
    def find_top_k(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Find top-K matching KPIs sorted by similarity.
        
        Args:
            query: Free-text question
            k: Number of results to return
        
        Returns:
            List of (kpi_name, score) tuples, sorted by score descending
        """
        if not self.model or not self.kpi_list or not self._st_util:
            return []
        
        try:
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            scores = self._st_util.cos_sim(query_embedding, self.kpi_embeddings)[0]
            
            # Get top-K indices
            top_k_scores, top_k_idx = scores.topk(k=min(k, len(self.kpi_list)))
            
            results = [
                (self.kpi_list[idx.item()], score.item())
                for score, idx in zip(top_k_scores, top_k_idx)
            ]
            
            logger.debug(f"Top-{k} KPI matches for '{query}': {results}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Top-K KPI search failed: {e}")
            return []
    
    def __repr__(self) -> str:
        return f"KPISimilarity(model={self.MODEL_NAME}, kpis={len(self.kpi_list)})"


# Singleton instance (lazy-initialized in main.py)
_kpi_similarity_instance = None


def get_kpi_similarity(kpi_list: List[str] = None) -> Optional[KPISimilarity]:
    """Get or create KPI similarity singleton."""
    global _kpi_similarity_instance
    if _kpi_similarity_instance is None and kpi_list:
        _kpi_similarity_instance = KPISimilarity(kpi_list)
    return _kpi_similarity_instance

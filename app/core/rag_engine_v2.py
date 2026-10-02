# =================================================================
# app/core/rag_engine_v2.py — Hybrid RAG (Semantic + BM25 + Reranking)
# =================================================================
import logging
import pandas as pd
from typing import Optional, Dict, List, Tuple
from rank_bm25 import BM25Okapi
import re

logger = logging.getLogger(__name__)

class HybridRAGEngine:
    """Hybrid RAG combining semantic search (ChromaDB), BM25, and CrossEncoder reranking."""
    
    def __init__(self, chroma_collection, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Args:
            chroma_collection: ChromaDB collection object for semantic search
            reranker_model: CrossEncoder model for reranking results
        """
        self.chroma = chroma_collection
        self.bm25 = None
        self.corpus = []  # Store original texts for BM25
        self.reranker = None
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(reranker_model)
            logger.info(f"Loaded reranker model: {reranker_model}")
        except Exception as e:
            logger.warning(f"Failed to load reranker model: {e}")
            self.reranker = None
        self._index_corpus()
    
    def _index_corpus(self):
        """Build BM25 index from all KB documents."""
        try:
            results = self.chroma.get(include=["documents", "metadatas"])
            if not results or not results.get("documents"):
                logger.warning("No documents available from ChromaDB; skipping BM25 index build.")
                self.bm25 = None
                return

            self.corpus = results["documents"]
            # Tokenize for BM25 (simple whitespace + punctuation splitting)
            tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"Built BM25 index with {len(self.corpus)} documents")
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self.bm25 = None
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
    
    def search_semantic(self, query: str, top_k: int = 5) -> List[Dict]:
        """Semantic search via ChromaDB."""
        try:
            results = self.chroma.query(query_texts=[query], n_results=top_k, include=["documents", "metadatas", "distances"])
            
            if not results or not results.get("documents"):
                return []
            
            # ChromaDB returns distances (lower is better)
            semantic_results = []
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i]
                # Convert distance to similarity (0-1, higher is better)
                similarity = 1 / (1 + distance)
                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                
                semantic_results.append({
                    "text": doc,
                    "similarity": round(similarity, 3),
                    "source": "semantic",
                    "metadata": metadata
                })
            
            return semantic_results
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []
    
    def search_bm25(self, query: str, top_k: int = 5) -> List[Dict]:
        """Keyword search via BM25."""
        if not self.bm25:
            return []
        
        try:
            tokens = self._tokenize(query)
            if not tokens:
                return []
            
            scores = self.bm25.get_scores(tokens)
            # Get top-k indices
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            
            bm25_results = []
            for idx in top_indices:
                if idx < len(self.corpus):
                    bm25_results.append({
                        "text": self.corpus[idx],
                        "score": round(float(scores[idx]), 3),
                        "source": "bm25",
                        "metadata": {}
                    })
            
            return bm25_results
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return []
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 3) -> List[Dict]:
        """Rerank candidates using CrossEncoder."""
        if not candidates:
            return []
        
        try:
            # Prepare pairs for reranking
            pairs = [[query, cand["text"]] for cand in candidates]
            
            # Get reranking scores (0-1, higher is better)
            scores = self.reranker.predict(pairs)
            
            # Combine with original candidates and sort
            ranked = []
            for i, (cand, score) in enumerate(zip(candidates, scores)):
                ranked.append({
                    **cand,
                    "rerank_score": round(float(score), 3),
                    "rank": i + 1
                })
            
            # Sort by rerank score descending
            ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            return ranked[:top_k]
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            return candidates[:top_k]
    
    def hybrid_search(self, query: str, top_k: int = 3, use_rerank: bool = True) -> List[Dict]:
        """
        Hybrid search: combine semantic + BM25, optionally rerank.
        
        Returns:
            List of results with text, scores, source, and metadata
        """
        # Get results from both methods
        semantic_results = self.search_semantic(query, top_k=top_k)
        bm25_results = self.search_bm25(query, top_k=top_k)
        
        # Merge: prioritize semantic, fill with BM25
        merged_texts = set()
        merged_results = []
        
        for result in semantic_results:
            merged_results.append(result)
            merged_texts.add(result["text"][:50])  # Use first 50 chars as key
        
        for result in bm25_results:
            if result["text"][:50] not in merged_texts:
                merged_results.append(result)
                merged_texts.add(result["text"][:50])
            
            if len(merged_results) >= top_k * 2:
                break
        
        # Rerank if enabled
        if use_rerank and len(merged_results) > top_k:
            merged_results = self.rerank(query, merged_results, top_k=top_k)
            logger.info(f"Hybrid search returned {len(merged_results)} reranked results")
        else:
            merged_results = merged_results[:top_k]
            logger.info(f"Hybrid search returned {len(merged_results)} merged results (no rerank)")
        
        return merged_results
    
    def search_by_kpi(self, kpi_name: str, top_k: int = 3) -> List[Dict]:
        """Search KB for a specific KPI."""
        if not kpi_name:
            return []
        
        # Create query with KPI name and common follow-ups
        query = f"KPI {kpi_name} definition formula drivers investigation playbook"
        return self.hybrid_search(query, top_k=top_k, use_rerank=True)
    
    def get_stats(self) -> Dict:
        """Return indexing statistics."""
        return {
            "semantic_index_size": len(self.corpus) if self.corpus else 0,
            "bm25_ready": self.bm25 is not None,
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
        }

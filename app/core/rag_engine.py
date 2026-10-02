"""
app/core/rag_engine.py — Retrieval-Augmented Generation engine using ChromaDB
Supports multilingual (Arabic/English) semantic search across structured KB sheets.
"""
import os
import logging
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional
import pandas as pd
from app.config import KB_PATH, VECTOR_DB_PATH, HUGGINGFACE_HUB_TOKEN, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Vector-based knowledge retrieval system for OPD KPIs.
    
    Features:
    ✓ Multilingual embeddings (Arabic + English) via paraphrase-multilingual-MiniLM-L12-v2
    ✓ Indexes all 6 KB sheets with structured metadata
    ✓ Persistent ChromaDB storage with automatic reload
    ✓ Distance-based relevance scoring
    ✓ Safe cache invalidation
    """
    
    # Model supports 50+ languages including Arabic with good performance/size tradeoff
    EMBEDDING_MODEL = EMBEDDING_MODEL
    COLLECTION_NAME = "opd_knowledge_base"
    
    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name = collection_name or self.COLLECTION_NAME
        self.is_loaded = False
        
        # Initialize embedding function (lazy-load model on first use)
        self._embedding_func = None
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
            self.collection = None
            logger.info(f"ChromaDB client initialized at {VECTOR_DB_PATH}")
        except Exception as e:
            logger.error(f"Failed to init ChromaDB: {e}")
            self.client = None
            self.collection = None
        
        # Sheet-specific formatting templates
        self.sheet_templates = {
            'adx_kpi_knowledge_map': self._format_knowledge_map,
            'adx_kpi_formula_definition': self._format_formula,
            'adx_kpi_relationship_map': self._format_relationship,
            'adx_kpi_investigation_playbook': self._format_playbook,
            'adx_kpi_filter_compatibility': self._format_filter_compat,
            'adx_dim_kpi_scope': self._format_scope
        }
    
    @property
    def embedding_func(self):
        """Lazy-load the embedding model to avoid startup delay."""
        if self._embedding_func is None:
            if HUGGINGFACE_HUB_TOKEN:
                os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", HUGGINGFACE_HUB_TOKEN)
                os.environ.setdefault("HF_TOKEN", HUGGINGFACE_HUB_TOKEN)

            try:
                self._embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.EMBEDDING_MODEL
                )
                logger.info(f"Loaded embedding model: {self.EMBEDDING_MODEL}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self._embedding_func = None
        return self._embedding_func
    
    def _format_knowledge_map(self, row: pd.Series) -> str:
        """Format a row from adx_kpi_knowledge_map for embedding."""
        return f"""KPI: {row.get('KPI_Name', '')}
Layer: {row.get('KPI_Layer', '')}
Question: {row.get('Business_Question', '')}
Formula: {row.get('Financial_Impact_Formula', '')}
Primary Driver: {row.get('Primary_Driver_KPI', '')}
Investigation: {row.get('Investigation_Step_1', '')} | {row.get('Investigation_Step_2', '')}
Action: {row.get('Recommended_Action', '')}
Owner: {row.get('Action_Owner', '')} | Escalate: {row.get('Escalation_Level', '')}"""
    
    def _format_formula(self, row: pd.Series) -> str:
        return f"""KPI: {row.get('KPI_Name', '')}
Type: {row.get('Formula_Type', '')}
Logic: {row.get('Formula_Logic', '')}
Source: {row.get('Source', '')}"""
    
    def _format_relationship(self, row: pd.Series) -> str:
        return f"""Parent KPI: {row.get('Parent_KPI', '')}
Child KPI: {row.get('Child_KPI', '')}
Relationship: {row.get('Relationship_Type', '')} (weight: {row.get('Weight', '')})
Investigation Order: {row.get('Investigation_Order', '')}"""
    
    def _format_playbook(self, row: pd.Series) -> str:
        return f"""KPI: {row.get('KPI', '')}
Scenario: {row.get('Scenario', '')}
Threshold: {row.get('Threshold', '')} | Severity: {row.get('Severity', '')}
Root Cause Focus: {row.get('Root_Cause_Focus', '')}
Investigation: {row.get('Recommended_Investigation', '')}
Action: {row.get('Recommended_Action', '')}
Escalation: {row.get('Escalation', '')}"""
    
    def _format_filter_compat(self, row: pd.Series) -> str:
        return f"""KPI: {row.get('KPI_Name', '')}
Filters: Doctor={row.get('Doctor_Filter','')}, Specialty={row.get('Specialty_Filter','')}, 
         Date={row.get('Date_Filter','')}, BU={row.get('BU_Filter','')}, Payer={row.get('Payer_Filter','')}"""
    
    def _format_scope(self, row: pd.Series) -> str:
        return f"""KPI: {row.get('KPI_Name', '')}
Available at: Doctor={row.get('Available_Doctor_Level','')}, BU={row.get('Available_BU_Level','')}
Granularity: {row.get('Lowest_Granularity','')}
Default View: {row.get('Default_Display_Level','')}
Not Available Msg: {row.get('Not_Available_Message','')}"""
    
    def load_and_index_kb(self, kb_path: Optional[str] = None) -> int:
        """
        Load Knowledge Base Excel and index all sheets into ChromaDB.
        
        Returns:
            Total number of documents indexed.
        """
        if self.is_loaded:
            logger.info("RAG Engine already loaded")
            return self.collection.count() if self.collection else 0
        
        path = kb_path or KB_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Knowledge Base not found: {path}")
        
        # Read all sheets
        try:
            sheets = pd.read_excel(path, sheet_name=None)
            logger.info(f"Loaded {len(sheets)} KB sheets: {list(sheets.keys())}")
        except Exception as e:
            logger.error(f"Error reading KB file: {e}")
            return 0
        
        # Get or create collection with embedding function
        if self.embedding_func is None:
            logger.warning("Skipping KB indexing because embedding model failed to load.")
            return 0

        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_func
            )
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return 0
        
        docs, metas, ids = [], [], []
        doc_id = 0
        
        for sheet_name, df in sheets.items():
            # Clean sheet name (handle Excel tab characters)
            clean_name = sheet_name.replace('\t', '').strip('_').strip()
            formatter = self.sheet_templates.get(clean_name, lambda r: " | ".join(str(v) for v in r.values if pd.notna(v)))
            
            for _, row in df.iterrows():
                if pd.isna(row).all():
                    continue
                    
                try:
                    text = formatter(row)
                    docs.append(text)
                    metas.append({
                        "sheet": clean_name,
                        "kpi_name": row.get('KPI_Name') or row.get('KPI') or row.get('Parent_KPI') or '',
                        "doc_id": f"kb_{doc_id}"
                    })
                    ids.append(f"kb_{doc_id}")
                    doc_id += 1
                except Exception as e:
                    logger.warning(f"Skipping row due to error: {e}")
        
        if docs:
            # Batch add to ChromaDB (handles embedding internally)
            try:
                self.collection.add(documents=docs, metadatas=metas, ids=ids)
                logger.info(f"✅ Indexed {len(docs)} KB chunks into '{self.collection_name}'")
                self.is_loaded = True
                return len(docs)
            except Exception as e:
                logger.error(f"Error adding docs to collection: {e}")
                return 0
        else:
            logger.warning("⚠️ No documents to index — check KB file structure")
            return 0
    
    def search(self, query: str, top_k: int = 4, 
               sheet_filter: Optional[str] = None,
               min_distance: float = 2.0) -> List[Dict[str, any]]:
        """
        Perform semantic search over indexed knowledge.
        """
        if not self.is_loaded and self.collection is None:
            # Try to load if not loaded
            self.load_and_index_kb()

        if not self.collection:
            return []
            
        where_clause = {"sheet": sheet_filter} if sheet_filter else None
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            contexts = []
            if results["documents"]:
                for doc, meta, dist in zip(
                    results["documents"][0], 
                    results["metadatas"][0], 
                    results["distances"][0]
                ):
                    # Filter by relevance threshold
                    if dist > min_distance:
                        continue
                        
                    contexts.append({
                        "text": doc,
                        "sheet": meta.get("sheet", "unknown"),
                        "kpi_name": meta.get("kpi_name", ""),
                        "distance": round(float(dist), 4),
                        "relevance": "high" if dist < 0.8 else "medium" if dist < 1.5 else "low"
                    })
            
            # Sort by relevance
            contexts.sort(key=lambda x: x["distance"])
            return contexts
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    def search_by_kpi(self, kpi_name: str, top_k: int = 3) -> List[Dict[str, any]]:
        """Convenience method: search using canonical KPI name as query."""
        return self.search(f"KPI: {kpi_name}", top_k=top_k, min_distance=1.8)
    
    def clear_cache(self) -> None:
        """
        Safer cache clearing that deletes the collection instead of the folder.
        This avoids Windows file-lock errors.
        """
        import shutil
        import time
        
        # Method 1: Delete the collection via ChromaDB API (preferred)
        if self.client:
            try:
                self.client.delete_collection(name=self.collection_name)
                print(f"✅ Collection '{self.collection_name}' deleted successfully.")
                self.is_loaded = False
                self.collection = None
                return
            except Exception as e:
                logger.warning(f"Collection delete failed (may not exist): {e}")
        
        # Method 2: Delete the folder and reinitialize (fallback for lock issues)
        try:
            # Release references
            self.collection = None
            self.is_loaded = False
            
            # Small delay to help OS release file locks on Windows
            time.sleep(0.3)
            
            if os.path.exists(VECTOR_DB_PATH):
                shutil.rmtree(VECTOR_DB_PATH, ignore_errors=True)
                print("🗑️ Vector DB folder cleared.")
            
            # Reinitialize the client immediately so it's ready for next use
            self.client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
            print("✅ ChromaDB client reinitialized.")
            
        except Exception as ex:
            logger.error(f"Cache clear fallback failed: {ex}")
            print(f"⚠️ Could not fully clear cache. Try restarting your IDE if issues persist.")
            # At least ensure client is reinitialized
            try:
                self.client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
            except:
                pass

    def get_stats(self) -> Dict[str, any]:
        """Return collection statistics for monitoring."""
        if not self.collection:
            return {"status": "not_initialized"}
        
        try:
            return {
                "collection": self.collection_name,
                "document_count": self.collection.count(),
                "model": self.EMBEDDING_MODEL,
                "is_loaded": self.is_loaded
            }
        except Exception:
            return {"status": "error_reading_stats"}
    
    def __repr__(self) -> str:
        try:
            count = self.collection.count() if self.collection else 0
            return f"RAGEngine(collection='{self.collection_name}', docs={count}, loaded={self.is_loaded})"
        except:
            return f"RAGEngine(collection='{self.collection_name}', loaded={self.is_loaded})"


# Singleton instance
rag_engine = RAGEngine()
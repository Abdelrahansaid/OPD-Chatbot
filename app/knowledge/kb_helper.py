"""
app/knowledge/kb_helper.py — Knowledge Base Helper with improved column handling
Phase 1 FIX: Robust column name matching (handles KPI_Name, KPI Name, Name variations)
"""
import pandas as pd
import re
import logging
from typing import Dict, List, Any, Optional
from app.utils.filters import normalize_kpi_name

logger = logging.getLogger(__name__)


class KBHelper:
    """
    Helper class for Knowledge Base lookups with improved error handling.
    Handles column name variations and fuzzy matching.
    """
    
    def __init__(self, formulas: pd.DataFrame, knowledge_map: pd.DataFrame, 
                 relationship: pd.DataFrame, playbook: pd.DataFrame,
                 filter_compat: pd.DataFrame, scope: pd.DataFrame):
        self.formulas = formulas
        self.knowledge_map = knowledge_map
        self.relationship = relationship
        self.playbook = playbook
        self.filter_compat = filter_compat
        self.scope = scope
        logger.info("KBHelper initialized successfully")
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find actual column name by trying normalized variations."""
        normalized_columns = {
            re.sub(r'[_\s]+', ' ', col.lower()).strip(): col
            for col in df.columns
        }

        for col_name in possible_names:
            if col_name in df.columns:
                return col_name
            normalized = re.sub(r'[_\s]+', ' ', col_name.lower()).strip()
            if normalized in normalized_columns:
                return normalized_columns[normalized]

        # fallback to any KPI-like column if query asks for KPI fields
        for actual_col in df.columns:
            if 'kpi' in actual_col.lower() and 'kpi' in ' '.join(possible_names).lower():
                return actual_col

        return None
    
    def _search(self, df: pd.DataFrame, col: str, kpi_name: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        col_map = {re.sub(r'[_\s]+', ' ', c.lower()).strip(): c for c in df.columns}
        search_col = None
        for name in [col, 'KPI', 'KPI_Name', 'KPI Name', 'Name', col.replace('_', ' '), col.replace(' ', '_')]:
            norm = re.sub(r'[_\s]+', ' ', name.lower()).strip()
            if norm in col_map:
                search_col = col_map[norm]
                break
        if search_col is None:
            logger.warning(f"_search: Column '{col}' not found.")
            return pd.DataFrame()

        def norm_text(t: str) -> str:
            return re.sub(r'[\s%_\-]+', ' ', str(t).lower()).strip()

        search_val = norm_text(kpi_name)
        canonical = normalize_kpi_name(kpi_name)
        variants = [search_val]
        if canonical:
            variants.append(norm_text(canonical))

        if re.search(r'doctor\s*pms|pms', search_val, re.I):
            variants.extend(['doctor pms', 'doctor pms %', 'pms', 'pms %'])
        if re.search(r'no[\-\s]*show', search_val, re.I):
            variants.extend(['no-show %', 'no-show', 'noshow'])
        if re.search(r'service\s+leakage', search_val, re.I):
            variants.extend(['service leakage %', 'service leakage'])

        variants = list(dict.fromkeys(v for v in variants if v))
        values = df[search_col].astype(str).fillna('').apply(norm_text)

        for v in variants:
            mask = values.eq(v)
            if mask.any():
                print(f"DEBUG _search: Found match for '{kpi_name}' with variant '{v}'")
                return df[mask]

        for v in variants:
            try:
                mask = values.str.contains(re.escape(v), na=False)
                if mask.any():
                    print(f"DEBUG _search: Found match for '{kpi_name}' with variant '{v}'")
                    return df[mask]
            except re.error:
                continue

        logger.warning(f"No match for '{kpi_name}' in '{search_col}'")
        return pd.DataFrame()
    
    def get_formula(self, kpi_name: str) -> dict:
        kpi_name = normalize_kpi_name(kpi_name) or kpi_name
        match = self._search(self.formulas, 'KPI_Name', kpi_name)
        if match.empty:
            return {"found": False, "kpi": kpi_name}
        
        row = match.iloc[0]
        return {
            "found": True,
            "kpi_name": row['KPI_Name'],
            "formula_type": row['Formula_Type'],
            "formula_logic": row['Formula_Logic'],
            "source": row['Source'],
            "kpi_id": row['KPI_ID']
        }
    
    def get_kpi_info(self, kpi_name: str) -> dict:
        kpi_name = normalize_kpi_name(kpi_name) or kpi_name
        match = self._search(self.knowledge_map, 'KPI_Name', kpi_name)
        if match.empty:
            return {"found": False}
        
        row = match.iloc[0]
        return {
            "found": True,
            "kpi_name": row['KPI_Name'],
            "business_question": row['Business_Question'],
            "kpi_layer": row['KPI_Layer'],
            "owner_role": row['KPI_Owner_Role'],
            "function_owner": row['Function_Owner'],
            "financial_impact": row['Financial_Impact_Formula'],
            "primary_driver": row['Primary_Driver_KPI'],
            "secondary_driver": row['Secondary_Driver_KPI'],
            "investigation_steps": [
                row['Investigation_Step_1'], row['Investigation_Step_2'],
                row['Investigation_Step_3'], row['Investigation_Step_4']
            ],
            "action_owner": row['Action_Owner'],
            "escalation_level": row['Escalation_Level'],
            "recommended_action": row['Recommended_Action']
        }
    
    def get_drivers(self, kpi_name: str) -> list:
        kpi_name = normalize_kpi_name(kpi_name) or kpi_name
        matches = self._search(self.relationship, 'Parent_KPI', kpi_name)
        if matches.empty:
            matches = self._search(self.relationship, 'Child_KPI', kpi_name)
        if matches.empty:
            return []
        
        if 'Investigation_Order' in matches.columns:
            matches = matches.sort_values('Investigation_Order')
        
        cols = [c for c in ['Child_KPI', 'Relationship_Type', 'Weight'] if c in matches.columns]
        return matches[cols].to_dict(orient='records')
    
    def get_playbook(self, kpi_name: str) -> list:
        kpi_name = normalize_kpi_name(kpi_name) or kpi_name
        matches = self._search(self.playbook, 'KPI', kpi_name)
        if matches.empty and re.search(r'doctor\s*pms|pms', kpi_name, re.I):
            # Fallback for small KPI name variations in playbook
            for fallback in ['Doctor PMS', 'Doctor PMS %', 'PMS', 'PMS %']:
                matches = self._search(self.playbook, 'KPI', fallback)
                if not matches.empty:
                    break
        if matches.empty:
            return []
        
        cols = [c for c in ['Scenario', 'Threshold', 'Severity', 'Root_Cause_Focus',
                           'Recommended_Investigation', 'Recommended_Action', 'Escalation']
                if c in matches.columns]
        return matches[cols].to_dict(orient='records')
    
    def get_filter_info(self, kpi_name: str) -> dict:
        kpi_name = normalize_kpi_name(kpi_name) or kpi_name
        match = self._search(self.filter_compat, 'KPI_Name', kpi_name)
        if match.empty:
            return {}
        
        row = match.iloc[0]
        return {
            "doctor_filter": row.get('Doctor_Filter', ''),
            "specialty_filter": row.get('Specialty_Filter', ''),
            "date_filter": row.get('Date_Filter', ''),
            "bu_filter": row.get('BU_Filter', ''),
            "payer_filter": row.get('Payer_Filter', '')
        }
    
    def get_scope(self, kpi_name: str) -> dict:
        kpi_name = normalize_kpi_name(kpi_name) or kpi_name
        match = self._search(self.scope, 'KPI_Name', kpi_name)
        if match.empty:
            return {}
        
        row = match.iloc[0]
        return {
            "available_doctor_level": row.get('Available_Doctor_Level', ''),
            "available_bu_level": row.get('Available_BU_Level', ''),
            "lowest_granularity": row.get('Lowest_Granularity', ''),
            "default_display_level": row.get('Default_Display_Level', ''),
            "not_available_message": row.get('Not_Available_Message', '')
        }
    
    def list_all_kpis(self) -> list:
        return self.knowledge_map['KPI_Name'].dropna().tolist() if not self.knowledge_map.empty else []


def create_kb_helper_from_raw(kb_raw: dict):
    """Create KBHelper from the raw dict returned by load_data()."""
    import pandas as pd
    return KBHelper(
        formulas=kb_raw.get('adx_kpi_formula_definition', pd.DataFrame()),
        knowledge_map=kb_raw.get('adx_kpi_knowledge_map', pd.DataFrame()),
        relationship=kb_raw.get('adx_kpi_relationship_map', pd.DataFrame()),
        playbook=kb_raw.get('adx_kpi_investigation_playbook', pd.DataFrame()),
        filter_compat=kb_raw.get('adx_kpi_filter_compatibility', pd.DataFrame()),
        scope=kb_raw.get('adx_dim_kpi_scope', pd.DataFrame()),
    )
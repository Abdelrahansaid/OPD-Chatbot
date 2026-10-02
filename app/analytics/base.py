# =================================================================
# app/analytics/base.py — OPD Analytics Engine (Phase 3 Fixed)
# =================================================================
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any
from app.config import BU_LABELS, THRESHOLDS, RANKING_KPI_MAP

class OPDAnalytics:
    """Analytics engine for OPD KPI analysis."""
    
    MONTH_NAMES = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.available_years = sorted(df['Year'].unique().tolist())
        self.available_branches = sorted(df['BU'].unique().tolist())
        self.available_doctors = sorted(df['Doctor Name'].unique().tolist())
    
    def _apply_multi_filter(self, d: pd.DataFrame, values: Optional[Any], column: str) -> pd.DataFrame:
        """Apply filter that supports single value or list of values."""
        if values is None:
            return d
        
        # Convert to list if string/int
        if isinstance(values, (str, int)):
            values = [values]
        elif not isinstance(values, (list, tuple)):
            return d
        
        # Clean and normalize
        clean_vals = [str(v).strip().upper() for v in values if v]
        if not clean_vals:
            return d
        
        # Apply filter
        return d[d[column].str.strip().str.upper().isin(clean_vals)]
    
    # ── Filter Helper ──────────────────────────────────────────
    def filter(self, bu: Optional[Any] = None, doctor: Optional[str] = None, 
               year: Optional[Any] = None, month: Optional[int] = None) -> pd.DataFrame:
        """Apply filters with robust handling of invalid values.

        `bu` and `year` may be single values or lists. If a list is provided,
        the filter uses .isin() semantics.
        """
        d = self.df.copy()
        INVALID = {'UNKNOWN', 'ALL', 'NONE', 'N/A', 'NULL', ''}

        # Business Unit filter: support single string or list using helper
        if bu:
            if isinstance(bu, (list, tuple, set)):
                bu_list = [str(x).strip() for x in bu if str(x).strip() and str(x).upper().strip() not in INVALID]
                if bu_list:
                    d = self._apply_multi_filter(d, bu_list, 'BU')
            else:
                if str(bu).upper().strip() not in INVALID:
                    d = self._apply_multi_filter(d, [bu], 'BU')

        # Doctor filter (string search)
        if doctor and str(doctor).upper().strip() not in INVALID:
            d = d[d['Doctor Name'].str.lower().str.contains(str(doctor).lower(), na=False)]

        # Year filter: support single int or list
        if year:
            try:
                if isinstance(year, (list, tuple, set)):
                    year_list = [int(y) for y in year]
                    d = d[d['Year'].isin(year_list)]
                else:
                    d = d[d['Year'] == int(year)]
            except (ValueError, TypeError):
                pass

        # Month (single)
        if month:
            try:
                d = d[d['Month No'] == int(month)]
            except (ValueError, TypeError):
                pass

        return d
    
    # ── Status Classifier ──────────────────────────────────────
    @staticmethod
    def classify(value: float, warning: float = 80, critical: float = 60) -> str:
        if pd.isna(value):
            return "⚪ N/A"
        if value >= warning:
            return "✅ Good"
        if value >= critical:
            return "⚠️ Warning"
        return "🚨 Critical"
    
    # ── Composite Score Calculation ────────────────────────────
    def compute_composite_scores(self, bu: Optional[str] = None, year: Optional[int] = None, 
                               month: Optional[int] = None) -> List[Dict]:
        """Calculate composite excellence score for doctors."""
        d = self.filter(bu=bu, year=year, month=month)
        if d.empty:
            return []
        
        # Aggregate by doctor. When no BU is selected, keep same-name doctors
        # separate per branch because the test dataset reuses first names.
        group_cols = ['Doctor Name'] if bu else ['BU', 'Doctor Name']
        grp = d.groupby(group_cols).agg(
            total_revenue=('Total Revenue', 'sum'),
            target_revenue=('Target Revenue', 'sum'),
            credit_revenue=('Credit Revenue', 'sum'),
            cash_revenue=('Cash Revenue', 'sum'),
            avg_pms=('Doctor PMS %', 'mean'),
            avg_coe=('Actual COE Compliance %', 'mean'),
            avg_noshow=('No-Show %', 'mean'),
            total_leakage=('Total Leakage Revenue Losses', 'sum'),
            avg_retention=('Patient Retention %', 'mean'),
            avg_service_leakage=('Service Leakage %', 'mean'),
            total_cases=('No. Cases', 'sum'),
            total_cancelled=('No. Cancelled Clinics', 'sum'),
        ).reset_index()
        if not bu and 'BU' in grp.columns:
            grp['Doctor Name'] = grp['Doctor Name'].astype(str) + ' (' + grp['BU'].astype(str) + ')'
        
        # Calculate derived metrics
        grp['revenue_ach_pct'] = (grp['total_revenue'] / grp['target_revenue'] * 100).round(1)
        grp['avg_pms_pct'] = (grp['avg_pms'] * 100).round(1)
        grp['avg_coe_pct'] = (grp['avg_coe'] * 100).round(1)
        grp['avg_noshow_pct'] = (grp['avg_noshow'] * 100).round(1)
        grp['avg_retention_pct'] = (grp['avg_retention'] * 100).round(1)
        
        # Leakage score (lower is better)
        max_lk = grp['total_leakage'].max()
        grp['leakage_score'] = ((1 - grp['total_leakage'] / max_lk) * 100 if max_lk > 0 else 100).round(1)
        grp['noshow_score'] = (100 - grp['avg_noshow_pct']).clip(0, 100).round(1)
        
        # Composite excellence score
        grp['excellence_score'] = (
            grp['revenue_ach_pct'] * 0.35 +
            grp['avg_coe_pct'] * 0.25 +
            grp['avg_pms_pct'] * 0.20 +
            grp['noshow_score'] * 0.10 +
            grp['leakage_score'] * 0.10
        ).round(1)
        
        # Status indicators
        grp['revenue_status'] = grp['revenue_ach_pct'].apply(lambda x: self.classify(x, 90, 70))
        grp['pms_status'] = grp['avg_pms_pct'].apply(lambda x: self.classify(x, 80, 60))
        grp['coe_status'] = grp['avg_coe_pct'].apply(lambda x: self.classify(x, 80, 60))
        grp['excellence_status'] = grp['excellence_score'].apply(lambda x: self.classify(x, 80, 65))
        grp['noshow_status'] = grp['avg_noshow_pct'].apply(
            lambda x: '✅ Good' if x <= 15 else ('⚠️ Warning' if x <= 25 else '🚨 Critical')
        )
        
        # Round numeric columns
        for col in ['total_revenue', 'target_revenue', 'total_leakage', 'credit_revenue', 'cash_revenue']:
            if col in grp.columns:
                grp[col] = grp[col].round(2)
        
        return grp.to_dict(orient='records')
    
    # ── Dynamic Ranking ────────────────────────────────────────
    def dynamic_ranking(self, rank_by: str = 'composite', bu: Optional[str] = None, 
                       year: Optional[int] = None, month: Optional[int] = None, 
                       top_n: int = 5, ascending: Optional[bool] = None) -> Dict:
        """Rank doctors by specified KPI."""
        data = self.compute_composite_scores(bu=bu, year=year, month=month)
        if not data:
            return {"error": "No data found"}
        
        # Map rank_by to column
        key = str(rank_by).lower().strip()
        config = next(
            ((col, asc, lbl) for kw, (col, asc, lbl) in RANKING_KPI_MAP.items() if kw in key),
            ("excellence_score", True, "Composite Excellence Score")
        )
        col, asc_def, label = config
        sort_asc = ascending if ascending is not None else asc_def
        
        # Sort and limit
        data_sorted = sorted(data, key=lambda x: x.get(col, 0), reverse=(not sort_asc))[:top_n]
        for i, r in enumerate(data_sorted):
            r['rank'] = i + 1
        
        return {
            "ranked_by": label,
            "ranking": data_sorted,
            "top_performer": data_sorted[0].get('Doctor Name') if data_sorted else None
        }
    
    # ── Best/Worst Doctor ──────────────────────────────────────
    def best_doctor(self, bu: Optional[str] = None, year: Optional[int] = None, 
                   month: Optional[int] = None) -> Dict:
        result = self.dynamic_ranking('composite', bu, year, month, top_n=5, ascending=False)
        return result
    
    def worst_doctor(self, bu: Optional[str] = None, year: Optional[int] = None, 
                    month: Optional[int] = None) -> Dict:
        data = self.compute_composite_scores(bu=bu, year=year, month=month)
        if not data:
            return {"error": "No data found"}
        
        data_sorted = sorted(data, key=lambda x: x.get('excellence_score', 100))
        worst = data_sorted[0]
        
        return {
            "analysis": "Worst performing doctor — Composite Excellence Score",
            "methodology": "Score = Revenue(35%) + COE(25%) + PMS(20%) + Low No-Show(10%) + Low Leakage(10%)",
            "worst_doctor": worst['Doctor Name'],
            "score": worst['excellence_score'],
            "status": worst['excellence_status'],
            "breakdown": {
                "revenue_achievement_pct": worst['revenue_ach_pct'],
                "avg_pms_pct": worst['avg_pms_pct'],
                "avg_coe_pct": worst['avg_coe_pct'],
                "avg_noshow_pct": worst['avg_noshow_pct'],
                "total_leakage": worst['total_leakage']
            },
            "full_ranking": [
                {"rank": i+1, "Doctor Name": r['Doctor Name'], "excellence_score": r['excellence_score'], "status": r['excellence_status']}
                for i, r in enumerate(data_sorted)
            ]
        }
    
    # ── High Revenue + Low COE Detection ───────────────────────
    def high_revenue_low_coe(self, bu: Optional[str] = None, year: Optional[int] = None) -> Dict:
        data = self.compute_composite_scores(bu=bu, year=year)
        if not data:
            return {"error": "No data found"}
        
        revs = sorted([r['total_revenue'] for r in data])
        median_rev = revs[len(revs) // 2] if revs else 0
        
        result = sorted(
            [r for r in data if r['total_revenue'] >= median_rev and r['avg_coe_pct'] < 70],
            key=lambda x: x['total_revenue'], reverse=True
        )
        
        return {
            "analysis": "High revenue (above median) + poor COE (<70%)",
            "doctors_found": len(result),
            "results": result
        }
    
    # ── Low PMS + High Leakage Detection ───────────────────────
    def low_pms_high_leakage(self, bu: Optional[str] = None, year: Optional[int] = None) -> Dict:
        data = self.compute_composite_scores(bu=bu, year=year)
        if not data:
            return {"error": "No data found"}
        
        lks = sorted([r['total_leakage'] for r in data])
        median_lk = lks[len(lks) // 2] if lks else 0
        
        result = sorted(
            [r for r in data if r['avg_pms_pct'] < 75 and r['total_leakage'] >= median_lk],
            key=lambda x: x['total_leakage'], reverse=True
        )
        
        return {
            "analysis": "PMS <75% AND leakage above median",
            "doctors_found": len(result),
            "results": result,
            "business_insight": "Low PMS + High Leakage = priority coaching candidates"
        }
    
    # ── ✅ FIX: Branch Comparison with bu_label & balance_score ─
    def branch_comparison(self, year: Optional[int] = None, month: Optional[int] = None) -> List[Dict]:
        """Compare branches with balance score and labels."""
        d = self.filter(year=year, month=month)
        if d.empty:
            return []
        
        result = d.groupby('BU').agg(
            total_revenue=('Total Revenue', 'sum'),
            target_revenue=('Target Revenue', 'sum'),
            total_cases=('No. Cases', 'sum'),
            avg_pms=('Doctor PMS %', 'mean'),
            total_leakage=('Total Leakage Revenue Losses', 'sum'),
            avg_noshow=('No-Show %', 'mean'),
            total_cancelled=('No. Cancelled Clinics', 'sum'),
            avg_coe=('Actual COE Compliance %', 'mean'),
            avg_retention=('Patient Retention %', 'mean')
        ).reset_index()
        
        # Calculate derived metrics
        result['revenue_achievement_pct'] = (result['total_revenue'] / result['target_revenue'] * 100).round(1)
        result['avg_pms_pct'] = (result['avg_pms'] * 100).round(1)
        result['avg_noshow_pct'] = (result['avg_noshow'] * 100).round(1)
        result['avg_coe_pct'] = (result['avg_coe'] * 100).round(1)
        
        # ✅ FIX: Calculate balance_score
        result['balance_score'] = (
            result['revenue_achievement_pct'] * 0.4 +
            result['avg_coe_pct'] * 0.3 +
            result['avg_pms_pct'] * 0.2 +
            (100 - result['avg_noshow_pct']) * 0.1
        ).round(1)
        
        # ✅ FIX: Add bu_label mapping
        result['bu_label'] = result['BU'].map(BU_LABELS)
        
        # Round numeric columns
        result['total_revenue'] = result['total_revenue'].round(2)
        result['total_leakage'] = result['total_leakage'].round(2)
        
        return result.to_dict(orient='records')
    
    # ── Operational Risks Detection ────────────────────────────
    def top_operational_risks(self, bu: Optional[str] = None, year: Optional[int] = None) -> Dict:
        d = self.filter(bu=bu, year=year)
        if d.empty:
            return {"error": "No data found"}
        
        total_rev = d['Total Revenue'].sum()
        target_rev = d['Target Revenue'].sum()
        rev_ach = round(total_rev / target_rev * 100, 1) if target_rev > 0 else 0
        avg_coe = round(d['Actual COE Compliance %'].mean() * 100, 1)
        avg_ns = round(d['No-Show %'].mean() * 100, 1)
        total_lk = round(d['Total Leakage Revenue Losses'].sum(), 2)
        lk_pct = round(total_lk / total_rev * 100, 1) if total_rev > 0 else 0
        cancelled = int(d['No. Cancelled Clinics'].sum())
        
        risks = []
        if rev_ach < 90:
            risks.append({
                "rank": 1,
                "risk": "Revenue Target Gap",
                "value": f"{rev_ach}% achievement (gap: {round(target_rev - total_rev, 0):,.0f})",
                "severity": "🚨 Critical" if rev_ach < 75 else "⚠️ Warning",
                "action": "Launch revenue recovery plan — analyze case volume and charge per case"
            })
        if avg_coe < 80:
            risks.append({
                "rank": len(risks) + 1,
                "risk": "Low COE Clinical Compliance",
                "value": f"{avg_coe}% (threshold: 80%)",
                "severity": "🚨 Critical" if avg_coe < 60 else "⚠️ Warning",
                "action": "Mandate clinical pathway audits"
            })
        if lk_pct > 4:
            risks.append({
                "rank": len(risks) + 1,
                "risk": "Revenue Leakage",
                "value": f"{lk_pct}% of revenue = {total_lk:,.0f} lost",
                "severity": "⚠️ Warning",
                "action": "Enforce service workflow — identify top leakage doctors"
            })
        if avg_ns > 12:
            risks.append({
                "rank": len(risks) + 1,
                "risk": "Patient No-Show Rate",
                "value": f"{avg_ns}% average",
                "severity": "⚠️ Warning",
                "action": "Activate automated appointment reminders"
            })
        if cancelled > 50:
            risks.append({
                "rank": len(risks) + 1,
                "risk": "Clinic Cancellations",
                "value": f"{cancelled} clinics cancelled",
                "severity": "⚠️ Warning",
                "action": "Review clinic scheduling policy"
            })
        
        return {
            "period": f"Year {year}" if year else "All available data",
            "top_3_risks": risks[:3],
            "data_basis": f"Based on {len(d)} OPD records across {d['BU'].nunique()} branches"
        }
    
    # ── Executive Summary ──────────────────────────────────────
    def executive_summary(self, year: Optional[int] = None, month: Optional[int] = None, 
                         bu: Optional[str] = None) -> Dict:
        d = self.filter(bu=bu, year=year, month=month)
        if d.empty:
            return {"error": "No data found"}
        
        total_rev = d['Total Revenue'].sum()
        target_rev = d['Target Revenue'].sum()
        rev_ach = round(total_rev / target_rev * 100, 1) if target_rev > 0 else 0
        
        period = (
            f"{self.MONTH_NAMES.get(int(month), '')} {year}" if month and year 
            else str(year) if year else "All available data"
        )
        
        return {
            "period": period,
            "scope": bu if bu else "All branches (ASH+SMH+HJH)",
            "revenue": {
                "total": round(total_rev, 2),
                "target": round(target_rev, 2),
                "achievement_pct": rev_ach,
                "gap": round(total_rev - target_rev, 2),
                "status": self.classify(rev_ach, 90, 70)
            },
            "volume": {
                "total_cases": int(d['No. Cases'].sum()),
                "target_cases": int(d['Target No. cases'].sum()),
                "achievement_pct": round(d['No. Cases'].sum() / d['Target No. cases'].sum() * 100, 1) if d['Target No. cases'].sum() > 0 else None
            },
            "quality": {
                "avg_coe_pct": round(d['Actual COE Compliance %'].mean() * 100, 1),
                "coe_status": self.classify(round(d['Actual COE Compliance %'].mean() * 100, 1), 80, 60),
                "avg_pms_pct": round(d['Doctor PMS %'].mean() * 100, 1),
                "pms_status": self.classify(round(d['Doctor PMS %'].mean() * 100, 1), 80, 60)
            },
            "operational_issues": {
                "avg_noshow_pct": round(d['No-Show %'].mean() * 100, 1),
                "total_leakage": round(d['Total Leakage Revenue Losses'].sum(), 2),
                "leakage_pct": round(d['Total Leakage Revenue Losses'].sum() / total_rev * 100, 1) if total_rev > 0 else 0,
                "cancelled_clinics": int(d['No. Cancelled Clinics'].sum()),
                "cancellation_loss": round(d['Total Losses Revenue_Cancellation_Modification'].sum(), 2)
            }
        }
    
    # ── Standard KPI Methods (unchanged, shown for completeness) ─
    def revenue_vs_target(self, bu: Optional[str] = None, doctor: Optional[str] = None, 
                         year: Optional[int] = None, month: Optional[int] = None) -> Dict:
        d = self.filter(bu, doctor, year, month)
        if d.empty:
            return {"error": "No data found"}
        
        tr = d['Total Revenue'].sum()
        tg = d['Target Revenue'].sum()
        tc = d['No. Cases'].sum()
        tgc = d['Target No. cases'].sum()
        ach = round(tr / tg * 100, 1) if tg > 0 else None
        
        return {
            "total_revenue": round(tr, 2),
            "target_revenue": round(tg, 2),
            "revenue_achievement_pct": ach,
            "revenue_status": self.classify(ach or 0, 90, 70),
            "revenue_gap": round(tr - tg, 2),
            "total_cases": int(tc),
            "target_cases": int(tgc),
            "cases_achievement_pct": round(tc / tgc * 100, 1) if tgc > 0 else None,
            "cases_gap": int(tc - tgc),
            "filters_applied": {"bu": bu, "doctor": doctor, "year": year, "month": month}
        }

    def revenue_gap_analysis(self, bu: Optional[str] = None, doctor: Optional[str] = None,
                             year: Optional[int] = None, month: Optional[int] = None) -> Dict:
        base = self.revenue_vs_target(bu, doctor, year, month)
        if base.get("error"):
            return base

        d = self.filter(bu, doctor, year, month)
        total_rev = d['Total Revenue'].sum()
        total_leak = d['Total Leakage Revenue Losses'].sum()
        avg_pms = round(d['Doctor PMS %'].mean() * 100, 1)
        avg_coe = round(d['Actual COE Compliance %'].mean() * 100, 1)
        avg_ns = round(d['No-Show %'].mean() * 100, 1)
        leakage_pct = round(total_leak / total_rev * 100, 1) if total_rev > 0 else None
        case_gap = int(base['cases_gap'])

        top_leakage = d.groupby('Doctor Name')['Total Leakage Revenue Losses']\
            .sum().sort_values(ascending=False).head(3).reset_index()
        top_leakage['Total Leakage Revenue Losses'] = top_leakage['Total Leakage Revenue Losses'].round(2)

        low_perf = d.groupby('Doctor Name').agg(
            avg_pms_pct=('Doctor PMS %', 'mean'),
            avg_coe_pct=('Actual COE Compliance %', 'mean'),
            avg_noshow_pct=('No-Show %', 'mean')
        ).reset_index()
        low_perf['avg_pms_pct'] = (low_perf['avg_pms_pct'] * 100).round(1)
        low_perf['avg_coe_pct'] = (low_perf['avg_coe_pct'] * 100).round(1)
        low_perf['avg_noshow_pct'] = (low_perf['avg_noshow_pct'] * 100).round(1)
        low_perf = low_perf.sort_values(by=['avg_pms_pct', 'avg_coe_pct']).head(3)

        drivers = []
        if leakage_pct is not None and leakage_pct > THRESHOLDS['leakage_pct_warning']:
            drivers.append({
                "driver": "High revenue leakage",
                "value": f"{leakage_pct}% of revenue is lost to leakage",
                "priority": "high"
            })
        if avg_pms < 80:
            drivers.append({
                "driver": "Low doctor PMS score",
                "value": f"Average PMS is {avg_pms}%",
                "priority": "medium"
            })
        if avg_coe < 80:
            drivers.append({
                "driver": "Below-target COE compliance",
                "value": f"Average COE is {avg_coe}%",
                "priority": "medium"
            })
        if case_gap < 0:
            drivers.append({
                "driver": "Lower case volume than plan",
                "value": f"Case volume is {abs(case_gap)} below target",
                "priority": "medium"
            })
        if not drivers:
            drivers.append({
                "driver": "No strong driver identified",
                "value": "The dataset does not show a single dominant gap driver.",
                "priority": "low"
            })

        return {
            **base,
            "average_pms_pct": avg_pms,
            "average_coe_pct": avg_coe,
            "average_noshow_pct": avg_ns,
            "leakage_pct_of_revenue": leakage_pct,
            "top_leakage_doctors": top_leakage.to_dict(orient='records'),
            "potential_low_performance_doctors": low_perf.to_dict(orient='records'),
            "revenue_gap_drivers": drivers
        }

    def revenue_breakdown(self, bu: Optional[str] = None, doctor: Optional[str] = None, 
                         year: Optional[int] = None, month: Optional[int] = None) -> Dict:
        d = self.filter(bu, doctor, year, month)
        if d.empty:
            return {"error": "No data found"}
        
        total = d['Total Revenue'].sum()
        credit = d['Credit Revenue'].sum()
        cash = d['Cash Revenue'].sum()
        
        return {
            "definition_credit": "Credit Revenue = Revenue from insured patients",
            "definition_cash": "Cash Revenue = Revenue from self-paying patients (no insurance)",
            "total_revenue": round(total, 2),
            "credit_revenue": round(credit, 2),
            "cash_revenue": round(cash, 2),
            "credit_share_pct": round(credit / total * 100, 1) if total > 0 else None,
            "cash_share_pct": round(cash / total * 100, 1) if total > 0 else None
        }
    
    def leakage_analysis(self, bu: Optional[str] = None, doctor: Optional[str] = None, 
                        year: Optional[int] = None, month: Optional[int] = None) -> Dict:
        d = self.filter(bu, doctor, year, month)
        if d.empty:
            return {"error": "No data found"}
        
        by_doc = d.groupby('Doctor Name').agg(
            leakage_losses=('Total Leakage Revenue Losses', 'sum'),
            service_leakage_pct=('Service Leakage %', 'mean')
        ).sort_values('leakage_losses', ascending=False).head(5).round(3).reset_index().to_dict(orient='records')
        
        total = d['Total Revenue'].sum()
        
        return {
            "total_leakage_losses": round(d['Total Leakage Revenue Losses'].sum(), 2),
            "avg_service_leakage_pct": round(d['Service Leakage %'].mean() * 100, 2),
            "leakage_as_pct_of_revenue": round(d['Total Leakage Revenue Losses'].sum() / total * 100, 2) if total > 0 else None,
            "top_leakage_by_doctor": by_doc
        }

    def coe_compliance_analysis(self, bu: Optional[str] = None, doctor: Optional[str] = None, 
                                year: Optional[int] = None, month: Optional[int] = None) -> Dict:
        d = self.filter(bu=bu, doctor=doctor, year=year, month=month)
        if d.empty:
            return {"error": "No data found"}

        avg_coe = round(d['Actual COE Compliance %'].mean() * 100, 1)
        top_bu = d.groupby('BU')['Actual COE Compliance %'].mean().round(3).sort_values(ascending=False).head(5).reset_index()
        top_bu['avg_coe_pct'] = (top_bu['Actual COE Compliance %'] * 100).round(1)
        group_cols = ['Doctor Name'] if bu else ['BU', 'Doctor Name']
        by_doc = d.groupby(group_cols)['Actual COE Compliance %'].mean().sort_values(ascending=False).reset_index()
        by_doc['Actual COE Compliance %'] = (by_doc['Actual COE Compliance %'] * 100).round(1)
        if not bu and 'BU' in by_doc.columns:
            by_doc['Doctor Name'] = by_doc['Doctor Name'].astype(str) + ' (' + by_doc['BU'].astype(str) + ')'

        return {
            "average_coe_pct": avg_coe,
            "coe_status": self.classify(avg_coe, 80, 60),
            "top_coe_by_branch": top_bu[['BU', 'avg_coe_pct']].to_dict(orient='records'),
            "by_doctor": by_doc.to_dict(orient='records'),
            "filters_applied": {"bu": bu, "doctor": doctor, "year": year, "month": month}
        }
    
    def pms_ranking(self, bu: Optional[str] = None, year: Optional[int] = None, 
                   month: Optional[int] = None, top_n: Optional[int] = None) -> List[Dict]:
        d = self.filter(bu=bu, year=year, month=month)
        if d.empty:
            return []
        
        group_cols = ['Doctor Name'] if bu else ['BU', 'Doctor Name']
        result = d.groupby(group_cols)['Doctor PMS %'].mean().sort_values(ascending=False).reset_index()
        result['Doctor PMS %'] = (result['Doctor PMS %'] * 100).round(1)
        if not bu and 'BU' in result.columns:
            result['Doctor Name'] = result['Doctor Name'].astype(str) + ' (' + result['BU'].astype(str) + ')'
        result = result.rename(columns={'Doctor PMS %': 'Avg PMS Score (%)'})
        result['Status'] = result['Avg PMS Score (%)'].apply(lambda x: self.classify(x, 80, 60))
        
        return result.head(top_n).to_dict(orient='records') if top_n else result.to_dict(orient='records')
    
    def noshow_analysis(self, bu: Optional[str] = None, doctor: Optional[str] = None, 
                       year: Optional[int] = None, month: Optional[int] = None) -> Dict:
        d = self.filter(bu, doctor, year, month)
        if d.empty:
            return {"error": "No data found"}
        
        by_doc = d.groupby('Doctor Name')['No-Show %'].mean().sort_values(ascending=False).reset_index()
        by_doc['No-Show %'] = (by_doc['No-Show %'] * 100).round(1)
        by_doc['severity'] = by_doc['No-Show %'].apply(
            lambda x: '🚨 Critical' if x > 25 else ('⚠️ High' if x > 15 else '✅ Normal')
        )
        
        return {
            "overall_avg_noshow_pct": round(d['No-Show %'].mean() * 100, 1),
            "threshold_alert": "Critical if > 25%",
            "by_doctor": by_doc.to_dict(orient='records')
        }
    
    def doctor_multi_branch_summary(self, doctor_name: str, year: Optional[int] = None, bu: Optional[str] = None) -> Dict:
        """Get performance summary for a doctor across multiple branches."""
        d = self.filter(bu=bu, doctor=doctor_name, year=year)
        if d.empty:
            if bu:
                return {"error": f"Doctor '{doctor_name}' not found for BU '{bu}' in dataset"}
            return {"error": f"Doctor '{doctor_name}' not found in dataset"}
        
        branches = sorted(d['BU'].unique().tolist())
        per_bu = []
        
        for bu_name in branches:
            bd = d[d['BU'] == bu_name]
            tr = bd['Total Revenue'].sum()
            tg = bd['Target Revenue'].sum()
            slots = bd['No. Planned booking Slots'].sum()
            
            per_bu.append({
                "bu": bu_name,
                "bu_label": BU_LABELS.get(bu_name, bu_name),
                "total_revenue": round(tr, 2),
                "target_revenue": round(tg, 2),
                "revenue_achievement_pct": round(tr / tg * 100, 1) if tg > 0 else None,
                "revenue_status": self.classify(round(tr / tg * 100, 1) if tg > 0 else 0, 90, 70),
                "total_cases": int(bd['No. Cases'].sum()),
                "avg_pms_score_pct": round(bd['Doctor PMS %'].mean() * 100, 1),
                "pms_status": self.classify(round(bd['Doctor PMS %'].mean() * 100, 1), 80, 60),
                "avg_coe_compliance_pct": round(bd['Actual COE Compliance %'].mean() * 100, 1),
                "coe_status": self.classify(round(bd['Actual COE Compliance %'].mean() * 100, 1), 80, 60),
                "avg_noshow_pct": round(bd['No-Show %'].mean() * 100, 1),
                "total_leakage_losses": round(bd['Total Leakage Revenue Losses'].sum(), 2),
                "avg_retention_pct": round(bd['Patient Retention %'].mean() * 100, 1),
                "booking_utilization_pct": round(bd['No. Booking'].sum() / slots * 100, 1) if slots > 0 else None
            })
        
        tr_all = d['Total Revenue'].sum()
        tg_all = d['Target Revenue'].sum()
        
        return {
            "important_note": "⚠️ Same doctor name may appear in multiple BUs. Results split per BU for clarity.",
            "per_bu_performance": per_bu,
            "overall_aggregated": {
                "doctor_name": d['Doctor Name'].iloc[0],
                "branches_found": branches,
                "branches_labels": [BU_LABELS.get(b, b) for b in branches],
                "total_revenue_all_bus": round(tr_all, 2),
                "target_revenue_all_bus": round(tg_all, 2),
                "revenue_achievement_pct": round(tr_all / tg_all * 100, 1) if tg_all > 0 else None,
                "total_cases_all_bus": int(d['No. Cases'].sum()),
                "avg_pms_score_pct": round(d['Doctor PMS %'].mean() * 100, 1),
                "avg_coe_compliance_pct": round(d['Actual COE Compliance %'].mean() * 100, 1),
                "avg_noshow_pct": round(d['No-Show %'].mean() * 100, 1),
                "total_leakage_losses": round(d['Total Leakage Revenue Losses'].sum(), 2)
            }
        }
    
    def monthly_trend(self, kpi_col: str = 'Total Revenue', bu: Optional[str] = None, 
                     doctor: Optional[str] = None, year: Optional[int] = None) -> List[Dict]:
        d = self.filter(bu=bu, doctor=doctor, year=year)
        if d.empty:
            return []
        
        result = d.groupby(['Year', 'Month No'])[kpi_col].sum().reset_index().sort_values(['Year', 'Month No'])
        result[kpi_col] = result[kpi_col].round(2)
        result['Month'] = result['Month No'].map(self.MONTH_NAMES)
        
        return result[['Year', 'Month No', 'Month', kpi_col]].to_dict(orient='records')

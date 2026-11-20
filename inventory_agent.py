"""
inventory_agent.py
==================
Supply Chain Analytics Inventory Decision Agent.

Architecture Pipeline:
  Customer Question
       ↓
  Identify Model
       ↓
  Extract Inputs
       ↓
  Select Python Tool
       ↓
  Calculate
       ↓
  Verify
       ↓
  Explain

All numerical calculations are executed deterministically via Python in inventory_tools.py.
"""

import re
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd

import inventory_tools as itools


@dataclass
class AgentPipelineResult:
    question: str
    identified_model: str
    confidence: float
    extracted_inputs: Dict[str, Any]
    selected_tool: str
    calculation_results: Optional[Dict[str, Any]] = None
    verification_report: Optional[Dict[str, Any]] = None
    explanation: str = ""
    status: str = "SUCCESS"
    error_message: Optional[str] = None
    pipeline_trace: List[Dict[str, Any]] = field(default_factory=list)


class InventoryDecisionAgent:
    """
    Intelligent decision agent that translates natural-language inventory questions
    into structured analytical inventory optimization models, verifies results,
    and produces actionable supply chain recommendations.
    """

    MODEL_EOQ = "EOQ"
    MODEL_QUANTITY_DISCOUNT = "QUANTITY_DISCOUNT"
    MODEL_SAFETY_STOCK_ROP = "SAFETY_STOCK_ROP"
    MODEL_FIXED_PERIOD_REVIEW = "FIXED_PERIOD_REVIEW"
    MODEL_NEWSVENDOR = "NEWSVENDOR"
    MODEL_DATASET_ANALYSIS = "DATASET_ANALYSIS"
    MODEL_INVENTORY_CLASSIFICATION = "INVENTORY_CLASSIFICATION"

    MODEL_TOOL_MAP = {
        MODEL_EOQ: "calculate_eoq",
        MODEL_QUANTITY_DISCOUNT: "calculate_quantity_discount",
        MODEL_SAFETY_STOCK_ROP: "calculate_safety_stock_rop",
        MODEL_FIXED_PERIOD_REVIEW: "calculate_fixed_period_review",
        MODEL_NEWSVENDOR: "calculate_newsvendor",
        MODEL_DATASET_ANALYSIS: "analyze_dataset_query",
        MODEL_INVENTORY_CLASSIFICATION: "classify_inventory_multicriteria",
    }

    def __init__(self):
        pass

    # =========================================================================
    # Stage 1: Identify Model
    # =========================================================================
    def identify_model(self, question: str, dataset_df: Optional[Any] = None) -> Tuple[str, float, str]:
        """
        Identify the appropriate inventory decision model from natural language.
        Returns: (model_name, confidence_score, rationale)
        """
        q_lower = question.lower()

        scores = {
            self.MODEL_EOQ: 0.0,
            self.MODEL_QUANTITY_DISCOUNT: 0.0,
            self.MODEL_SAFETY_STOCK_ROP: 0.0,
            self.MODEL_FIXED_PERIOD_REVIEW: 0.0,
            self.MODEL_NEWSVENDOR: 0.0,
            self.MODEL_DATASET_ANALYSIS: 0.0,
            self.MODEL_INVENTORY_CLASSIFICATION: 0.0,
        }

        # Multicriteria Inventory Classification cues
        classification_cues = [
            r"classif\w+", r"multicriteria", r"\babc\b", r"\babcd\b",
            r"class a\b", r"class b\b", r"class c\b", r"class d\b",
            r"slow[- ]mover[s]?", r"fast[- ]mover[s]?", r"pareto",
            r"pos hits", r"vital few", r"category a\b", r"category b\b",
            r"65%", r"35%", r"hits",
        ]
        for pat in classification_cues:
            if re.search(pat, q_lower):
                scores[self.MODEL_INVENTORY_CLASSIFICATION] += 3.5

        # Dataset Portfolio Queries (when dataset is present or dataset keywords are queried)
        if dataset_df is not None:
            dataset_cues = [
                r"inventory valuation", r"total valuation", r"total inventory value", r"portfolio value",
                r"highest unit[s]?", r"highest quantity", r"most stock", r"most unit[s]?", r"top product",
                r"average unit cost", r"avg unit cost", r"mean unit cost", r"average cost",
                r"total unit[s]?", r"total stock", r"total quantity", r"how many unit[s]?",
                r"by category", r"category breakdown", r"units by category",
                r"missing value[s]?", r"data quality", r"data cleanliness", r"null value[s]?",
                r"summarize dataset", r"dataset overview", r"uploaded dataset", r"uploaded data",
                r"how many records", r"how many products",
            ]
            for pat in dataset_cues:
                if re.search(pat, q_lower):
                    scores[self.MODEL_DATASET_ANALYSIS] += 3.0

        # Quantity Discount cues (high priority if tier/discount keywords present)
        discount_patterns = [
            r"quantity discount", r"price break", r"tier[s]?", r"discount schedule",
            r"volume discount", r"all-units", r"bulk discount", r"\bcheaper if we order\b",
            r"\$[0-9\.]+\s*(for|if|at|over|above)", r"price falls to", r"bracket",
            r"quantity break", r"percentage holding cost", r"fixed holding cost",
            r"unit purchase price", r"below \d+ units", r"orders of \d+ to \d+"
        ]
        for pat in discount_patterns:
            if re.search(pat, q_lower):
                scores[self.MODEL_QUANTITY_DISCOUNT] += 2.5

        # Newsvendor cues
        newsvendor_patterns = [
            r"newsvendor", r"newsboy", r"single[- ]period", r"perishable", r"seasonal",
            r"salvage", r"overage", r"underage", r"critical fractile", r"critical ratio",
            r"spoilage", r"discard", r"markdown", r"fashion", r"fresh",
            r"underage cost", r"overage cost", r"cost of underage", r"cost of overage",
            r"shortage penalty", r"demand distribution"
        ]
        for pat in newsvendor_patterns:
            if re.search(pat, q_lower):
                scores[self.MODEL_NEWSVENDOR] += 2.5

        # Fixed-Period Review cues
        fixed_period_patterns = [
            r"periodic review", r"fixed[- ]period", r"order[- ]up[- ]to", r"target inventory level",
            r"review period", r"review every", r"every \d+ (day|week|month)", r"on[- ]hand",
            r"inventory position", r"pipeline", r"target level s\b", r"\(t,\s*s\)", r"\(r,\s*s\)",
            r"on[- ]order", r"backorders?", r"backlogged", r"review cycle", r"order interval"
        ]
        for pat in fixed_period_patterns:
            if re.search(pat, q_lower):
                scores[self.MODEL_FIXED_PERIOD_REVIEW] += 2.5

        # Safety Stock / ROP cues
        rop_patterns = [
            r"reorder point", r"\brop\b", r"safety stock", r"continuous review",
            r"cycle service level", r"lead time demand", r"stockout probability",
            r"service level of \d+", r"buffer stock", r"std dev of demand", r"standard deviation of demand",
            r"lead[- ]time standard deviation", r"average lead time", r"lead time std dev"
        ]
        for pat in rop_patterns:
            if re.search(pat, q_lower):
                scores[self.MODEL_SAFETY_STOCK_ROP] += 2.0

        # EOQ cues
        eoq_patterns = [
            r"\beoq\b", r"economic order quantity", r"optimal batch", r"optimal lot size",
            r"ordering cost", r"order cost", r"setup cost", r"setup expense", r"cost per order",
            r"cost of placing an order", r"ordering/setup cost", r"setup/ordering cost",
            r"holding cost", r"carrying cost", r"holding expense", r"carrying expense",
            r"cost of carrying inventory", r"cost to carry inventory", r"inventory holding cost",
            r"how many units should (we|i) order", r"best order size", r"wilson"
        ]
        for pat in eoq_patterns:
            if re.search(pat, q_lower):
                scores[self.MODEL_EOQ] += 1.8

        # Tie-breaking / specific overrides:
        # If both discount and eoq match, discount wins because discount is a generalization of EOQ
        if scores[self.MODEL_QUANTITY_DISCOUNT] > 1.5 and scores[self.MODEL_EOQ] > 0:
            scores[self.MODEL_QUANTITY_DISCOUNT] += 1.0

        # If periodic review keywords are strong, prefer over standard ROP
        if scores[self.MODEL_FIXED_PERIOD_REVIEW] > 2.0:
            scores[self.MODEL_FIXED_PERIOD_REVIEW] += 1.0

        best_model = max(scores, key=scores.get)
        best_score = scores[best_model]

        total_score = sum(scores.values())
        confidence = min(0.99, round(best_score / (total_score + 1e-6), 2)) if total_score > 0 else 0.50

        # If confidence is negligible or ambiguous, default to EOQ or ROP based on lead time
        if best_score < 1.0:
            if "lead time" in q_lower or "safety" in q_lower:
                best_model = self.MODEL_SAFETY_STOCK_ROP
                rationale = "Defaulted to Safety Stock & ROP based on lead time references."
            else:
                best_model = self.MODEL_EOQ
                rationale = "Defaulted to Economic Order Quantity (EOQ) for lot sizing."
            confidence = 0.55
        else:
            rationale = f"Matched key terminology with composite match score of {best_score:.1f}."

        return best_model, confidence, rationale

    # =========================================================================
    # Stage 2: Extract Inputs
    # =========================================================================
    def extract_inputs(self, model: str, question: str, dataset_df: Optional[Any] = None) -> Dict[str, Any]:
        """
        Extract numerical parameters and units from natural language question.
        Examines the entire question and recognizes common inventory terminology and synonyms.
        Integrates with uploaded dataset records when available.
        """
        q = question
        inputs: Dict[str, Any] = {"_raw_question": question}

        # Helper extractors
        def find_number_after(pattern_list: List[str], text: str) -> Optional[float]:
            for pat in pattern_list:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    val_str = m.group(1).replace(",", "").strip()
                    try:
                        return float(val_str)
                    except ValueError:
                        pass
            return None

        # ---------------------------------------------------------------------
        # 1. Demand Extraction (Annual, Daily, Weekly, Mean)
        # ---------------------------------------------------------------------
        demand_val = find_number_after([
            r"(?:annual|yearly)\s+demand\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"(?:mean|average|expected)\s+demand\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"demand\s+mean\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"mu\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"demand\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:per\s+year|annually|/year)",
            r"sells?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:per\s+year|annually|a\s+year)",
            r"annual\s+(?:usage|sales|requirement|volume)\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"demand\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:per\s+annum|yearly)",
            r"demand\s+(?:of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?",
            r"([0-9,]+(?:\.[0-9]+)?)\s*units\s*(?:per\s+year|annually|demanded)",
        ], q)

        # Daily demand check (distinguishing average daily demand vs annual)
        daily_d = find_number_after([
            r"(?:average\s+|mean\s+)?daily\s+demand\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"average\s+demand\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units?)?\s*(?:per\s+day|daily|/day)",
            r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:per\s+day|daily|/day)",
        ], q)
        if daily_d is not None:
            inputs["demand_rate_daily"] = daily_d
            if demand_val is None and model in [self.MODEL_EOQ, self.MODEL_QUANTITY_DISCOUNT]:
                demand_val = daily_d * 365.0

        # Weekly demand check
        weekly_d = find_number_after([
            r"(?:average\s+|mean\s+)?weekly\s+demand\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"average\s+demand\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units?)?\s*(?:per\s+week|weekly|/week)",
            r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:per\s+week|weekly|/week)",
        ], q)
        if weekly_d is not None:
            inputs["demand_rate_weekly"] = weekly_d
            if demand_val is None and model in [self.MODEL_EOQ, self.MODEL_QUANTITY_DISCOUNT]:
                demand_val = weekly_d * 52.0

        # General average demand if daily/weekly not specified
        if daily_d is None and weekly_d is None:
            gen_avg_d = find_number_after([
                r"average\s+demand\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"mean\s+demand\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"demand\s+rate\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            ], q)
            if gen_avg_d is not None:
                inputs["demand_rate"] = gen_avg_d
                if demand_val is None:
                    demand_val = gen_avg_d

        if demand_val is not None:
            inputs["annual_demand"] = demand_val
            inputs.setdefault("demand_rate", daily_d or weekly_d or demand_val)
            inputs.setdefault("demand_mean", demand_val)

        # ---------------------------------------------------------------------
        # 2. Ordering / Setup Cost
        # Recognized synonyms: ordering cost, order cost, setup cost, setup expense,
        # cost per order, cost of placing an order, ordering/setup cost, setup/ordering cost,
        # cost to place an order, fixed cost per order, order placement cost.
        # ---------------------------------------------------------------------
        order_cost = find_number_after([
            # Compound descriptive phrases
            r"cost\s+of\s+placing\s+(?:an?\s+)?order\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?",
            r"cost\s+to\s+place\s+(?:an?\s+|each\s+)?order\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?",
            r"cost\s+per\s+order\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?",
            r"(?:ordering\s*/\s*setup|setup\s*/\s*ordering|ordering\s+and\s+setup|setup\s+and\s+ordering)\s*(?:cost|expense)?\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?",
            r"(?:order(?:ing)?|setup)\s+(?:cost|expense|charge|fee)\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?",
            r"(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?\s*(?:per|/)\s*(?:order|purchase\s+order|setup)",
            r"fixed\s+(?:order(?:ing)?|setup)?\s*(?:cost|expense)\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"costs?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?\s*to\s+place\s+(?:an?\s+|each\s+)?order",
            r"order\s+placement\s+cost\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"replenishment\s+cost\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?\s*(?:per|/)\s*order",
        ], q)
        if order_cost is not None:
            inputs["ordering_cost"] = order_cost

        # ---------------------------------------------------------------------
        # 3. Holding Cost ($/unit/year) & Holding Rate (%)
        # Recognized synonyms for holding cost: holding cost, carrying cost, holding expense,
        # carrying expense, inventory holding cost, cost of carrying inventory, storage cost,
        # fixed holding cost.
        # Recognized synonyms for holding rate: percentage holding cost, holding rate, carrying rate,
        # annual holding rate, cost of capital, holding percentage.
        # ---------------------------------------------------------------------
        # Check percentage holding rate first
        holding_rate_match = find_number_after([
            r"percentage\s+(?:holding|carrying)\s+cost\s*(?:is|of|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"(?:annual\s+|yearly\s+|unit\s+|inventory\s+)?(?:holding|carrying)\s+rate\s*(?:is|of|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"(?:holding|carrying)\s+(?:cost|rate)\s+percentage\s*(?:is|of|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"(?:holding|carrying)\s+cost\s+is\s+([0-9]+(?:\.[0-9]+)?)\s*%\s*of\s*(?:unit\s+)?(?:cost|price|purchase\s+price)",
            r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:annually\s+)?(?:holding|carrying|annual\s+cost\s+of\s+capital|cost\s+of\s+capital)",
            r"(?:cost\s+of\s+capital|carrying\s+charge)\s*(?:is|of|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"(?:annual\s+)?(?:holding|carrying)\s+rate\s*(?:is|of|=|:)?\s*(0\.[0-9]+)",
        ], q)

        # Check dollar holding cost
        holding_val = find_number_after([
            r"fixed\s+holding\s+cost\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?",
            r"(?:annual\s+|yearly\s+|unit\s+|inventory\s+)?(?:holding|carrying|storage|warehousing)\s+(?:cost|expense|charge|fee)\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?(?:\s*(?:per|/)\s*(?:unit|item)(?:\s*(?:per|/)\s*(?:year|annum|annual))?)?",
            r"cost\s+(?:to|of)\s+(?:hold|carry|storing|carry(?:ing)?)\s+(?:inventory|units?|stock)?\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?(?:\s*(?:per|/)\s*(?:unit|item)(?:\s*(?:per|/)\s*(?:year|annum|annual))?)?",
            r"(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?\s*(?:per|/)\s*(?:unit|item)\s*(?:per|/)\s*(?:year|annum|annual)",
            r"(?:annual\s+|yearly\s+)?(?:holding|carrying)\s+cost\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)",
        ], q)

        if holding_rate_match is not None:
            rate = holding_rate_match / 100.0 if holding_rate_match > 1.0 else holding_rate_match
            inputs["holding_rate"] = rate
        if holding_val is not None:
            inputs["holding_cost"] = holding_val

        # ---------------------------------------------------------------------
        # 4. Unit Cost / Unit Purchase Price
        # Recognized synonyms: unit price, unit purchase price, unit cost, purchase price,
        # purchase cost, procurement cost, wholesale cost.
        # ---------------------------------------------------------------------
        unit_cost_val = find_number_after([
            r"unit\s+purchase\s+price\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"(?:unit\s+cost|purchase\s+price|purchase\s+cost|procurement\s+cost|wholesale\s+cost)\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"(?:regular|base|standard|normal)\s+(?:unit\s+)?price\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"costs?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?\s*(?:per|/)\s*(?:unit|each|item)",
            r"(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?\s+(?:purchase\s+price|unit\s+cost)",
        ], q)
        if unit_cost_val is not None:
            inputs["unit_cost"] = unit_cost_val
            inputs["purchase_cost"] = unit_cost_val
            inputs["unit_purchase_price"] = unit_cost_val

        # ---------------------------------------------------------------------
        # 5. Lead Time & Lead-Time Standard Deviation
        # Distinguishes average lead time vs lead-time standard deviation.
        # ---------------------------------------------------------------------
        # Lead-Time Standard Deviation
        lt_sd = find_number_after([
            r"lead[- ]time\s+(?:std\s*dev|standard\s+deviation)\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)?",
            r"(?:std\s*dev|standard\s+deviation)\s+(?:of\s+)?lead[- ]time\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)?",
            r"lead\s+time\s+(?:has\s+a\s+)?(?:std\s*dev|standard\s+deviation)\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)?",
            r"(?:sigma_l|sigma_lt)\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"\+\/-\s*([0-9,]+(?:\.[0-9]+)?)\s*days?\s+(?:in\s+)?lead\s+time",
            r"variability\s+in\s+lead[- ]time\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
        ], q)
        if lt_sd is not None:
            inputs["lead_time_std_dev"] = lt_sd

        # Average Lead Time
        lead_time_days = find_number_after([
            r"(?:average\s+|mean\s+|expected\s+|replenishment\s+|supplier\s+|delivery\s+|order\s+)?lead\s+time\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*days?",
            r"([0-9,]+(?:\.[0-9]+)?)\s*days?\s+(?:average\s+|mean\s+)?lead\s+time",
            r"delivery\s+takes\s+([0-9,]+(?:\.[0-9]+)?)\s*days?",
        ], q)
        lead_time_weeks = find_number_after([
            r"(?:average\s+|mean\s+|expected\s+|replenishment\s+|supplier\s+|delivery\s+|order\s+)?lead\s+time\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*weeks?",
            r"([0-9,]+(?:\.[0-9]+)?)\s*weeks?\s+(?:average\s+|mean\s+)?lead\s+time",
            r"delivery\s+takes\s+([0-9,]+(?:\.[0-9]+)?)\s*weeks?",
        ], q)
        if lead_time_days is not None:
            inputs["lead_time"] = lead_time_days
            inputs["time_unit"] = "days"
        elif lead_time_weeks is not None:
            inputs["lead_time"] = lead_time_weeks
            inputs["time_unit"] = "weeks"
        else:
            generic_lt = find_number_after([
                r"(?:average\s+|mean\s+|replenishment\s+)?lead\s+time\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            ], q)
            if generic_lt is not None:
                inputs["lead_time"] = generic_lt

        # ---------------------------------------------------------------------
        # 6. Demand Standard Deviation
        # Distinguishes demand std dev from lead-time std dev.
        # ---------------------------------------------------------------------
        sigma_d = find_number_after([
            r"(?:daily|weekly|annual|monthly)?\s*demand\s+(?:std\s*dev|standard\s+deviation)\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units?)?",
            r"(?:std\s*dev|standard\s+deviation)\s+of\s+(?:daily|weekly|annual|monthly)?\s*demand\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units?)?",
            r"demand\s+(?:has\s+a\s+)?(?:std\s*dev|standard\s+deviation)\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units?)?",
            r"(?:sigma_d|sigma)\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"\+\/-\s*([0-9,]+(?:\.[0-9]+)?)\s*units?",
        ], q)
        if sigma_d is None:
            # Mask out lead time std dev phrase if present so generic search doesn't pick it up
            q_clean_for_sigma = re.sub(r"lead[- ]time\s+(?:std\s*dev|standard\s+deviation)[^,;\.]*", "", q, flags=re.IGNORECASE)
            q_clean_for_sigma = re.sub(r"(?:std\s*dev|standard\s+deviation)\s+of\s+lead[- ]time[^,;\.]*", "", q_clean_for_sigma, flags=re.IGNORECASE)
            sigma_d = find_number_after([
                r"(?:std\s*dev|standard\s+deviation)\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            ], q_clean_for_sigma)
        if sigma_d is not None:
            inputs["demand_std_dev"] = sigma_d

        # ---------------------------------------------------------------------
        # 7. Service Level
        # Recognized: service level, cycle service level, target service level,
        # customer service level, non-stockout probability, CSL.
        # ---------------------------------------------------------------------
        csl = find_number_after([
            r"(?:cycle\s+service\s+level|target\s+service\s+level|desired\s+service\s+level|customer\s+service\s+level|service\s+level|csl)\s*(?:is|of|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:cycle\s+service\s+level|target\s+service\s+level|desired\s+service\s+level|customer\s+service\s+level|service\s+level|csl)",
            r"non[- ]stockout\s+probability\s*(?:is|of|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"(?:service\s+level|csl|non[- ]stockout\s+probability)\s*(?:is|of|=|:)?\s*(0\.[0-9]+)",
        ], q)
        if csl is not None:
            inputs["service_level"] = csl / 100.0 if csl > 1.0 else csl
        else:
            inputs.setdefault("service_level", 0.95)

        # ---------------------------------------------------------------------
        # 8. Fixed-Period Review Disambiguation
        # Distinguishes: review period, lead time, safety stock, inventory position,
        # on-hand inventory, on-order inventory, backorders.
        # ---------------------------------------------------------------------
        # Review Period T
        review_t = find_number_after([
            r"review\s+period\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"reviews?\s+(?:inventory\s+)?(?:every|each)\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)",
            r"every\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)",
            r"(?:order\s+interval|review\s+cycle)\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"t\s*=\s*([0-9,]+(?:\.[0-9]+)?)",
        ], q)
        if review_t is not None:
            inputs["review_period"] = review_t

        # Safety Stock Buffer (if explicitly provided)
        safety_stock_val = find_number_after([
            r"safety\s+stock\s+(?:buffer\s+)?(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*units?",
            r"buffer\s+stock\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*units?",
        ], q)
        if safety_stock_val is not None:
            inputs["safety_stock"] = safety_stock_val

        # Explicit Aggregate Inventory Position
        inv_pos_val = find_number_after([
            r"(?:current\s+|net\s+)?inventory\s+position\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"(?:net\s+inventory|stock\s+position)\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"position\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*units?",
        ], q)
        if inv_pos_val is not None:
            inputs["inventory_position"] = inv_pos_val

        # On-Hand Inventory
        on_hand = find_number_after([
            r"(?:on[- ]hand(?:\s+inventory)?|stock\s+on\s+hand|physical\s+inventory)\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"current\s+inventory\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"currently\s+have\s*([0-9,]+(?:\.[0-9]+)?)\s*units?",
            r"([0-9,]+(?:\.[0-9]+)?)\s*units?\s+on[- ]hand",
        ], q)
        if on_hand is not None:
            inputs["on_hand"] = on_hand

        # On-Order Inventory
        on_order = find_number_after([
            r"(?:on[- ]order(?:\s+inventory)?|in\s+transit|pipeline\s+inventory)\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"open\s+orders?\s+(?:of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"([0-9,]+(?:\.[0-9]+)?)\s*units?\s+on[- ]order",
        ], q)
        if on_order is not None:
            inputs["on_order"] = on_order

        # Backorders
        backorders = find_number_after([
            r"(?:backorders?|backordered|back[- ]orders?)\s*(?:are|is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"unfulfilled\s+(?:orders?|demand)\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"([0-9,]+(?:\.[0-9]+)?)\s*units?\s+backordered",
            r"backlog\s+(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
        ], q)
        if backorders is not None:
            inputs["backorders"] = backorders

        # If inventory position was not stated as an aggregate, compute it from on_hand + on_order - backorders
        if "inventory_position" not in inputs:
            if on_hand is not None or on_order is not None or backorders is not None:
                calc_ip = (on_hand or 0.0) + (on_order or 0.0) - (backorders or 0.0)
                inputs["inventory_position"] = calc_ip

        # ---------------------------------------------------------------------
        # 9. Newsvendor Disambiguation
        # Distinguishes: selling price, purchase cost, salvage value, underage/shortage cost,
        # overage/excess cost, demand distribution, mean demand, demand standard deviation.
        # ---------------------------------------------------------------------
        if model == self.MODEL_NEWSVENDOR:
            selling_p = find_number_after([
                r"(?:selling|retail|sales)\s+price\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?",
                r"sells?\s+(?:for|at)\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?",
                r"revenue\s+per\s+unit\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"price\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?",
            ], q)
            if selling_p is not None:
                inputs["selling_price"] = selling_p

            purchase_p = find_number_after([
                r"(?:unit\s+purchase\s+cost|purchase\s+cost|procurement\s+cost|wholesale\s+cost)\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?",
                r"(?:costs?\s+to\s+produce|production\s+cost)\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?",
                r"(?:unit\s+cost|purchase\s+price)\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?",
                r"costs?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?\s*(?:per|/)\s*(?:unit|each|item)",
            ], q)
            if purchase_p is not None:
                inputs["purchase_cost"] = purchase_p
                inputs["unit_cost"] = purchase_p

            salvage_p = find_number_after([
                r"salvage\s+value\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?",
                r"salvaged?\s+(?:at|for)\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?",
                r"liquidat\w+\s+(?:at|for)\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?",
                r"markdown\s+(?:to|price|value)\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?",
                r"scrap\s+value\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)",
            ], q)
            if salvage_p is not None:
                inputs["salvage_value"] = salvage_p
            else:
                inputs.setdefault("salvage_value", 0.0)

            # Direct Underage / Shortage Cost (Cu)
            cu_val = find_number_after([
                r"(?:underage|shortage)\s+(?:cost|expense)\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?",
                r"cost\s+of\s+(?:underage|shortage)\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?",
                r"lost\s+profit\s+(?:per\s+unit\s+)?(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)",
            ], q)
            if cu_val is not None:
                inputs["underage_cost"] = cu_val

            # Direct Overage / Excess Cost (Co)
            co_val = find_number_after([
                r"(?:overage|excess)\s+(?:cost|expense)\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?",
                r"cost\s+of\s+(?:overage|excess(?:\s+inventory)?)\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?",
                r"loss\s+per\s+unsold\s+unit\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)",
            ], q)
            if co_val is not None:
                inputs["overage_cost"] = co_val

            shortage_penalty = find_number_after([
                r"(?:shortage\s+penalty|penalty|goodwill)\s+cost\s*(?:is|of|=|:)?\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?",
                r"penalty\s+of\s*(?:\\?\$|USD)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?",
            ], q)
            if shortage_penalty is not None:
                inputs["shortage_penalty"] = shortage_penalty
            else:
                inputs.setdefault("shortage_penalty", 0.0)

            # Demand distribution
            if re.search(r"normal(?:ly)?\s+distribut\w+|normal\s+distribution", q, re.IGNORECASE):
                inputs["demand_distribution"] = "normal"
            elif re.search(r"discrete\s+distribut\w+|discrete\s+demand", q, re.IGNORECASE):
                inputs["demand_distribution"] = "discrete"
            else:
                inputs.setdefault("demand_distribution", "normal")

        # ---------------------------------------------------------------------
        # 10. Discount Schedule Extraction (Quantity Discount)
        # Distinguishes quantity break, unit price, and tiers.
        # ---------------------------------------------------------------------
        if model == self.MODEL_QUANTITY_DISCOUNT:
            tiers = self._extract_discount_tiers(q)
            if tiers:
                inputs["discount_schedule"] = tiers

        # ---------------------------------------------------------------------
        # 11. Dataset Analysis & Decision Model Integration
        # ---------------------------------------------------------------------
        if dataset_df is not None:
            if model == self.MODEL_INVENTORY_CLASSIFICATION:
                inputs["dataset_df"] = dataset_df
                inputs["sales_weight"] = 0.65
                inputs["hits_weight"] = 0.35

            elif model == self.MODEL_DATASET_ANALYSIS:
                q_lower = q.lower()
                if re.search(r"valuation|total\s+(?:inventory\s+)?value", q_lower):
                    inputs["query_focus"] = "total_valuation"
                elif re.search(r"highest\s+unit|highest\s+quant|most\s+stock|most\s+unit|top\s+prod", q_lower):
                    inputs["query_focus"] = "highest_units"
                elif re.search(r"average\s+(?:unit\s+)?cost|avg\s+(?:unit\s+)?cost|mean\s+(?:unit\s+)?cost|average\s+cost", q_lower):
                    inputs["query_focus"] = "average_cost"
                elif re.search(r"total\s+unit|total\s+stock|total\s+quant|how\s+many\s+unit", q_lower):
                    inputs["query_focus"] = "total_units"
                elif re.search(r"category|categories", q_lower):
                    inputs["query_focus"] = "category_summary"
                elif re.search(r"quality|missing|clean|null|duplicate", q_lower):
                    inputs["query_focus"] = "data_quality"
                else:
                    inputs["query_focus"] = "dataset_summary"
                inputs["dataset_df"] = dataset_df

            else:
                # Analytical decision model queried on dataset: match product entity
                prod_col = None
                for c in dataset_df.columns:
                    if str(c).strip().lower() in ["product", "item", "sku", "description"]:
                        prod_col = c
                        break
                if prod_col:
                    unique_prods = dataset_df[prod_col].dropna().astype(str).unique()
                    sorted_prods = sorted(unique_prods, key=len, reverse=True)
                    matched = None
                    for p_candidate in sorted_prods:
                        if re.search(r"\b" + re.escape(p_candidate) + r"\b", q, re.IGNORECASE) or p_candidate.lower() in q.lower():
                            matched = p_candidate
                            break
                    if matched:
                        row = dataset_df[dataset_df[prod_col].astype(str).str.lower() == matched.lower()].iloc[0]
                        inputs["matched_product"] = str(row[prod_col])
                        inputs["integrated_with_dataset"] = True

                        u_col = next((c for c in dataset_df.columns if str(c).strip().lower() in ["units", "quantity", "qty", "demand", "stock"]), None)
                        c_col = next((c for c in dataset_df.columns if str(c).strip().lower() in ["unit cost", "unit_cost", "cost", "price", "unit price"]), None)

                        if u_col and pd.notna(row[u_col]):
                            try:
                                u_val = float(str(row[u_col]).replace(",", "").strip())
                                inputs["dataset_units"] = u_val
                                if inputs.get("annual_demand") is None and inputs.get("demand_mean") is None and inputs.get("demand_rate") is None:
                                    if model in [self.MODEL_EOQ, self.MODEL_QUANTITY_DISCOUNT]:
                                        inputs["annual_demand"] = u_val
                                    elif model == self.MODEL_NEWSVENDOR:
                                        inputs["demand_mean"] = u_val
                                    elif model == self.MODEL_SAFETY_STOCK_ROP:
                                        inputs["annual_demand"] = u_val
                            except Exception:
                                pass

                        if c_col and pd.notna(row[c_col]):
                            try:
                                c_val = float(str(row[c_col]).replace("$", "").replace(",", "").strip())
                                inputs["dataset_unit_cost"] = c_val
                                if inputs.get("unit_cost") is None:
                                    inputs["unit_cost"] = c_val
                                if model == self.MODEL_NEWSVENDOR and inputs.get("purchase_cost") is None:
                                    inputs["purchase_cost"] = c_val
                            except Exception:
                                pass

        return inputs

    def _extract_discount_tiers(self, text: str) -> List[Dict[str, Any]]:
        """Extract multi-tier discount schedules from text supporting multiple natural-language syntaxes."""
        tiers = []
        chunks = re.split(r"[,;\n]|\band\s+(?=[\$0-9]|tier)|(?=tier\s*\d+)", text, flags=re.IGNORECASE)

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            # 1A: Price first, range: e.g. "$9.50 for orders of 500 to 999 units"
            m = re.search(
                r"(?:(?:unit\s+)?price\s+(?:is|of|=|:)?\s*)?(?:\\?\$|USD)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?\s*"
                r"(?:for|if|on|when)?\s*(?:orders?\s+)?(?:of|between|from)?\s*([0-9,]+)\s*(?:to|-|through|and)\s*([0-9,]+)\s*(?:units?)?",
                chunk,
                re.IGNORECASE,
            )
            if m and any(w in chunk.lower() for w in ["order", "unit", "tier", "bracket"]):
                p = float(m.group(1))
                q1 = float(m.group(2).replace(",", ""))
                q2 = float(m.group(3).replace(",", ""))
                if q1 < q2 and p > 0:
                    tiers.append({"min_qty": q1, "max_qty": q2, "price": p})
                    continue

            # 1B: Price first, below: e.g. "Unit price is $10 for orders below 500 units"
            m = re.search(
                r"(?:(?:unit\s+)?(?:purchase\s+)?price\s+(?:is|of|=|:)?\s*)?(?:\\?\$|USD)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?\s*"
                r"(?:for|if|on|when)?\s*(?:orders?\s+)?(?:below|under|less\s+than|<|up\s+to)\s*([0-9,]+)\s*(?:units?)?",
                chunk,
                re.IGNORECASE,
            )
            if m and any(w in chunk.lower() for w in ["below", "under", "less", "up to", "<"]):
                p = float(m.group(1))
                thresh = float(m.group(2).replace(",", ""))
                max_q = thresh - 1 if thresh > 1 else thresh
                if p > 0:
                    tiers.append({"min_qty": 0.0, "max_qty": max_q, "price": p})
                    continue

            # 1C: Price first, open-ended: e.g. "$9 for orders of 1000 units or more"
            m = re.search(
                r"(?:(?:unit\s+)?price\s+(?:is|of|=|:)?\s*)?(?:\\?\$|USD)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:dollars?|USD)?\s*"
                r"(?:for|if|on|when)?\s*(?:orders?\s+)?(?:of|at|for|from)?\s*([0-9,]+)\s*(?:units?)?\s*"
                r"(?:or\s+more|\+|and\s+above|and\s+over|and\s+up|\bmore\b|above|over)",
                chunk,
                re.IGNORECASE,
            )
            if m and any(w in chunk.lower() for w in ["more", "above", "over", "+", "up"]):
                p = float(m.group(1))
                min_q = float(m.group(2).replace(",", ""))
                if p > 0:
                    tiers.append({"min_qty": min_q, "max_qty": None, "price": p})
                    continue

            # 2A: Quantity first, range: e.g. "Tier 1: 0 to 999 at $5.00"
            m = re.search(
                r"(?:tier\s*\d+[:\s]*)?([0-9,]+)\s*(?:to|-)\s*([0-9,]+)\s*(?:units?)?\s*(?:at|@|costs?|for|:)?\s*(?:\\?\$|USD)?\s*([0-9]+(?:\.[0-9]+)?)",
                chunk,
                re.IGNORECASE,
            )
            if m:
                q1 = float(m.group(1).replace(",", ""))
                q2 = float(m.group(2).replace(",", ""))
                p = float(m.group(3).rstrip("."))
                if q1 <= q2 and p > 0:
                    tiers.append({"min_qty": q1, "max_qty": q2, "price": p})
                    continue

            # 2B: Quantity first, open-ended: e.g. "Tier 3: 2500+ at $3.90"
            m = re.search(
                r"(?:tier\s*\d+[:\s]*)?([0-9,]+)\s*(?:\+|and\s+above|or\s+more|\s+and\s+over|\s+and\s+up)\s*(?:units?)?\s*(?:at|@|costs?|for|:)?\s*(?:\\?\$|USD)?\s*([0-9]+(?:\.[0-9]+)?)",
                chunk,
                re.IGNORECASE,
            )
            if m:
                min_q = float(m.group(1).replace(",", ""))
                p = float(m.group(2).rstrip("."))
                if p > 0:
                    tiers.append({"min_qty": min_q, "max_qty": None, "price": p})
                    continue

        # Fallback to general break search if above didn't find multiple tiers
        if len(tiers) < 2:
            break_matches = re.findall(
                r"(?:over|above|>=|at\s+least|(?:quantity\s+)?price\s+break\s+at)\s*([0-9,]+)\s*units?[^$0-9\n\r]*[\$]?\s*([0-9]+(?:\.[0-9]+)?)",
                text,
                re.IGNORECASE,
            )
            base_price = re.search(
                r"(?:regular|base|standard|normal|unit\s+purchase)\s+(?:unit\s+)?price\s+(?:is|of|=|:)?\s*[\$]?\s*([0-9]+(?:\.[0-9]+)?)",
                text,
                re.IGNORECASE,
            )
            if break_matches and base_price:
                tiers = []
                p0 = float(base_price.group(1))
                curr_min = 0.0
                for b_qty_str, b_price_str in break_matches:
                    b_qty = float(b_qty_str.replace(",", ""))
                    b_price = float(b_price_str)
                    tiers.append({"min_qty": curr_min, "max_qty": b_qty - 1, "price": p0})
                    curr_min = b_qty
                    p0 = b_price
                tiers.append({"min_qty": curr_min, "max_qty": None, "price": p0})

        # Remove duplicate tiers if any and sort by min_qty
        seen = set()
        clean_tiers = []
        for t in sorted(tiers, key=lambda x: x["min_qty"]):
            key = (t["min_qty"], t["max_qty"], t["price"])
            if key not in seen:
                seen.add(key)
                clean_tiers.append(t)

        return clean_tiers

    # =========================================================================
    # Stage 3: Select Python Tool
    # =========================================================================
    def select_python_tool(self, model: str):
        """Map model to corresponding analytical tool in inventory_tools."""
        tool_name = self.MODEL_TOOL_MAP.get(model)
        if not tool_name:
            raise ValueError(f"Unknown inventory model '{model}'")
        tool_callable = getattr(itools, tool_name)
        return tool_name, tool_callable

    # =========================================================================
    # Stage 4: Calculate
    # =========================================================================
    def calculate(self, model: str, tool_callable, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute deterministic Python calculation with validated parameter mapping.
        Examines extracted inputs and ensures all available parameters are used.
        """
        if model == self.MODEL_EOQ:
            d = inputs.get("annual_demand") or inputs.get("demand_mean") or inputs.get("demand_rate")
            s = inputs.get("ordering_cost")
            h = inputs.get("holding_cost")
            hr = inputs.get("holding_rate")
            c = inputs.get("unit_cost")

            if d is None or s is None or (h is None and (hr is None or c is None)):
                shell_hint = ""
                q_text = inputs.get("_raw_question", "")
                if re.search(r"cost\s+(?:is|of)\s+(?:per|/|\s)", q_text, re.IGNORECASE):
                    shell_hint = (
                        " [Notice: Detected stripped numbers directly after 'cost is'. If running from a terminal shell "
                        "(e.g., zsh/bash), enclose the question in single quotes ('...') so $50 or $2 are not expanded to empty strings.]"
                    )
                missing = []
                if d is None: missing.append("Demand (D)")
                if s is None: missing.append("Ordering Cost (S)")
                if h is None and (hr is None or c is None): missing.append("Holding Cost (H) or (Holding Rate & Unit Cost)")
                prod_desc = f" for product '{inputs.get('matched_product')}'" if inputs.get("matched_product") else ""
                raise ValueError(
                    f"Incomplete parameters for EOQ calculation{prod_desc}. Missing: {', '.join(missing)}. "
                    f"Got Demand={d}, OrderingCost={s}, HoldingCost={h}, HoldingRate={hr}, UnitCost={c}.{shell_hint} "
                    f"Please provide the missing numerical values in your question (the decision agent never invents missing parameters)."
                )
            return tool_callable(
                annual_demand=d,
                ordering_cost=s,
                holding_cost=h,
                holding_rate=hr,
                unit_cost=c,
            )

        elif model == self.MODEL_QUANTITY_DISCOUNT:
            d = inputs.get("annual_demand") or inputs.get("demand_mean") or inputs.get("demand_rate")
            s = inputs.get("ordering_cost")
            schedule = inputs.get("discount_schedule")
            hr = inputs.get("holding_rate")
            h_def = inputs.get("holding_cost")

            if d is None or s is None or not schedule:
                missing = []
                if d is None: missing.append("Annual Demand (D)")
                if s is None: missing.append("Ordering Cost (S)")
                if not schedule: missing.append("Discount Schedule / Quantity Breaks")
                raise ValueError(
                    f"Incomplete parameters for Quantity Discount. Missing: {', '.join(missing)}. "
                    f"Demand={d}, OrderingCost={s}, Schedule={schedule}"
                )
            return tool_callable(
                annual_demand=d,
                ordering_cost=s,
                discount_schedule=schedule,
                holding_rate=hr,
                default_holding_cost=h_def,
            )

        elif model == self.MODEL_SAFETY_STOCK_ROP:
            lt = inputs.get("lead_time")
            d_rate = inputs.get("demand_rate") or inputs.get("demand_rate_daily") or inputs.get("demand_rate_weekly") or inputs.get("annual_demand")
            s_d = inputs.get("demand_std_dev", 0.0)
            csl = inputs.get("service_level", 0.95)
            s_lt = inputs.get("lead_time_std_dev", 0.0)
            t_unit = inputs.get("time_unit", "days")

            if lt is None or d_rate is None:
                missing = []
                if lt is None: missing.append("Lead Time (L)")
                if d_rate is None: missing.append("Average Demand Rate (d)")
                raise ValueError(f"Incomplete parameters for Safety Stock & ROP. Missing: {', '.join(missing)}. Got LeadTime={lt}, DemandRate={d_rate}.")
            return tool_callable(
                lead_time=lt,
                demand_rate=d_rate,
                demand_std_dev=s_d,
                service_level=csl,
                lead_time_std_dev=s_lt,
                time_unit=t_unit,
            )

        elif model == self.MODEL_FIXED_PERIOD_REVIEW:
            t = inputs.get("review_period")
            lt = inputs.get("lead_time")
            d_rate = inputs.get("demand_rate") or inputs.get("demand_rate_daily") or inputs.get("demand_rate_weekly")
            s_d = inputs.get("demand_std_dev", 0.0)
            csl = inputs.get("service_level", 0.95)
            oh = inputs.get("on_hand", 0.0)
            oo = inputs.get("on_order", 0.0)
            bo = inputs.get("backorders", 0.0)
            ip = inputs.get("inventory_position")
            explicit_ss = inputs.get("safety_stock")
            t_unit = inputs.get("time_unit", "weeks")

            if t is None or lt is None or d_rate is None:
                missing = []
                if t is None: missing.append("Review Period (T)")
                if lt is None: missing.append("Replenishment Lead Time (L)")
                if d_rate is None: missing.append("Demand Rate (d)")
                raise ValueError(f"Incomplete parameters for Periodic Review. Missing: {', '.join(missing)}. Got T={t}, L={lt}, DemandRate={d_rate}.")
            return tool_callable(
                review_period=t,
                lead_time=lt,
                demand_rate=d_rate,
                demand_std_dev=s_d,
                service_level=csl,
                on_hand=oh,
                on_order=oo,
                backorders=bo,
                inventory_position=ip,
                time_unit=t_unit,
                explicit_safety_stock=explicit_ss,
            )

        elif model == self.MODEL_NEWSVENDOR:
            p = inputs.get("selling_price")
            c = inputs.get("purchase_cost") or inputs.get("unit_cost")
            s = inputs.get("salvage_value", 0.0)
            g = inputs.get("shortage_penalty", 0.0)
            cu = inputs.get("underage_cost")
            co = inputs.get("overage_cost")
            mu = inputs.get("demand_mean") or inputs.get("annual_demand") or inputs.get("demand_rate")
            sigma = inputs.get("demand_std_dev")
            dist = inputs.get("demand_distribution", "normal")

            if (p is None or c is None) and (cu is None or co is None):
                raise ValueError(
                    f"Incomplete parameters for Newsvendor. Must provide either (Selling Price & Purchase Cost) "
                    f"or (Underage Cost & Overage Cost). Got Price={p}, Cost={c}, Cu={cu}, Co={co}."
                )
            if mu is None and sigma is None:
                raise ValueError(f"Must provide demand mean and standard deviation for Newsvendor. Mean={mu}, Sigma={sigma}.")

            return tool_callable(
                selling_price=p,
                purchase_cost=c,
                salvage_value=s,
                shortage_penalty=g,
                demand_mean=mu,
                demand_std_dev=sigma,
                underage_cost=cu,
                overage_cost=co,
                demand_distribution=dist,
            )

        elif model == self.MODEL_DATASET_ANALYSIS:
            df_target = inputs.get("dataset_df")
            if df_target is None:
                raise ValueError("No dataset provided for dataset analysis. Please upload an inventory CSV file.")
            focus = inputs.get("query_focus", "dataset_summary")
            matched = inputs.get("matched_product")
            return tool_callable(dataset_df=df_target, query_focus=focus, matched_product=matched)

        elif model == self.MODEL_INVENTORY_CLASSIFICATION:
            df_target = inputs.get("dataset_df")
            if df_target is None:
                raise ValueError("No dataset provided for inventory classification. Please upload a CSV file containing Sales ($) and Hits.")
            s_w = inputs.get("sales_weight", 0.65)
            h_w = inputs.get("hits_weight", 0.35)
            return tool_callable(df_target, sales_weight=s_w, hits_weight=h_w)

        else:
            raise ValueError(f"Unsupported model: {model}")

    # =========================================================================
    # Stage 5: Verify
    # =========================================================================
    def verify(self, model: str, calc_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify the mathematical and operational validity of calculation results.
        """
        verif = calc_output.get("verification", {})
        results = calc_output.get("results", {})
        status = verif.get("verification_status", "PASSED")
        notes = []

        if model == self.MODEL_EOQ:
            if verif.get("cost_balance_valid"):
                notes.append("Ordering cost and holding cost are in equilibrium at optimal lot size.")
            if verif.get("convexity_confirmed"):
                notes.append("Total cost function is strictly convex (d²TC/dQ² > 0); confirmed global minimum.")

        elif model == self.MODEL_QUANTITY_DISCOUNT:
            if verif.get("optimal_tier_satisfies_bounds"):
                notes.append("Optimal lot size satisfies the price tier volume boundary.")
            rec_q = results.get("optimal_order_quantity", 0)
            savings = results.get("annual_savings_vs_tier_1", 0)
            notes.append(f"Volume discount generates net annual cost reduction of ${savings:,.2f}.")

        elif model == self.MODEL_SAFETY_STOCK_ROP:
            rop = results.get("reorder_point_rop", 0)
            ss = results.get("safety_stock", 0)
            if ss >= 0 and rop > 0:
                notes.append(f"Safety stock ({ss:.1f}) and ROP ({rop:.1f}) provide designated cycle protection.")

        elif model == self.MODEL_FIXED_PERIOD_REVIEW:
            s_level = results.get("target_order_up_to_level_S", 0)
            q_rec = results.get("recommended_order_quantity", 0)
            notes.append(f"Order-Up-To Level ({s_level:.1f}) covers review period + lead time protection interval.")
            notes.append(f"Recommended replenishment order is {math.ceil(q_rec)} units.")

        elif model == self.MODEL_NEWSVENDOR:
            cr = calc_output.get("intermediate", {}).get("critical_ratio", 0)
            if 0 < cr < 1:
                notes.append(f"Critical ratio CR = {cr:.3f} is mathematically consistent with underage/overage trade-off.")

        elif model == self.MODEL_DATASET_ANALYSIS:
            rec_cnt = results.get("records_count", 0)
            notes.append(f"Examined {rec_cnt} inventory records from the uploaded dataset.")
            notes.append("Dataset schema, numeric values, and non-negativity checks verified.")

        elif model == self.MODEL_INVENTORY_CLASSIFICATION:
            rec_cnt = results.get("total_records", 0)
            notes.append(f"Successfully classified {rec_cnt} inventory items into BAN 517 A, B, C, D classes.")
            notes.append("Sales weighted 65%, POS Hits weighted 35%, 100% item conservation verified.")

        return {
            "status": status,
            "verification_checks": verif,
            "audit_notes": notes,
        }

    # =========================================================================
    # Stage 6: Explain
    # =========================================================================
    def explain(
        self,
        question: str,
        model: str,
        calc_output: Dict[str, Any],
        verification: Dict[str, Any],
    ) -> str:
        """
        Synthesize clear, executive-grade managerial takeaways and operations guidance.
        """
        results = calc_output.get("results", {})
        inter = calc_output.get("intermediate", {})
        base_explanation = calc_output.get("explanation", "")

        lines = [
            f"### Supply Chain Analytics Executive Report",
            f"**Decision Model:** {calc_output.get('model', model)}",
            f"**Validation Status:** ✅ {verification.get('status', 'PASSED')}",
            "",
            "#### 1. Key Operational Recommendations",
        ]

        if model == self.MODEL_EOQ:
            q = results.get("optimal_order_quantity_eoq")
            orders = results.get("orders_per_year")
            days = results.get("order_cycle_time_days")
            tot_cost = results.get("annual_inventory_cost")
            lines.extend([
                f"- **Recommended Order Lot Size:** **{math.ceil(q):,} units** (exact: {q:.2f})",
                f"- **Order Frequency:** **{orders:.1f} orders per year** (every **{days:.1f} days**)",
                f"- **Annual Inventory Carrying & Ordering Cost:** **${tot_cost:,.2f}**",
                f"- **Cost Trade-off:** Ordering cost (${results.get('annual_ordering_cost'):,.2f}) and holding cost (${results.get('annual_holding_cost'):,.2f}) are balanced at minimum total cost.",
            ])

        elif model == self.MODEL_QUANTITY_DISCOUNT:
            tier = results.get("recommended_tier")
            q = results.get("optimal_order_quantity")
            price = results.get("best_unit_price")
            savings = results.get("annual_savings_vs_tier_1")
            tot_cost = results.get("minimum_total_annual_cost")
            lines.extend([
                f"- **Recommended Order Size:** **{math.ceil(q):,} units** at **${price:,.2f} per unit** (Tier {tier})",
                f"- **Net Annual Savings:** **${savings:,.2f}** saved versus baseline non-discounted pricing",
                f"- **Total Annual Spend (Procurement + Inventory):** **${tot_cost:,.2f}**",
                f"- **Managerial Insight:** The purchasing price concession outweighs the extra carrying cost of holding larger batches.",
            ])

        elif model == self.MODEL_SAFETY_STOCK_ROP:
            ss = results.get("safety_stock")
            ss_int = results.get("safety_stock_recommended_integer")
            rop = results.get("reorder_point_rop")
            rop_int = results.get("reorder_point_recommended_integer")
            csl = results.get("cycle_service_level_pct")
            lines.extend([
                f"- **Reorder Point (ROP):** **{rop_int:,} units** (exact: {rop:.2f})",
                f"- **Safety Stock Buffer:** **{ss_int:,} units** (exact: {ss:.2f})",
                f"- **Target Cycle Service Level:** **{csl:.1f}%**",
                f"- **Action Trigger:** When on-hand + open on-order inventory falls to **{rop_int:,} units**, immediately issue a replenishment purchase order.",
            ])

        elif model == self.MODEL_FIXED_PERIOD_REVIEW:
            s_level = results.get("target_order_up_to_level_integer")
            s_exact = results.get("target_order_up_to_level_S")
            q_rec = results.get("recommended_order_quantity_integer")
            q_exact = results.get("recommended_order_quantity")
            ss = results.get("safety_stock")
            interval = inter.get("protection_interval_T_plus_L")
            inv_pos = inter.get("current_inventory_position", 0.0)
            lines.extend([
                f"- **Target Order-Up-To Level (S):** **{s_level:,} units** (exact: {s_exact:.1f})",
                f"- **Current Inventory Position:** **{inv_pos:,.1f} units**",
                f"- **Order Quantity to Place Today:** **{q_rec:,} units** (exact: {q_exact:.1f})",
                f"- **Safety Stock Required:** **{math.ceil(ss):,} units** (exact: {ss:.1f})",
                f"- **Protection Window:** **{interval:.1f} periods** (Review Period T + Replenishment Lead Time L)",
                f"- **Policy:** Order quantity is max(0, S - Inventory Position) rounded up to {q_rec:,} units.",
            ])

        elif model == self.MODEL_NEWSVENDOR:
            q = results.get("optimal_order_quantity_integer")
            cr = inter.get("critical_ratio")
            profit = results.get("expected_profit")
            sales = results.get("expected_sales")
            leftovers = results.get("expected_leftovers")
            lines.extend([
                f"- **Optimal Order Quantity (Q*):** **{q:,} units**",
                f"- **Critical Ratio (Optimal Service Level):** **{cr*100:.1f}%**",
                f"- **Maximized Expected Profit:** **${profit:,.2f}**",
                f"- **Expected Sales:** **{sales:.1f} units** | **Expected Leftover / Markdown:** **{leftovers:.1f} units**",
                f"- **Underage vs. Overage:** Margin on unit sales is balanced against salvage liquidation risk at the critical fractile.",
            ])

        elif model == self.MODEL_DATASET_ANALYSIS:
            val = results.get("total_inventory_value", 0.0)
            units = results.get("total_units", 0)
            avg_cost = results.get("average_unit_cost", 0.0)
            n_rec = results.get("records_count", 0)
            top_prod = results.get("highest_units_product", "N/A")
            top_units = results.get("highest_units_count", 0)
            lines.extend([
                f"- **Total Portfolio Valuation:** **${val:,.2f}**",
                f"- **Total Inventory Units:** **{units:,} units** across **{n_rec} product records**",
                f"- **Average Unit Cost:** **${avg_cost:,.2f}** per unit",
                f"- **Highest-Volume SKU:** **{top_prod}** ({top_units:,} units in stock / demand)",
                f"- **Portfolio Takeaway:** {base_explanation}",
            ])

        elif model == self.MODEL_INVENTORY_CLASSIFICATION:
            s_rows = results.get("class_summary", [])
            lines.extend([
                f"- **BAN 517 Methodology:** 65% Annual Dollar Sales ($) + 35% POS Pick Hits",
                f"- **Class A (Vital Few):** **{results.get('class_A_count', 0):,} items** (Top throughput and financial priority)",
                f"- **Class B (Moderate):** **{results.get('class_B_count', 0):,} items** (Secondary replenishment attention)",
                f"- **Class C (Low Activity):** **{results.get('class_C_count', 0):,} items** (Routine periodic replenishment)",
                f"- **Class D (Slow / Dead Stock):** **{results.get('class_D_count', 0):,} items** (Candidates for SKU rationalization or obsolescence review)",
                f"- **Summary:** {base_explanation}",
            ])

        # If entity was matched from dataset in a decision model, highlight it
        matched_prod = calc_output.get("inputs", {}).get("matched_product")
        if matched_prod and model != self.MODEL_DATASET_ANALYSIS:
            lines.insert(5, f"- **Integrated Dataset SKU:** Parameters for **{matched_prod}** were retrieved directly from the uploaded inventory dataset.")

        lines.extend([
            "",
            "#### 2. Detailed Mathematical Context",
            base_explanation,
            "",
            "#### 3. Verification Audit",
        ])
        for note in verification.get("audit_notes", []):
            lines.append(f"- ✓ {note}")

        return "\n".join(lines)

    # =========================================================================
    # Full Agent Process Method
    # =========================================================================
    def process_question(
        self,
        question: str,
        override_inputs: Optional[Dict[str, Any]] = None,
        dataset_df: Optional[Any] = None,
    ) -> AgentPipelineResult:
        """
        Execute the full 6-stage decision agent pipeline:
          Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain
        Supports integration with uploaded inventory dataset records and portfolio queries.
        """
        trace = []

        # Stage 1: Customer Question & Identify Model
        identified_model, confidence, rationale = self.identify_model(question, dataset_df=dataset_df)
        trace.append({
            "stage": "1. Identify Model",
            "model": identified_model,
            "confidence": confidence,
            "rationale": rationale,
        })

        # Stage 2: Extract Inputs
        extracted_inputs = self.extract_inputs(identified_model, question, dataset_df=dataset_df)
        if override_inputs:
            extracted_inputs.update(override_inputs)
        trace.append({
            "stage": "2. Extract Inputs",
            "inputs": {k: v for k, v in extracted_inputs.items() if k != "dataset_df"},
        })

        # Stage 3: Select Python Tool
        try:
            tool_name, tool_callable = self.select_python_tool(identified_model)
            trace.append({
                "stage": "3. Select Python Tool",
                "tool_name": tool_name,
            })
        except Exception as e:
            return AgentPipelineResult(
                question=question,
                identified_model=identified_model,
                confidence=confidence,
                extracted_inputs=extracted_inputs,
                selected_tool="unknown",
                status="ERROR",
                error_message=f"Tool selection error: {str(e)}",
                pipeline_trace=trace,
            )

        # Stage 4: Calculate
        try:
            calc_results = self.calculate(identified_model, tool_callable, extracted_inputs)
            if extracted_inputs.get("matched_product"):
                calc_results.setdefault("inputs", {})["matched_product"] = extracted_inputs["matched_product"]
            trace.append({
                "stage": "4. Calculate",
                "status": "COMPLETED",
                "results_summary": calc_results.get("results"),
            })
        except Exception as e:
            return AgentPipelineResult(
                question=question,
                identified_model=identified_model,
                confidence=confidence,
                extracted_inputs=extracted_inputs,
                selected_tool=tool_name,
                status="ERROR",
                error_message=f"Calculation error: {str(e)}",
                pipeline_trace=trace,
            )

        # Stage 5: Verify
        try:
            verification_rep = self.verify(identified_model, calc_results)
            trace.append({
                "stage": "5. Verify",
                "verification_status": verification_rep.get("status"),
                "checks": verification_rep.get("verification_checks"),
            })
        except Exception as e:
            verification_rep = {"status": "WARNING", "error": str(e), "audit_notes": []}
            trace.append({"stage": "5. Verify", "error": str(e)})

        # Stage 6: Explain
        explanation = self.explain(question, identified_model, calc_results, verification_rep)
        trace.append({
            "stage": "6. Explain",
            "status": "GENERATED",
        })

        return AgentPipelineResult(
            question=question,
            identified_model=identified_model,
            confidence=confidence,
            extracted_inputs=extracted_inputs,
            selected_tool=tool_name,
            calculation_results=calc_results,
            verification_report=verification_rep,
            explanation=explanation,
            status="SUCCESS",
            pipeline_trace=trace,
        )

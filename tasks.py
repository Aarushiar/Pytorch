"""
Customer Support Environment - Task Definitions & Graders

Three tasks with increasing difficulty:
1. EASY: Ticket classification + basic routing
2. MEDIUM: Multi-step resolution with clarifications
3. HARD: Ambiguous issue requiring reasoning + edge case handling

Each task includes:
- Example ticket
- Expected outputs
- Deterministic grading logic
- Reward function (partial scoring)
- Penalties for incorrect/harmful responses
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Literal, Tuple
from enum import Enum
import re


# ============================================================================
# TASK 1: EASY - Ticket Classification + Basic Routing
# ============================================================================

@dataclass
class Task1_InputTicket:
    """Task 1 Input: Simple ticket that's easy to classify"""
    ticket_id: str = "TKT-001-EASY"
    customer_message: str = "I was charged twice for my subscription. Please refund one charge."
    customer_tier: str = "free"
    priority: str = "medium"
    category: str = "billing"


@dataclass
class Task1_ExpectedOutputs:
    """Task 1: What we expect the agent to do"""
    # Primary outputs (in order of likelihood)
    primary_action: str = "suggest_knowledge_base"  # 70% probability optimal
    alternative_actions: List[str] = None  # ["request_more_info", "escalate_to_human"]
    
    def __post_init__(self):
        if self.alternative_actions is None:
            self.alternative_actions = ["request_more_info", "escalate_to_human"]


class Task1_Grader:
    """Deterministic grader for EASY task - Ticket Classification"""

    @staticmethod
    def grade(
        action_taken: str,
        reasoning: str,
        ticket_context: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Grade Task 1 response.
        
        Returns: (score, breakdown)
        - score: 0.0-1.0
        - breakdown: {'action_score': X, 'reasoning_score': Y, 'penalties': Z}
        
        Constraints:
        ✓ Never returns constant score
        ✓ Supports partial scoring
        ✓ Penalizes incorrect responses
        """

        breakdown = {
            "action_score": 0.0,
            "reasoning_score": 0.0,
            "tier_awareness": 0.0,
            "penalties": 0.0,
            "final_score": 0.0,
        }

        # ==================== ACTION SCORING ====================
        if action_taken == "suggest_knowledge_base":
            breakdown["action_score"] = 1.0  # Perfect
            kb_suggestion_reward = 0.8
        elif action_taken == "request_more_info":
            breakdown["action_score"] = 0.6  # Acceptable
            kb_suggestion_reward = 0.5
        elif action_taken == "escalate_to_human":
            breakdown["action_score"] = 0.4  # Safe but costly for free tier
            kb_suggestion_reward = 0.3
        elif action_taken == "assign_department":
            breakdown["action_score"] = 0.3  # Wrong (billing has self-service)
            kb_suggestion_reward = 0.2
        else:
            breakdown["action_score"] = 0.0  # Unknown or harmful
            kb_suggestion_reward = 0.0

        # ==================== REASONING SCORING ====================
        reasoning_score = 0.0
        
        # Check if reasoning mentions the problem correctly
        if "duplicate" in reasoning.lower() or "charge" in reasoning.lower():
            reasoning_score += 0.4
        
        # Check if reasoning shows understanding of ticket category
        if "billing" in reasoning.lower() or "payment" in reasoning.lower():
            reasoning_score += 0.3
        
        # Check if reasoning is substantive (not empty or trivial)
        if len(reasoning.strip()) > 20:
            reasoning_score += 0.2
        else:
            reasoning_score = 0.0  # Penalty for no reasoning
        
        breakdown["reasoning_score"] = min(reasoning_score, 1.0)

        # ==================== TIER AWARENESS ====================
        # Free tier should prefer no escalation (costs money)
        if ticket_context.get("customer_tier") == "free":
            if action_taken in ["suggest_knowledge_base", "request_more_info"]:
                breakdown["tier_awareness"] = 1.0
            elif action_taken == "escalate_to_human":
                breakdown["tier_awareness"] = 0.3  # Penalize unnecessarily escalating free tier
            else:
                breakdown["tier_awareness"] = 0.5
        else:
            breakdown["tier_awareness"] = 0.8

        # ==================== PENALTIES ====================
        penalties = 0.0
        
        # Penalty: Escalating routine billing issue
        if action_taken == "escalate_to_human" and "billing" in ticket_context.get("category", ""):
            penalties += 0.15
        
        # Penalty: Closing without addressing issue
        if action_taken == "close_resolved":
            penalties += 0.3  # Can't close without resolving
        
        # Penalty: Harmful actions
        if action_taken == "request_callback":
            penalties += 0.1  # Unnecessary callback for simple issue
        
        breakdown["penalties"] = penalties

        # ==================== FINAL CALCULATION ====================
        breakdown["final_score"] = (
            breakdown["action_score"] * 0.5 +      # Action choice (50%)
            breakdown["reasoning_score"] * 0.2 +   # Reasoning quality (20%)
            breakdown["tier_awareness"] * 0.2 +    # Tier awareness (20%)
            - breakdown["penalties"]                # Penalize mistakes
        )

        # Clamp to [0, 1]
        final_score = max(0.0, min(1.0, breakdown["final_score"]))
        breakdown["final_score"] = final_score

        return final_score, breakdown


# ============================================================================
# TASK 2: MEDIUM - Multi-Step Resolution
# ============================================================================

@dataclass
class Task2_InputTicket:
    """Task 2 Input: Requires multiple steps to resolve"""
    ticket_id: str = "TKT-002-MEDIUM"
    customer_message: str = """
    I'm getting error 403 when trying to access my dashboard. 
    I just updated to the new plan yesterday. Nothing works.
    """
    customer_tier: str = "pro"
    priority: str = "high"
    category: str = "technical"


@dataclass
class Task2_ExpectedPath:
    """Optimal solution path (not rigid, allows variations)"""
    step_1_action: str = "request_more_info"  # Get error details, browser, etc.
    step_1_details: str = "Ask about error details, browser, cache clearing steps"
    
    step_2_action: str = "suggest_knowledge_base"  # Link KB article on 403 errors
    step_2_details: str = "Provide troubleshooting KB article for 403 errors"
    
    step_3_action: str = "close_resolved"
    step_3_details: str = "Confirm resolution or escalate if KB doesn't help"


class Task2_Grader:
    """Deterministic grader for MEDIUM task - Multi-step resolution"""

    @staticmethod
    def grade_sequence(
        actions_taken: List[str],
        reasoning_list: List[str],
        ticket_context: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Grade multi-step response sequence.
        
        Returns: (score, breakdown)
        - Score varies based on path taken
        - Partial credit for correct first steps
        - Penalizes wrong sequences
        """

        breakdown = {
            "step_quality": [],
            "sequence_efficiency": 0.0,
            "escalation_handling": 0.0,
            "penalties": 0.0,
            "final_score": 0.0,
        }

        max_steps = 5

        # ==================== EVALUATE EACH STEP ====================
        for step_idx, action in enumerate(actions_taken[:max_steps]):
            step_score = Task2_Grader._score_step(
                step_idx,
                action,
                reasoning_list[step_idx] if step_idx < len(reasoning_list) else "",
                ticket_context
            )
            breakdown["step_quality"].append({
                "step": step_idx + 1,
                "action": action,
                "score": step_score
            })

        # ==================== SEQUENCE EFFICIENCY ====================
        # Optimal: ask → suggest KB → close
        if len(actions_taken) <= 3 and actions_taken[0] == "request_more_info":
            breakdown["sequence_efficiency"] = 0.9
        elif len(actions_taken) <= 4:
            breakdown["sequence_efficiency"] = 0.7
        elif len(actions_taken) <= 6:
            breakdown["sequence_efficiency"] = 0.5
        else:
            breakdown["sequence_efficiency"] = 0.2

        # ==================== ESCALATION HANDLING ====================
        # For PRO tier technical issue, escalation is acceptable
        if "escalate_to_human" in actions_taken:
            if ticket_context.get("priority") == "high":
                breakdown["escalation_handling"] = 0.6  # Acceptable
            else:
                breakdown["escalation_handling"] = 0.3  # Premature
        else:
            breakdown["escalation_handling"] = 0.8  # Good: solved without escalation

        # ==================== PENALTIES ====================
        penalties = 0.0
        
        # Penalty: Too many steps (inefficient)
        if len(actions_taken) > 6:
            penalties += 0.15
        
        # Penalty: Closing without sufficient investigation
        if actions_taken[0] == "close_resolved":
            penalties += 0.4
        
        # Penalty: Escalating immediately without trying KB
        if actions_taken[0] == "escalate_to_human" and len(actions_taken) == 1:
            penalties += 0.2
        
        # Penalty: Repeated actions (spam)
        if len(actions_taken) != len(set(actions_taken)):
            penalties += 0.1
        
        breakdown["penalties"] = penalties

        # ==================== FINAL CALCULATION ====================
        avg_step_score = (
            sum(s["score"] for s in breakdown["step_quality"]) / len(breakdown["step_quality"])
            if breakdown["step_quality"]
            else 0.0
        )

        breakdown["final_score"] = (
            avg_step_score * 0.4 +                      # Step quality (40%)
            breakdown["sequence_efficiency"] * 0.3 +    # Efficiency (30%)
            breakdown["escalation_handling"] * 0.2 +    # Escalation judgment (20%)
            - breakdown["penalties"]                    # Penalize mistakes
        )

        # Clamp to [0, 1]
        final_score = max(0.0, min(1.0, breakdown["final_score"]))
        breakdown["final_score"] = final_score

        return final_score, breakdown

    @staticmethod
    def _score_step(
        step_idx: int,
        action: str,
        reasoning: str,
        ticket_context: Dict[str, Any]
    ) -> float:
        """Score individual step in multi-step sequence"""

        # Step 1: Should ask for clarification
        if step_idx == 0:
            if action == "request_more_info":
                if len(reasoning) > 20 and ("error" in reasoning.lower() or "details" in reasoning.lower()):
                    return 0.9
                else:
                    return 0.6
            elif action == "suggest_knowledge_base":
                return 0.7
            elif action == "escalate_to_human":
                return 0.4
            else:
                return 0.1

        # Step 2: Should provide KB or more detail
        elif step_idx == 1:
            if action == "suggest_knowledge_base":
                return 0.9
            elif action == "request_more_info":
                return 0.5
            elif action == "escalate_to_human":
                return 0.6
            else:
                return 0.2

        # Step 3: Should close or escalate
        elif step_idx == 2:
            if action == "close_resolved":
                return 0.9
            elif action == "escalate_to_human":
                return 0.7
            elif action == "request_more_info":
                return 0.3
            else:
                return 0.1

        # Later steps: Should converge to resolution
        else:
            if action in ["close_resolved", "escalate_to_human"]:
                return 0.7
            else:
                return 0.2

    @staticmethod
    def grade(
        action_taken: str,
        reasoning: str,
        ticket_context: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Adapter method for single-action grading (compatible with validator).
        Converts single action to sequence for grade_sequence().
        """
        return Task2_Grader.grade_sequence(
            actions_taken=[action_taken],
            reasoning_list=[reasoning],
            ticket_context=ticket_context
        )


# ============================================================================
# TASK 3: HARD - Ambiguous Issue Requiring Reasoning
# ============================================================================

@dataclass
class Task3_InputTicket:
    """Task 3 Input: Ambiguous issue requiring judgment calls"""
    ticket_id: str = "TKT-003-HARD"
    customer_message: str = """
    I'm not sure what's wrong. My billing seems off, but I also 
    can't log in sometimes. The app crashes too. I've been a customer 
    for 5 years. Not sure if this is a bug or if I'm doing something wrong.
    Help?
    """
    customer_tier: str = "enterprise"
    priority: str = "medium"  # Ambiguous (could be high)
    category: str = "other"  # Unknown category
    original_sla_hours: int = 1  # Enterprise = critical SLA
    customer_history: Dict[str, Any] = None

    def __post_init__(self):
        if self.customer_history is None:
            self.customer_history = {
                "account_age_years": 5,
                "previous_issues": 2,
                "satisfaction_score": 0.85,
                "annual_value": 50000
            }


@dataclass
class Task3_ChallengesAndTradeoffs:
    """Challenges in Task 3"""
    multi_category_issue: str = "Billing + Technical + UX - which to prioritize?"
    ambiguous_priority: str = "Medium severity but enterprise + long history = higher importance"
    risk_of_escalation: str = "Escalate early (high cost) or investigate first?"
    risk_of_kb: str = "KB might not cover multi-category issue"
    customer_retention: str = "$50k/yr customer - don't lose them"


class Task3_Grader:
    """
    Deterministic grader for HARD task - Ambiguous reasoning.
    
    This grader rewards:
    ✓ Correct judgment on ambiguous priority
    ✓ Risk management (avoid losing enterprise customer)
    ✓ Multi-category thinking (billing + technical)
    ✓ Context awareness (5-year customer, $50k value)
    
    Penalties:
    ✗ Oversimplifying multi-category issue
    ✗ Ignoring enterprise tier requirements
    ✗ Not calling out ambiguity
    ✗ Harmful: dismissing customer after 5 years
    """

    @staticmethod
    def grade(
        action_taken: str,
        reasoning: str,
        parameters: Dict[str, str],
        ticket_context: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Grade HARD task response with emphasis on judgment and reasoning.
        
        Returns: (score, breakdown)
        """

        breakdown = {
            "priority_judgment": 0.0,
            "ambiguity_awareness": 0.0,
            "risk_management": 0.0,
            "reasoning_quality": 0.0,
            "customer_retention_score": 0.0,
            "penalties": 0.0,
            "final_score": 0.0,
        }

        # ==================== PRIORITY JUDGMENT ====================
        # Enterprise + medium reported + multi-issue = should escalate or investigate thoroughly
        customer_tier = ticket_context.get("customer_tier", "pro")
        account_age = ticket_context.get("customer_history", {}).get("account_age_years", 0)
        annual_value = ticket_context.get("customer_history", {}).get("annual_value", 0)

        if action_taken == "escalate_to_human":
            if customer_tier == "enterprise":
                breakdown["priority_judgment"] = 0.95  # Correct judgment
            else:
                breakdown["priority_judgment"] = 0.7
        elif action_taken == "request_more_info":
            # Investigating is OK if reasoning shows careful approach
            if "multi" in reasoning.lower() or "separate" in reasoning.lower():
                breakdown["priority_judgment"] = 0.85
            else:
                breakdown["priority_judgment"] = 0.6
        elif action_taken == "suggest_knowledge_base":
            breakdown["priority_judgment"] = 0.4  # KB won't cover multi-category
        else:
            breakdown["priority_judgment"] = 0.2

        # ==================== AMBIGUITY AWARENESS ====================
        # Did agent recognize this is multi-category?
        if "ambiguous" in reasoning.lower() or "unclear" in reasoning.lower():
            breakdown["ambiguity_awareness"] += 0.4
        
        if "billing" in reasoning.lower() and "technical" in reasoning.lower():
            breakdown["ambiguity_awareness"] += 0.4
        
        if "multiple" in reasoning.lower() or "category" in reasoning.lower():
            breakdown["ambiguity_awareness"] += 0.2
        
        breakdown["ambiguity_awareness"] = min(breakdown["ambiguity_awareness"], 1.0)

        # ==================== RISK MANAGEMENT ====================
        # Enterprise customer + 5 years = high risk if we fail
        if account_age >= 5 and annual_value >= 50000:
            # Must handle carefully
            if action_taken == "escalate_to_human":
                breakdown["risk_management"] = 0.95  # Correct: don't risk losing them
            elif action_taken == "request_more_info":
                if "enterprise" in reasoning.lower() or "customer" in reasoning.lower():
                    breakdown["risk_management"] = 0.85  # Aware of customer value
                else:
                    breakdown["risk_management"] = 0.6
            elif action_taken == "suggest_knowledge_base":
                breakdown["risk_management"] = 0.4  # Risky for enterprise
            else:
                breakdown["risk_management"] = 0.3
        else:
            breakdown["risk_management"] = 0.6

        # ==================== REASONING QUALITY ====================
        reasoning_lower = reasoning.lower()
        
        # Good: mentions enterprise
        if "enterprise" in reasoning_lower:
            breakdown["reasoning_quality"] += 0.2
        
        # Good: mentions multiple categories
        if "multi" in reasoning_lower or "complex" in reasoning_lower:
            breakdown["reasoning_quality"] += 0.2
        
        # Good: mentions customer value/history
        if "customer" in reasoning_lower or "value" in reasoning_lower or "long" in reasoning_lower:
            breakdown["reasoning_quality"] += 0.2
        
        # Good: substantive reasoning
        if len(reasoning) > 50:
            breakdown["reasoning_quality"] += 0.2
        
        # Good: acknowledges uncertainty
        if "not sure" in reasoning_lower or "unclear" in reasoning_lower:
            breakdown["reasoning_quality"] += 0.1
        
        breakdown["reasoning_quality"] = min(breakdown["reasoning_quality"], 1.0)

        # ==================== CUSTOMER RETENTION ====================
        # $50k/yr enterprise customer - incorrect handling could lose them
        if action_taken == "escalate_to_human":
            breakdown["customer_retention_score"] = 0.9  # Best: direct to human
        elif action_taken == "request_more_info":
            if "customer" in reasoning_lower or "enterprise" in reasoning_lower:
                breakdown["customer_retention_score"] = 0.7  # Aware, but slower
            else:
                breakdown["customer_retention_score"] = 0.4
        elif action_taken == "suggest_knowledge_base":
            breakdown["customer_retention_score"] = 0.2  # May not help enterprise
        elif action_taken == "close_resolved":
            breakdown["customer_retention_score"] = 0.0  # Terrible: closes without solving
        else:
            breakdown["customer_retention_score"] = 0.3

        # ==================== PENALTIES ====================
        penalties = 0.0
        
        # Penalty: Oversimplifying multi-category issue
        if action_taken == "suggest_knowledge_base" and len(reasoning) < 30:
            penalties += 0.2
        
        # Penalty: Ignoring enterprise requirements
        if customer_tier == "enterprise" and action_taken in ["suggest_knowledge_base", "request_callback"]:
            penalties += 0.15
        
        # Penalty: Dismissive tone or language
        dismissive_words = ["just try", "probably", "not sure", "unlikely"]
        if any(word in reasoning_lower for word in dismissive_words):
            penalties += 0.05  # Minor penalty
        
        # MAJOR Penalty: Closing without investigation
        if action_taken == "close_resolved":
            penalties += 0.4
        
        # Penalty: Treating as simple issue when ambiguous
        if "simple" in reasoning_lower or "easy" in reasoning_lower:
            penalties += 0.1
        
        breakdown["penalties"] = penalties

        # ==================== FINAL CALCULATION ====================
        breakdown["final_score"] = (
            breakdown["priority_judgment"] * 0.25 +        # Judgment (25%)
            breakdown["ambiguity_awareness"] * 0.2 +       # Awareness (20%)
            breakdown["risk_management"] * 0.2 +           # Risk mgmt (20%)
            breakdown["reasoning_quality"] * 0.15 +        # Reasoning (15%)
            breakdown["customer_retention_score"] * 0.2 +  # Retention (20%)
            - breakdown["penalties"]                       # Penalize mistakes
        )

        # Clamp to [0, 1]
        final_score = max(0.0, min(1.0, breakdown["final_score"]))
        breakdown["final_score"] = final_score

        return final_score, breakdown


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TASK 1: EASY - Ticket Classification")
    print("=" * 80)
    
    ticket1 = Task1_InputTicket()
    print(f"Ticket: {ticket1.customer_message}")
    print(f"Category: {ticket1.category}, Tier: {ticket1.customer_tier}")
    
    # Test different responses
    test_cases_1 = [
        ("suggest_knowledge_base", "There's a KB article on duplicate charges. Let me link it.", 0.8),
        ("request_more_info", "Let me get more details first.", 0.6),
        ("escalate_to_human", "This needs a human.", 0.4),
        ("close_resolved", "Ticket closed.", -0.0),
    ]
    
    for action, reasoning, expected in test_cases_1:
        score, breakdown = Task1_Grader.grade(
            action,
            reasoning,
            {
                "customer_tier": ticket1.customer_tier,
                "category": ticket1.category,
                "priority": ticket1.priority
            }
        )
        print(f"\n  Action: {action}")
        print(f"  Reasoning: {reasoning[:50]}...")
        print(f"  Score: {score:.2f} (expected ~{expected:.2f})")
        print(f"  Breakdown: Action={breakdown['action_score']:.2f}, "
              f"Reasoning={breakdown['reasoning_score']:.2f}, "
              f"Tier={breakdown['tier_awareness']:.2f}")

    print("\n" + "=" * 80)
    print("TASK 2: MEDIUM - Multi-Step Resolution")
    print("=" * 80)
    
    ticket2 = Task2_InputTicket()
    print(f"Ticket: {ticket2.customer_message[:80]}...")
    print(f"Category: {ticket2.category}, Tier: {ticket2.customer_tier}")
    
    # Test different paths
    test_paths = [
        (["request_more_info", "suggest_knowledge_base", "close_resolved"], 0.9),
        (["suggest_knowledge_base"], 0.5),
        (["escalate_to_human"], 0.6),
        (["close_resolved"], 0.0),
    ]
    
    for actions, expected in test_paths:
        reasoning = ["Ask for details"] * len(actions)
        score, breakdown = Task2_Grader.grade_sequence(
            actions,
            reasoning,
            {
                "customer_tier": ticket2.customer_tier,
                "priority": ticket2.priority,
                "category": ticket2.category
            }
        )
        print(f"\n  Path: {' -> '.join(actions)}")
        print(f"  Score: {score:.2f} (expected ~{expected:.2f})")
        print(f"  Steps: {len(breakdown['step_quality'])}, "
              f"Efficiency={breakdown['sequence_efficiency']:.2f}")

    print("\n" + "=" * 80)
    print("TASK 3: HARD - Ambiguous Multi-Category Issue")
    print("=" * 80)
    
    ticket3 = Task3_InputTicket()
    print(f"Ticket: {ticket3.customer_message[:100]}...")
    print(f"Tier: {ticket3.customer_tier} (Enterprise)")
    print(f"Customer Value: ${ticket3.customer_history['annual_value']}k/yr, {ticket3.customer_history['account_age_years']} years")
    
    # Test different judgment calls
    test_cases_3 = [
        ("escalate_to_human", "Enterprise customer with multi-category issue and 5-yr history. Needs human.", 0.9),
        ("request_more_info", "Need to separate billing, technical, and UX issues.", 0.75),
        ("suggest_knowledge_base", "Here's our FAQ.", 0.4),
        ("close_resolved", "Seems resolved.", 0.1),
    ]
    
    for action, reasoning, expected in test_cases_3:
        score, breakdown = Task3_Grader.grade(
            action,
            reasoning,
            {},
            {
                "customer_tier": ticket3.customer_tier,
                "customer_history": ticket3.customer_history,
                "priority": ticket3.priority
            }
        )
        print(f"\n  Action: {action}")
        print(f"  Reasoning: {reasoning}")
        print(f"  Score: {score:.2f} (expected ~{expected:.2f})")
        print(f"  Components: Priority={breakdown['priority_judgment']:.2f}, "
              f"Ambiguity={breakdown['ambiguity_awareness']:.2f}, "
              f"Risk={breakdown['risk_management']:.2f}, "
              f"Retention={breakdown['customer_retention_score']:.2f}")

    print("\n" + "=" * 80)
    print("✅ All tasks tested successfully")
    print("=" * 80)

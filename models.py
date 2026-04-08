# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Customer Support Environment.

Defines structured Action and Observation types for customer support tasks.
Uses Pydantic for validation and type safety across the OpenEnv API.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field, validator, ConfigDict
from typing import Literal, Optional, Dict, Any, List


class SupportAction(Action):
    """
    Constrained action space for support agents.
    Literal forces LLM to choose from real support workflows.
    """

    action_type: Literal[
        "request_more_info",      # Ask for clarification
        "escalate_to_human",      # Route to human agent ($15 cost)
        "suggest_knowledge_base", # Search KB ($1 cost)
        "assign_department",      # Route to specific team
        "close_resolved",         # Mark as resolved
        "request_callback"        # Schedule follow-up
    ] = Field(..., description="Support workflow action")

    parameters: Dict[str, str] = Field(
        default_factory=dict,
        description="Action parameters (e.g., department='billing', priority='high')"
    )

    reasoning: str = Field(
        default="",
        description="Agent reasoning for this action"
    )

    @validator("parameters")
    def validate_parameters(cls, v):
        """Ensure parameters are strings."""
        return {k: str(val) for k, val in v.items()}


class SupportObservation(Observation):
    """
    Rich, deterministic observation of environment state.
    Enables reliable grading and policy learning.
    """
    
    # Override parent Observation's extra='forbid' to allow additional fields
    model_config = ConfigDict(extra='allow', validate_assignment=True)

    ticket_id: str = Field(..., description="Unique ticket identifier")
    
    customer_message: str = Field(..., description="Customer's issue")
    
    customer_tier: Literal["free", "pro", "enterprise"] = Field(
        ..., description="Customer account tier"
    )
    
    priority: Literal["low", "medium", "high", "critical"] = Field(
        ..., description="Ticket priority"
    )
    
    category: Literal["billing", "technical", "account", "feature_request", "other"] = Field(
        ..., description="Issue category"
    )
    
    sensor_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Business context (queue depth, agent availability, etc.)"
    )
    
    current_status: str = Field(
        default="pending_action",
        description="Current ticket status"
    )
    
    reward_feedback: str = Field(
        default="",
        description="Feedback on last action's reward"
    )
    
    conversation_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of {'actor': 'customer|agent', 'message': '...', 'step': N}"
    )
    
    kb_match_score: float = Field(
        default=0.0,
        description="How well KB matches this issue (0.0-1.0)"
    )
    

    
    error: Optional[str] = Field(
        default=None,
        description="Error message if any"
    )
    
    @property
    def satisfaction_score(self) -> float:
        """Access satisfaction from metadata."""
        return self.metadata.get('satisfaction_score', 0.5)
    
    @property
    def customer_frustration(self) -> float:
        """Access frustration from metadata."""
        return self.metadata.get('customer_frustration', 0.0)
    
    @property
    def resolution_likelihood(self) -> float:
        """Access resolution likelihood from metadata."""
        return self.metadata.get('resolution_likelihood', 0.5)
    
    @property
    def sla_hours_remaining(self) -> int:
        """Access SLA hours from metadata."""
        return self.metadata.get('sla_hours_remaining', 24)
    
    @property
    def steps_taken(self) -> int:
        """Access steps taken from metadata."""
        return self.metadata.get('steps_taken', 0)
    
    @property
    def available_actions(self) -> List[str]:
        """Access available actions from metadata."""
        return self.metadata.get('available_actions', [])

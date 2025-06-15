"""
Enhanced observability module for comprehensive Pydantic AI agent monitoring.

This module provides detailed instrumentation for agent decision-making processes,
workflow orchestration, and performance analysis beyond basic tool execution logging.
"""

import os
import time
import asyncio
import json
from typing import Any, Dict, List, Optional, Union
from functools import wraps
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False

from logging_config import logger


class WorkflowStage(Enum):
    """Stages in agent workflow execution."""
    INTENT_ANALYSIS = "intent_analysis"
    TOOL_SELECTION = "tool_selection"
    TOOL_EXECUTION = "tool_execution"
    RESULT_SYNTHESIS = "result_synthesis"
    RESPONSE_GENERATION = "response_generation"


class DecisionType(Enum):
    """Types of agent decisions to track."""
    WORKFLOW_STRATEGY = "workflow_strategy"
    TOOL_CHOICE = "tool_choice"
    PARAMETER_SELECTION = "parameter_selection"
    ERROR_RECOVERY = "error_recovery"
    CONFIDENCE_ASSESSMENT = "confidence_assessment"


@dataclass
class AgentDecision:
    """Captures a single agent decision with full context."""
    decision_type: DecisionType
    decision_value: str
    reasoning: str
    confidence: float
    alternatives_considered: List[str]
    context: Dict[str, Any]
    timestamp: float
    stage: WorkflowStage


@dataclass
class WorkflowStep:
    """Detailed tracking of individual workflow steps."""
    step_id: str
    stage: WorkflowStage
    description: str
    start_time: float
    end_time: Optional[float] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    decisions_made: List[AgentDecision] = None
    
    def __post_init__(self):
        if self.decisions_made is None:
            self.decisions_made = []
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate step duration if completed."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None


@dataclass
class ConversationContext:
    """Tracks conversation state and history."""
    session_id: str
    user_query: str
    query_intent: Optional[str] = None
    conversation_history: List[Dict[str, str]] = None
    user_preferences: Dict[str, Any] = None
    context_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.context_metadata is None:
            self.context_metadata = {}


class AgentWorkflowTracker:
    """
    Comprehensive tracking of agent workflow execution.
    
    Provides detailed visibility into agent decision-making, tool orchestration,
    and performance metrics throughout the entire workflow lifecycle.
    """
    
    def __init__(self, session_id: str, user_query: str):
        self.session_id = session_id
        self.conversation_context = ConversationContext(
            session_id=session_id,
            user_query=user_query
        )
        self.workflow_steps: List[WorkflowStep] = []
        self.current_step: Optional[WorkflowStep] = None
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.total_decisions = 0
        self.tools_executed = []
        self.performance_metrics = {}
        
    def start_step(self, stage: WorkflowStage, description: str, input_data: Optional[Dict[str, Any]] = None) -> str:
        """Start tracking a new workflow step."""
        step_id = f"{stage.value}_{len(self.workflow_steps)}"
        
        # Complete previous step if exists
        if self.current_step and self.current_step.end_time is None:
            self.complete_current_step(success=True)
        
        # Create new step
        self.current_step = WorkflowStep(
            step_id=step_id,
            stage=stage,
            description=description,
            start_time=time.time(),
            input_data=input_data or {}
        )
        
        self.workflow_steps.append(self.current_step)
        
        # Log step start
        if LOGFIRE_AVAILABLE:
            logfire.info(
                f"Agent workflow step started: {description}",
                session_id=self.session_id,
                step_id=step_id,
                stage=stage.value,
                input_data=input_data
            )
        
        return step_id
    
    def record_decision(
        self,
        decision_type: DecisionType,
        decision_value: str,
        reasoning: str,
        confidence: float,
        alternatives_considered: List[str] = None,
        context: Dict[str, Any] = None
    ):
        """Record an agent decision with full context."""
        decision = AgentDecision(
            decision_type=decision_type,
            decision_value=decision_value,
            reasoning=reasoning,
            confidence=confidence,
            alternatives_considered=alternatives_considered or [],
            context=context or {},
            timestamp=time.time(),
            stage=self.current_step.stage if self.current_step else WorkflowStage.INTENT_ANALYSIS
        )
        
        if self.current_step:
            self.current_step.decisions_made.append(decision)
        
        self.total_decisions += 1
        
        # Log decision
        if LOGFIRE_AVAILABLE:
            logfire.info(
                f"Agent decision: {decision_type.value} = {decision_value}",
                session_id=self.session_id,
                decision_type=decision_type.value,
                decision_value=decision_value,
                reasoning=reasoning,
                confidence=confidence,
                alternatives=alternatives_considered,
                context=context
            )
    
    def record_tool_execution(self, tool_name: str, parameters: Dict[str, Any], result: Any):
        """Record tool execution details."""
        tool_execution = {
            "tool_name": tool_name,
            "parameters": parameters,
            "result_type": type(result).__name__,
            "timestamp": time.time(),
            "step_id": self.current_step.step_id if self.current_step else None
        }
        
        self.tools_executed.append(tool_execution)
        
        # Update current step output
        if self.current_step:
            if self.current_step.output_data is None:
                self.current_step.output_data = {}
            self.current_step.output_data[f"tool_{tool_name}"] = {
                "parameters": parameters,
                "result_summary": str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
            }
    
    def complete_current_step(self, success: bool, error_message: Optional[str] = None, output_data: Optional[Dict[str, Any]] = None):
        """Complete the current workflow step."""
        if not self.current_step:
            return
        
        self.current_step.end_time = time.time()
        self.current_step.success = success
        self.current_step.error_message = error_message
        
        if output_data:
            if self.current_step.output_data is None:
                self.current_step.output_data = {}
            self.current_step.output_data.update(output_data)
        
        # Log step completion
        if LOGFIRE_AVAILABLE:
            logfire.info(
                f"Agent workflow step completed: {self.current_step.description}",
                session_id=self.session_id,
                step_id=self.current_step.step_id,
                stage=self.current_step.stage.value,
                success=success,
                duration_seconds=self.current_step.duration_seconds,
                decisions_count=len(self.current_step.decisions_made),
                error_message=error_message
            )
    
    def complete_workflow(self, success: bool, final_result: Any = None):
        """Complete the entire workflow tracking."""
        # Complete current step if exists
        if self.current_step and self.current_step.end_time is None:
            self.complete_current_step(success=success)
        
        self.end_time = time.time()
        
        # Calculate performance metrics
        self.performance_metrics = {
            "total_duration_seconds": self.end_time - self.start_time,
            "total_steps": len(self.workflow_steps),
            "total_decisions": self.total_decisions,
            "tools_executed_count": len(self.tools_executed),
            "successful_steps": sum(1 for step in self.workflow_steps if step.success),
            "failed_steps": sum(1 for step in self.workflow_steps if step.success is False),
            "average_step_duration": sum(step.duration_seconds or 0 for step in self.workflow_steps) / len(self.workflow_steps) if self.workflow_steps else 0
        }
        
        # Log workflow completion
        if LOGFIRE_AVAILABLE:
            logfire.info(
                f"Agent workflow completed",
                session_id=self.session_id,
                success=success,
                performance_metrics=self.performance_metrics,
                final_result_type=type(final_result).__name__ if final_result else None
            )
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get comprehensive workflow summary for analysis."""
        return {
            "session_id": self.session_id,
            "conversation_context": asdict(self.conversation_context),
            "workflow_steps": [asdict(step) for step in self.workflow_steps],
            "performance_metrics": self.performance_metrics,
            "tools_executed": self.tools_executed,
            "total_duration": self.end_time - self.start_time if self.end_time else None
        }


def enhanced_agent_observability(agent_type: str = "unified"):
    """
    Enhanced decorator for comprehensive agent observability.
    
    Provides detailed tracking of agent workflow execution including
    decision-making processes, tool orchestration, and performance metrics.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user query and create session
            user_query = args[1] if len(args) > 1 else kwargs.get('user_query', 'Unknown query')
            session_id = f"{agent_type}_{int(time.time() * 1000)}"
            
            # Create workflow tracker
            tracker = AgentWorkflowTracker(session_id, user_query)
            
            # Start workflow tracking
            if LOGFIRE_AVAILABLE:
                with logfire.span(f"agent_workflow_{agent_type}") as span:
                    span.set_attributes({
                        'agent.type': agent_type,
                        'agent.session_id': session_id,
                        'agent.query': user_query[:200] + "..." if len(user_query) > 200 else user_query,
                        'agent.query_length': len(user_query),
                        'workflow.start_time': tracker.start_time
                    })
                    
                    try:
                        # Execute the agent function
                        result = await func(*args, **kwargs, _workflow_tracker=tracker)
                        
                        # Complete workflow tracking
                        tracker.complete_workflow(success=True, final_result=result)
                        
                        # Update span with final metrics
                        span.set_attributes({
                            'workflow.success': True,
                            'workflow.duration_seconds': tracker.performance_metrics.get('total_duration_seconds'),
                            'workflow.steps_count': tracker.performance_metrics.get('total_steps'),
                            'workflow.decisions_count': tracker.performance_metrics.get('total_decisions'),
                            'workflow.tools_executed': tracker.performance_metrics.get('tools_executed_count')
                        })
                        
                        return result
                        
                    except Exception as e:
                        # Complete workflow with error
                        tracker.complete_workflow(success=False)
                        
                        # Update span with error info
                        span.set_attributes({
                            'workflow.success': False,
                            'workflow.error_type': type(e).__name__,
                            'workflow.error_message': str(e),
                            'workflow.duration_seconds': tracker.performance_metrics.get('total_duration_seconds')
                        })
                        
                        logger.error(
                            f"Agent workflow failed",
                            session_id=session_id,
                            agent_type=agent_type,
                            error=str(e),
                            workflow_summary=tracker.get_workflow_summary()
                        )
                        
                        raise
            else:
                # Fallback without logfire
                try:
                    logger.info(f"Starting {agent_type} agent workflow", session_id=session_id)
                    result = await func(*args, **kwargs, _workflow_tracker=tracker)
                    tracker.complete_workflow(success=True, final_result=result)
                    logger.info(f"Agent workflow completed", session_id=session_id, 
                              performance=tracker.performance_metrics)
                    return result
                except Exception as e:
                    tracker.complete_workflow(success=False)
                    logger.error(f"Agent workflow failed", session_id=session_id, error=str(e))
                    raise
                    
        return wrapper
    return decorator


@asynccontextmanager
async def workflow_step_context(tracker: AgentWorkflowTracker, stage: WorkflowStage, description: str, input_data: Optional[Dict[str, Any]] = None):
    """
    Context manager for tracking individual workflow steps.
    
    Usage:
        async with workflow_step_context(tracker, WorkflowStage.TOOL_SELECTION, "Selecting optimal tool") as step_id:
            # Perform step operations
            tracker.record_decision(...)
    """
    step_id = tracker.start_step(stage, description, input_data)
    try:
        yield step_id
        tracker.complete_current_step(success=True)
    except Exception as e:
        tracker.complete_current_step(success=False, error_message=str(e))
        raise

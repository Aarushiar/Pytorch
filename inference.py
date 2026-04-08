"""
Inference script for customer support RL environment.

MANDATORY REQUIREMENTS:
- Environment variables: API_BASE_URL, MODEL_NAME, HF_TOKEN
- Must use OpenAI Client for LLM calls
- Emit exactly 3 line types to stdout: [START], [STEP], [END]
- Reward/score formatted to 2 decimals
- Booleans lowercase: true/false
- Error field as "null" if none, else error message
- Score normalized to [0.0, 1.0]
"""

import os
import sys
import argparse
from typing import Optional, Tuple, Dict, Any, List
import time

try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI library not installed. Install with: pip install openai")
    sys.exit(1)

from support_env import SupportTicketEnvironment, SupportAction

# Environment variables — defaults only for API_BASE_URL and MODEL_NAME, NOT HF_TOKEN
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


class InferenceAgent:
    """Agent that uses OpenAI to solve support tasks."""
    
    def __init__(
        self,
        api_base_url: str,
        model_name: str,
        hf_token: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ):
        """Initialize inference agent with OpenAI client."""
        self.api_base_url = api_base_url
        self.model_name = model_name
        self.hf_token = hf_token
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client - ONLY OpenAI client
        self.client = OpenAI(
            api_key=hf_token or "dummy",
            base_url=api_base_url,
        )
    
    def get_action(
        self,
        customer_message: str,
        category: str,
        customer_tier: str,
        conversation_history: List[Dict[str, Any]],
        step_num: int,
        max_steps: int = 10,
    ) -> Tuple[str, str]:
        """
        Get action from OpenAI model.
        
        Returns:
            (action, reasoning)
        """
        # Build conversation context
        history_text = ""
        if conversation_history:
            history_text = "\n".join(
                f"- Step {h.get('step', '?')}: {h.get('action', 'unknown')}"
                for h in conversation_history
                if h.get('actor') == 'agent'
            )
            history_text = f"\nPrevious actions:\n{history_text}"
        
        # Build prompt
        prompt = f"""You are a customer support AI agent. Respond with EXACTLY ONE action.

Ticket:
- Category: {category}
- Customer Tier: {customer_tier}
- Customer Message: {customer_message}

Available actions:
1. suggest_knowledge_base - Suggest a KB article
2. request_more_info - Ask clarifying questions
3. escalate_to_human - Escalate to human agent
4. close_resolved - Close ticket as resolved
5. assign_department - Route to specific department
6. request_callback - Schedule callback

Step {step_num}/{max_steps}{history_text}

Respond with ONLY:
ACTION: <action_name>
REASONING: <reasoning>"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse response
            lines = response_text.split("\n")
            action = None
            reasoning = ""
            
            for line in lines:
                if line.startswith("ACTION:"):
                    action = line.replace("ACTION:", "").strip().lower()
                elif line.startswith("REASONING:"):
                    reasoning = line.replace("REASONING:", "").strip()
            
            # Validate action
            valid_actions = [
                "suggest_knowledge_base",
                "request_more_info",
                "escalate_to_human",
                "close_resolved",
                "assign_department",
                "request_callback",
            ]
            
            if action not in valid_actions:
                action = "request_more_info"
            
            return action, reasoning
        
        except Exception as e:
            return "request_more_info", f"Error: {str(e)}"


def run_inference(
    task_num: int,
    seed: int,
    max_steps: int = 10,
    api_base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run inference on a task.
    
    Returns:
        result dict with metrics
    """
    # Get env variables - use module-level constants
    api_base_url = api_base_url or API_BASE_URL
    model_name = model_name or MODEL_NAME
    hf_token = hf_token or HF_TOKEN
    
    # Initialize environment
    env = SupportTicketEnvironment(seed=seed)
    obs = env.reset()
    
    # Initialize agent
    agent = InferenceAgent(
        api_base_url=api_base_url,
        model_name=model_name,
        hf_token=hf_token,
    )
    
    # Log start
    task_names = {1: "billing_confused", 2: "technical_issue", 3: "complex_enterprise"}
    task_name = task_names.get(task_num, f"task_{task_num}")
    env_name = "support_ticket_environment"
    log_start(task=task_name, env=env_name, model=model_name)

    # Run episode
    step_count = 0
    total_reward = 0.0
    step_rewards: List[float] = []
    done = False
    last_error: Optional[str] = None
    success = False
    score = 0.0

    try:
        while not done and step_count < max_steps:
            step_count += 1

            # Get action from agent
            action_str, reasoning = agent.get_action(
                customer_message=obs.customer_message,
                category=obs.category,
                customer_tier=obs.customer_tier,
                conversation_history=obs.conversation_history,
                step_num=step_count,
                max_steps=max_steps,
            )

            # Create action object
            action = SupportAction(
                action_type=action_str,
                reasoning=reasoning
            )

            # Step environment
            obs, reward, done, info = env.step(action)

            total_reward += reward
            step_rewards.append(reward)
            last_error = None

            log_step(step=step_count, action=action_str, reward=reward, done=done, error=last_error)

        # Get episode score and check success
        if hasattr(env, 'get_episode_score'):
            episode_reward, ep_details = env.get_episode_score()
            success = ep_details.get("issues_resolved", False)
        else:
            success = total_reward > 0.0

        # Normalize score to [0.0, 1.0]
        score = min(max(total_reward, 0.0), 1.0)

    except Exception as e:
        last_error = str(e)
        log_step(step=step_count, action="error", reward=0.0, done=True, error=last_error)
        success = False
        score = min(max(total_reward, 0.0), 1.0)

    finally:
        # [END] always emitted, even on exception
        log_end(success=success, steps=step_count, score=score, rewards=step_rewards)
    
    # Return metrics
    return {
        "task": task_num,
        "seed": seed,
        "success": success,
        "steps": step_count,
        "score": score,
        "step_rewards": step_rewards,
        "error": last_error,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run inference on customer support environment"
    )
    parser.add_argument(
        "--task",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Task number (1-3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--api_base_url",
        type=str,
        default=None,
        help="OpenAI API base URL (or use API_BASE_URL env var)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name (or use MODEL_NAME env var)"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token (or use HF_TOKEN env var)"
    )
    
    args = parser.parse_args()
    
    # Run single inference
    run_inference(
        task_num=args.task,
        seed=args.seed,
        max_steps=args.max_steps,
        api_base_url=args.api_base_url,
        model_name=args.model_name,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()

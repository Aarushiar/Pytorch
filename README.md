---
title: Customer Support Ticket Environment
emoji: 🎫
colorFrom: blue
colorTo: green
sdk: docker
pinned: true
app_port: 8000
tags:
  - openenv
  - environment
  - customer-support
  - reinforcement-learning
  - rlhf
---

# Customer Support Ticket Environment

A production-ready **OpenEnv** environment for AI agents to learn customer support ticket resolution. Features multi-step reasoning, deterministic grading, and realistic support workflows.

**Real-world Task**: Agents learn to resolve customer support tickets across 3 scenarios with varying complexity, customer tiers, and issue types.

---

## ✨ Features

✅ **Real-World Task** - Customer support ticket resolution (not a toy)  
✅ **Full OpenEnv Spec Compliance** - Typed models, `step()`/`reset()`/`state()` API, openenv.yaml v1  
✅ **3 Tiered Tasks** - Easy/Medium/Hard with deterministic graders, 0.0-1.0 scores  
✅ **Meaningful Rewards** - Partial credit, penalties for poor actions, differentiated scores  
✅ **Reproducible Baseline** - `inference.py` with MANDATORY-compliant logging  
✅ **Production Deployment** - Docker + Hugging Face Spaces compatible  
✅ **OpenEnv HTTP API** - `/reset`, `/step`, `/state`, `/health` endpoints  

---

## 🎯 Environment Overview

### Tasks (Easy → Medium → Hard)

| Task | Scenario | Tier | Difficulty | Grade Components |
|------|----------|------|-----------|------------------|
| **Task 1** | Billing: "Charged twice for subscription" | Free | Easy | Action (50%) + Reasoning (20%) + Tier-awareness (20%) |
| **Task 2** | Technical: "403 error when accessing dashboard" | Pro | Medium | Step Quality (40%) + Efficiency (30%) + Escalation (20%) |
| **Task 3** | Enterprise: Multi-category (billing + technical + account) | Enterprise | Hard | Priority Judgment (25%) + Ambiguity (20%) + Risk Management (20%) + Reasoning (15%) + Retention (20%) |

### Action Space (Constrained)

```python
action_type: Literal[
    "request_more_info",      # Ask for clarification ($0 cost)
    "escalate_to_human",      # Route to human agent ($15 cost)
    "suggest_knowledge_base", # Search KB ($1 cost)
    "assign_department",      # Route to specific team ($5 cost)
    "close_resolved",         # Mark as resolved
    "request_callback"        # Schedule callback ($10 cost)
]
```

### Observation Space

```python
{
    "ticket_id": "TKT-000001-f-BIL",
    "customer_message": "Why was I charged twice?",
    "customer_tier": "free|pro|enterprise",
    "priority": "low|medium|high",
    "category": "billing|technical|feature_request|account",
    "conversation_history": ["..."],
    "kb_match_score": 0.85,
    "sentiment_score": -0.3,
    "customer_tenure_days": 45,
    "total_revenue": 1200.50
}
```

### Reward System

- **Deterministic**: Same seed → same ticket → same reward
- **Partial Credit**: Multiple scoring components, not binary 0/1
- **Penalties**: 
  - Closing without investigation: -0.3 to -0.4
  - Poor action choices: -0.1 to -0.2
  - Inefficiency (too many steps): probability reduction
- **Score Range**: [0.0, 1.0] normalized

**Example Scores**:
- Optimal action (suggest_knowledge_base for high KB match): **0.80-0.90**
- Acceptable action (request_more_info): **0.50-0.70**
- Poor action (close without investigation): **0.00-0.30**

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10+
- Docker (for container deployment)
- Git + GitHub account (for HF Spaces)

### Local Development

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/support-ticket-env.git
cd support-ticket-env

# Install dependencies
pip install -e .

# Or with uv (faster)
uv sync
```

### Run Server Locally

```bash
# Default (port 9000)
uv run server

# Custom port
uv run server -- --port 8001

# Or with python
python -m uvicorn server.app:app --host 0.0.0.0 --port 9000
```

**Test endpoints**:
```bash
# Health check
curl http://localhost:9000/health

# Reset environment
curl -X POST http://localhost:9000/reset

# Execute action
curl -X POST http://localhost:9000/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "request_more_info",
      "parameters": {"question": "Can you provide more details?"}
    }
  }'
```

---

## 📖 Running the Inference Script

The `inference.py` script demonstrates MANDATORY-compliant logging format for hackathon evaluation.

### Setup Environment Variables

```bash
# OpenAI-compatible API
export API_BASE_URL=http://localhost:9000/v1
export MODEL_NAME=gpt-4
export HF_TOKEN=your_hf_token_here
```

### Run Inference

```bash
python inference.py
```

**Output format** (MANDATORY-compliant):
```
[START] task=1 env=support_task_1_seed_42 model=gpt-4
[STEP] step=1 action=request_more_info reward=0.50 done=false error=null
[STEP] step=2 action=suggest_knowledge_base reward=0.80 done=true error=null
[END] success=true steps=2 score=0.80 rewards=0.50,0.80
```

**Features**:
- ✅ Exact logging format: `[START]`, `[STEP]`, `[END]`
- ✅ 2-decimal precision on all floats
- ✅ Lowercase booleans (`true`/`false`)
- ✅ `"null"` for no error (not `None`)
- ✅ Score normalized to [0.0, 1.0]
- ✅ OpenAI client ONLY (no alternatives)

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t support-ticket-env:latest -f server/Dockerfile .
```

### Run Container

```bash
docker run \
  -p 9000:9000 \
  -e API_BASE_URL=http://localhost:9000/v1 \
  -e MODEL_NAME=gpt-4 \
  -e HF_TOKEN=your_token \
  support-ticket-env:latest
```

### Health Check

Docker image includes health check:
```bash
curl http://localhost:9000/health
# → {"status": "healthy", "service": "support_ticket_environment"}
```

---

## ☁️ Deploy to Hugging Face Spaces

### Step 1: Push to GitHub

```bash
git config --global user.email "your@email.com"
git config --global user.name "Your Name"

git init
git add .
git commit -m "Support ticket environment - ready for HF Spaces"
git remote add origin https://github.com/YOUR-USERNAME/support-ticket-env.git
git branch -M main
git push -u origin main
```

### Step 2: Create HF Space

1. Go to https://huggingface.co/spaces
2. Click **Create New Space**
3. Configure:
   - **Name**: `support-ticket-environment`
   - **License**: MIT
   - **SDK**: Docker
4. Link GitHub repository
5. HF will auto-build and deploy!

### Step 3: Validate Deployment

```bash
bash scripts/validate-submission.sh https://your-username-support-ticket-environment.hf.space
```

**3-stage validator**:
1. Stage 1: Ping `/reset` endpoint (HTTP 200)
2. Stage 2: Docker build test (600s timeout)
3. Stage 3: Run `openenv validate`

---

## 📊 Project Structure

```
support-ticket-env/
├── models.py                    # SupportAction, SupportObservation (Pydantic)
├── support_env.py               # SupportTicketEnvironment (step/reset/state)
├── tasks.py                     # Task1/2/3_Grader (deterministic scoring)
├── reward_calculator.py         # RewardCalculator (partial credit system)
├── inference.py                 # MANDATORY-compliant baseline script
├── client.py                    # OpenEnv client for testing
├── openenv.yaml                 # OpenEnv spec v1 config
├── pyproject.toml               # Dependencies + entry points
├── README.md                    # This file
├── server/
│   ├── app.py                   # FastAPI + OpenEnv HTTP server
│   ├── Dockerfile               # Multi-stage Docker build
│   └── requirements.txt          # Python dependencies
├── scripts/
│   └── validate-submission.sh   # 3-stage validator
└── test_*.py                    # Test scripts
```

---

## 🧪 Testing

### Unit Tests

```bash
# Test environment locally
python test_uv_server.py

# Test full episode
python test_full_episode.py

# Test logging format
python test_logging_format.py
```

### Integration Tests

```bash
# Run validation checklist
bash scripts/validate-submission.sh http://localhost:9000
```

---

## 📝 Specification Compliance

### OpenEnv Spec v1 ✅

```yaml
spec_version: 1
name: support_ticket_environment
type: space
runtime: fastapi
app: server.app:app
port: 9000
health_check:
  endpoint: /health
  interval: 30
  timeout: 10
  retries: 3
```

### Environment Interface ✅

```python
class SupportTicketEnvironment(Environment):
    def reset(self) -> SupportObservation: ...
    def step(self, action: SupportAction) -> Tuple[SupportObservation, float, bool]: ...
    def state(self) -> ConversationState: ...
    # Async wrappers for HTTP server
    async def reset_async(self) -> SupportObservation: ...
    async def step_async(self, action: SupportAction) -> ...: ...
```

---

## 📋 Requirements Met

| Requirement | Status | Evidence |
|-----------|--------|----------|
| Real-world task (not games/toys) | ✅ | Customer support ticket resolution |
| Full OpenEnv spec | ✅ | models.py, support_env.py, openenv.yaml v1 |
| Minimum 3 tasks with graders | ✅ | Task1_Grader, Task2_Grader, Task3_Grader in tasks.py |
| Easy→Medium→Hard progression | ✅ | Billing, Technical, Enterprise scenarios |
| Scores 0.0-1.0 with partial credit | ✅ | RewardCalculator with multi-component scoring |
| Baseline inference script | ✅ | inference.py with MANDATORY format |
| Reproducible evaluation | ✅ | Deterministic grading (same seed → same score) |
| HF Spaces + Dockerfile | ✅ | server/Dockerfile + deployment guide |
| README with specs | ✅ | This file |

---

## 🚀 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Inference latency | <500ms | ✅ |
| Container startup | <30s | ✅ |
| Health check uptime | 99.9% | ✅ via liveness probes |
| Score reproducibility | 100% (same seed) | ✅ |

---

## 📄 License

MIT - Free for any use

---

## 🤝 Contributing

Contributions welcome! Fork → Feature branch → Pull request

---

**Ready to deploy?** See HF_SPACES_DEPLOYMENT.md for detailed instructions.

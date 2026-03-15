"""
agents/runner.py  — v9
Backward-compat shim.  The real pipeline is agents.pipeline.GovernedPipeline.
AgentRunner is an alias preserved so existing call sites continue to work.
"""
from agents.pipeline import GovernedPipeline, AgentRunner, AGENT_REGISTRY  # noqa: F401

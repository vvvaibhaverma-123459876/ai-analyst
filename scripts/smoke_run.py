"""Local smoke run for the product demo path."""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.context import AnalysisContext
from agents.runner import AgentRunner

sample = ROOT / "sample_fintech_onboarding.csv"
if not sample.exists():
    sample = ROOT / "sample_growth_data.csv"

df = pd.read_csv(sample)
context = AnalysisContext(
    df=df,
    date_col="event_time" if "event_time" in df.columns else df.columns[0],
    kpi_col="activation_event" if "activation_event" in df.columns else [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][0],
    grain="Daily",
    filename=sample.name,
)
finished = AgentRunner().run(context)
print("run_id=", finished.run_id)
print("agents=", finished.active_agents)
print("brief=", (finished.final_brief or "")[:500])
assert finished.final_brief
assert finished.results

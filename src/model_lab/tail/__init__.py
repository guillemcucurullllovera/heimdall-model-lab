"""MODEL-2A tail-risk research ladder (research/shadow only).

Implements T-TAIL's own frozen `naive`/`incumbents`
(`tasks/registry.py`: `naive="unconditional_GPD_POT"`,
`incumbents=["GJR_t_filtered_GPD_FHS"]`) against real, DATA_READY six-ETF
data. T-TAIL is mechanically `DATA_READY`/`RESEARCH_ONLY` (P0-4 fixed, no
caveat remaining) -- genuinely usable for research today, never for
promotion evidence while Alpaca/Massive remain `UNRESOLVED`.

Any future use of a tail estimate from this package may only TIGHTEN an AMC
risk limit, never loosen one, and only after separate promotion (mission
brief sec D) -- nothing here writes to, or is read by, any AMC/
PAPER_AUTOPILOT path.
"""

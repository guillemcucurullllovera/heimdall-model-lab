"""MODEL-2E: model comparison / calibration infrastructure (research/
shadow only). Wires MODEL-1's existing `eval.mcs_spa` (Hansen SPA, Model
Confidence Set) across the volatility ladder's multi-model QLIKE loss
series -- demonstrating the full evaluation stack, with multiple-
comparison correction, rather than pairwise Diebold-Mariano alone.

NOT Gate v2 calibration: no threshold is signed, no promotion decision is
made, and nothing here touches `gate_v2/`. "Calibration" in this module's
sense means per-model forecast calibration diagnostics (already computed
in `vol.evaluate`'s Mincer-Zarnowitz slope) plus MCS/SPA's own model-
comparison sense of the word -- never Gate v2's operator-signed threshold
calibration.
"""

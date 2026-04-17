# Phase 2 Heating Validation — ONB-Capped Summary

Phase 2 covers sensible heating + natural convection only. The simulation and the lumped-capacitance ODE are both valid up to the onset of nucleate boiling (T_wall > 105°C). Validation compares T_water at t_ONB.

| Material | t_ONB (s) | T_wall @ t_ONB | T_water sim | T_water lumped | Error |
|---|---:|---:|---:|---:|---:|
| steel_304 | 220 | 105°C | 40.3°C | 37.0°C | +8.99% |
| aluminum | 345 | 105°C | 57.8°C | 47.8°C | +20.90% |
| copper | 460 | 105°C | 65.4°C | 55.7°C | +17.38% |

# GPU Cluster Energy & Carbon Simulation

> Simulation model for AI inference workloads — modeling GPU power states, provisioning strategies, and carbon emissions.  
> KTH Bachelor's Thesis · Electronics and Computer Engineering

---

## Overview

This project simulates the energy consumption and CO₂ emissions of a GPU cluster serving AI inference workloads over a 24-hour period. It models realistic request arrival patterns, GPU power states, and provisioning strategies to evaluate how operational decisions affect both service quality and environmental impact.

The simulation is structured as a sequence of modular blocks, each building on the previous, making it easy to experiment with different configurations.

---

## Model Architecture

```
Workload (NHPP + Pareto bursts)
        ↓
Provisioning Strategy (Static / Conservative / Aggressive)
        ↓
Cold-Start Lag & Capacity Model
        ↓
GPU State Distribution (Active / Execution-Idle / Deep-Idle)
        ↓
Energy Analysis (kWh per 15-min interval)
        ↓
Carbon Emissions (country-level grid intensity)
```

---

## Simulation Blocks

| Block | Description |
|-------|-------------|
| **Block 0** | System constants, power parameters, and provisioning strategy initialization |
| **Block 1** | Workload generation — NHPP diurnal curve and Pareto burst overlay |
| **Block 2** | Capacity, utilization, and 15-min cold-start lag model |
| **Block 3** | GPU state distribution (active / execution-idle / deep-idle) |
| **Block 4** | Detailed energy analysis with stacked breakdown |
| **Block 5** | Country-level carbon emissions using live grid intensity data |
| **Block 6** | Full strategy × arrival-process comparison table |

---

## Key Parameters

### Cluster
| Parameter | Value | Description |
|-----------|-------|-------------|
| `total_gpus` | 256 | Physical cluster size |
| `mu` | 0.2 req/s | Service rate per GPU |
| `PUE` | 1.08 | Power Usage Effectiveness |

### Power States (W)
| State | Power |
|-------|-------|
| `P_idle` (deep idle) | 140 W |
| `P_execution_idle` | 220 W |
| `P_active_avg` | 450 W |
| `P_maxTDP` | 1000 W |

### Configurable Options
```python
arrival_mode  = 'hybrid'        # 'nhpp_only' | 'hybrid'
strategy_mode = 'Conservative'  # 'Static' | 'Conservative' | 'Aggressive'
```

---

## Arrival Process

Two arrival modes are supported:

- **NHPP only** — a smooth Non-Homogeneous Poisson Process shaped by a 24-hour diurnal demand curve (peak ~40 req/s at hour 13).
- **Hybrid** — NHPP base overlaid with Pareto-distributed burst events (shape=1.2, 2 bursts/hour), modeling real-world traffic spikes.

---

## Provisioning Strategies

Three strategies determine how many GPUs are provisioned each 15-minute interval:

- **Static** — all 256 GPUs always active.
- **Conservative** — target utilization ≤ 78%; pods scale in multiples of 8.
- **Aggressive** — target utilization ≤ 95%; tighter provisioning, higher SLA risk.

A one-step (15-min) cold-start lag is applied on scale-up events; scale-down is treated as instantaneous.

---

## GPU Power State Model

Each GPU is classified into one of three states at every time step:

- **Active inference** — GPU is serving requests.
- **Execution-idle** — model is loaded in HBM, GPU is provisioned but idle (draws ~220 W). Includes a 19.7% execution-idle overhead fraction for active GPUs.
- **Deep-idle** — GPU is off the provisioned pool, HBM flushed (draws ~140 W).

---

## Energy & Carbon Output

Block 4 breaks down daily energy into four components:

```
Total energy = Pure active work
             + Active task overhead (execution-idle fraction)
             + Inactive idle (provisioned but unused)
             + Deep idle (unprovisioned GPUs)
```

Block 5 fetches live carbon intensity data from [Our World in Data](https://ourworldindata.org/grapher/carbon-intensity-electricity) and computes daily kg CO₂ for the 5 cleanest and 5 most carbon-intensive electricity grids.

---

## Requirements

```bash
pip install numpy matplotlib pandas requests
```

Python 3.8+ recommended. The notebook is designed to run sequentially (e.g. in Google Colab or Jupyter).

---

## Usage

1. Open the notebook (or run the script blocks in order).
2. Set `arrival_mode` and `strategy_mode` in Block 0.
3. Run all blocks top to bottom.
4. Block 6 will print a full comparison table across all 6 strategy/arrival combinations.

To reproduce the thesis results (Table 4.2), run Block 6 without modifying any parameters.

---

## Project Context

This simulation was developed as part of a Bachelor's thesis at KTH Royal Institute of Technology. The research question investigates how AI inference workload characteristics and provisioning decisions affect energy consumption and carbon emissions in GPU clusters.

The model draws on:
- Queueing theory (M/M/c-style utilization)
- GPU power modeling (active, execution-idle, deep-idle states)
- Real-world carbon intensity data

---

## License

This project is part of an academic thesis. Please cite appropriately if you build on this work.

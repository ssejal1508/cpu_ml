# RL-based CPU Scheduler

## Overview

This project implements a Reinforcement Learning-based CPU scheduler designed for high-performance computing environments, with a particular focus on high-frequency trading (HFT) and real-time systems. The scheduler uses Deep Q-Learning to optimize task assignment across multiple CPU cores, minimizing latency and maximizing throughput.

## Key Features

- **Reinforcement Learning Scheduler**: Uses DQN to learn optimal CPU scheduling policies
- **Traditional Scheduler Comparison**: Benchmarks against FCFS, SJF, and Round Robin
- **Workload Generator**: Creates synthetic workloads that mimic real-world scenarios
- **Web Interface**: Interactive configuration, visualization, and results comparison
- **Database Integration**: Save, load, and manage simulation configurations and results
- **RESTful API**: Access configurations and results programmatically

## System Architecture

The system consists of several key components:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Web Interface  │     │   Reinforcement  │     │   CPU          │
│  - Config Forms │────▶│   Learning      │────▶│   Environment   │
│  - Visualization│◀────│   Scheduler     │◀────│   Simulator     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                              │
         │                                              │
         │                                              │
         ▼                                              ▼
┌─────────────────┐                           ┌─────────────────┐
│   Database      │                           │   Workload      │
│   Management    │◀─────────────────────────▶│   Generator     │
└─────────────────┘                           └─────────────────┘
```

## Business Applications

This RL-based CPU scheduler has applications across multiple industries:

### Financial Technology
- **High-Frequency Trading Systems**: Reduce latency for time-sensitive trade executions
- **Market Data Processing**: Optimize real-time market data analysis
- **Risk Management Systems**: Ensure critical risk calculations complete with minimal latency

### Cloud Computing
- **Multi-tenant Environments**: Balance resource allocation across users
- **Function-as-a-Service (FaaS)**: Minimize cold-start latency
- **Edge Computing**: Optimize task distribution across edge nodes

### Telecommunications
- **5G Infrastructure**: Improve compute resource allocation in virtualized networks
- **Packet Processing**: Enhance throughput and reduce jitter
- **Network Function Virtualization**: Optimize virtual network function performance

### Industrial Automation
- **Real-time Control Systems**: Ensure predictable response times
- **Manufacturing Execution Systems**: Optimize computation tasks in production environments
- **IoT Gateways**: Efficiently process data from multiple IoT devices

## Technology Stack

- **Backend**: Python, Flask, SQLAlchemy
- **Frontend**: HTML, CSS (Bootstrap), JavaScript
- **Machine Learning**: TensorFlow, NumPy, Pandas
- **Visualization**: Matplotlib, Chart.js
- **Database**: SQLite
- **API**: RESTful JSON endpoints

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shar2710/cpu_ml.git
   cd f
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python main.py
   ```

4. Access the web interface at http://localhost:5000 and https://rl-scheduler.onrender.com



### Web Interface

1. **Configure Simulation**: Set parameters for the CPU environment and RL model
2. **Train RL Model**: Set number of episodes and start training
3. **Compare Results**: View comparison between RL and traditional schedulers
4. **Save Results**: Store configurations and results in the database
5. **Load Configurations**: Reload saved configurations for further testing


## Performance Metrics

The system evaluates scheduler performance using several key metrics:

- **Average Latency**: Average time from task arrival to completion
- **Throughput**: Number of tasks completed per unit time
- **CPU Utilization**: Percentage of time CPUs are busy processing tasks
- **Reward**: Weighted combination of latency reduction and throughput improvement

```mermaid
flowchart TD
    %% Main User Flow and System Components
    USER([User]) --> HOME[Home Page]
    HOME --> CONFIG[Configure Simulation]
    CONFIG --> |Set Parameters| WG[Workload Generator]
    WG --> |Generate Tasks| ENV[CPU Environment]
    CONFIG --> |Set Parameters| RL[RL Scheduler]
    CONFIG --> |Set Parameters| TRAD[Traditional Schedulers]

    %% Core System
    subgraph Core["Core System"]
        direction TB
        subgraph Simulation["Simulation Environment"]
            ENV --> |Task Queue| TASK_ASSIGN[Task Assignment]
            TASK_ASSIGN --> CPU_EXEC[CPU Execution]
            CPU_EXEC --> METRICS[Performance Metrics]
            METRICS --> |State| ENV
        end
        
        subgraph Schedulers["CPU Schedulers"]
            RL --> |Select Action| TASK_ASSIGN
            
            subgraph TRAD["Traditional Schedulers"]
                FCFS[First Come First Served]
                SJF[Shortest Job First]
                RR[Round Robin]
            end
            
            TRAD --> |Select Action| TASK_ASSIGN
            FCFS --> TRAD
            SJF --> TRAD
            RR --> TRAD
        end
        
        ENV --> |Current State| RL
        METRICS --> |Reward| RL
    end

    %% Visualization and Results Flow
    METRICS --> COMPARE[Compare Results]
    COMPARE --> VIZ[Visualization]
    VIZ --> |Charts| USER

    %% Database Flow
    COMPARE --> SAVE[Save to Database]
    SAVE --> DB[(Database)]
    DB --> MANAGE[Database Management]
    MANAGE --> |View| USER
    MANAGE --> |Load| CONFIG

    %% API Integration
    subgraph API["API Layer"]
        API_CONFIGS[API Configs]
        API_RESULTS[API Results ID]
        API_METRICS[API Compare Metrics]
    end

    DB --> API_CONFIGS
    DB --> API_RESULTS
    METRICS --> API_METRICS

    API_CONFIGS --> EXT_APP([External Applications])
    API_RESULTS --> EXT_APP
    API_METRICS --> EXT_APP

    %% Training Loop
    CONFIG --> |Start Training| TRAIN[Training Process]
    TRAIN --> |Initialize| RL
    RL --> |Update Q-Values| RL
    TRAIN --> |Episodes Done| COMPARE

    %% Component Descriptions
    classDef blue fill:#2374ab,stroke:#1e5e8a,color:#fff
    classDef green fill:#298e8a,stroke:#1f615e,color:#fff
    classDef orange fill:#ec7505,stroke:#b85a04,color:#fff
    classDef red fill:#a83941,stroke:#7c2b31,color:#fff
    classDef purple fill:#8344ad,stroke:#5f327e,color:#fff

    class HOME,CONFIG,USER,TRAIN blue
    class Core,Simulation,ENV,TASK_ASSIGN,CPU_EXEC,METRICS green
    class Schedulers,RL,TRAD,FCFS,SJF,RR orange
    class COMPARE,VIZ,SAVE,DB,MANAGE red
    class API,API_CONFIGS,API_RESULTS,API_METRICS,EXT_APP,WG purple

    %% Legend
    subgraph Legend
        L1[User Interface] --- L2[Simulation System] --- L3[Schedulers] --- L4[Database] --- L5[API Integration]
    end

    class L1 blue
    class L2 green
    class L3 orange
    class L4 red
    class L5 purple
```

# ERC-4337 Simulation README

### Overview
This project simulates the behavior of transaction bundling and inclusion fees in an ERC-4337-like environment using the cadCAD simulation framework. The goal is to understand how different parameters, such as bundle size and inclusion fees, affect the performance and earnings of transaction bundlers. This project is part of the ROP-7. For more details, refer to the [ROP-7 page](https://efdn.notion.site/RIG-Open-Problems-ROPs-c11382c213f949a4b89927ef4e962adf?p=ec5390efab864ed49a8535e8bdfff182&pm=s).


### Prerequisites
Before running the simulations, ensure you have the following libraries installed:
- numpy
- pandas
- cadCAD
- matplotlib

### Initial Setup
The simulation initializes with an empty mempool, a base fee, and specific bundle sizes. It tracks the transactions and earnings of four different bundlers, each with distinct inclusion fee requirements.

### State Variables
- `mempool`: List of unprocessed transactions.
- `base_fee`: Fixed base fee for transactions.
- `bundle_size`: Maximum number of transactions a bundler can handle.
- `bundlers`: Dictionary of bundlers with their respective inclusion fee requirements.
- `bundler_transactions`: Tracks the number of transactions each bundler accepts.
- `bundler_earnings`: Tracks the earnings of each bundler.

### Transaction Generation
Transactions are generated randomly, with inclusion fees ranging between 0.3 and 2.5. Transactions are added to the mempool if they do not already exist.

### Transaction Allocation
Transactions are allocated to bundlers based on their inclusion fees and the bundler's capacity. Allocation follows specific rules:
- Inclusion fee < 0.5: Not added to any bundler.
- 0.5 ≤ fee < 1: Added to Bundler A.
- 1 ≤ fee < 1.5: Added to Bundler A or B.
- 1.5 ≤ fee < 2: Added to Bundler A, B, or C.
- fee ≥ 2: Added to any bundler.

### Simulation Execution
The simulation runs for 100 timesteps, skipping the initial step where the mempool is empty. Each step involves generating new transactions, allocating them to bundlers, and updating the state.

### Results Visualization
The simulation results include:
- Number of transactions per bundler over time.
- Earnings of each bundler over time.
- Total allocated transactions vs. transactions remaining in the mempool.

### Example Results
Simulations demonstrate various scenarios:
- Small bundle size with a lower number of transactions.
- Larger bundle sizes with higher transaction volumes.
- Effects of varying inclusion fees on bundler performance and earnings.

### Conclusion
By running these simulations, you can gain insights into how different parameters influence transaction processing in an ERC-4337-like environment. Experimenting with different configurations can help optimize bundler strategies and improve overall efficiency.

### Link to Other Repo
For more simulations that take into account different signature types, refer to the following repository: [ERC-4337 Simulations](https://github.com/DavideRezzoli/ERC-4337_Simulations/blob/main/ERC4337_episodes.py).
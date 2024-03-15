import numpy as np
from cadCAD.configuration import Configuration
from cadCAD import configs
from cadCAD.engine import ExecutionMode, ExecutionContext, Executor
from collections import deque, Counter
import matplotlib.pyplot as plt
import pandas as pd
import random

class User:
    def __init__(self, value):
        self.value = value  # Value of the user 

class Bundler:
    def __init__(self, name, omega):
        self.name = name  # Name of the bundler
        self.omega = omega  # Marginal off-chain cost
        self.posted_price_history = []  # List to store the history of posted prices
        self.bundle = []  # List to store the accepted transactions
        self.bundleSize = 30

    def posted_price(self, M):
        # Posted price function
        return M

    def calculate_profit(self):
        if len(self.bundle) == 0:
            return 0
        else:
            total_transaction_value = sum(self.bundle)
            transaction_cost = self.omega * len(self.bundle)
            return total_transaction_value - transaction_cost - 10
    
    def __str__(self):
        # Return a string representation of the bundler
        return f'Bundler_{self.name}'

# Function to generate users
def generate_users(params, step, sL, s):
    new_transactions = []  # Initialize an empty list to store new transactions
    num_transactions = np.random.randint(40, 50)  # Generate random number of transactions between 30 to 49 (to ensure variety)
    for bundler in s['bundlers']:
        bundler.bundle = []  # Clear the bundler's bundle
    for _ in range(num_transactions):
        new_user_value = np.random.uniform(3, 12)  # Random Value (User's bid)
        new_user = User(new_user_value)
        new_transactions.append(new_user)  # Add the new transaction to the list
    return {'new_transactions': new_transactions}

def user_choose_bundler(params, step, sL, s, _input):
    new_transactions = _input.get('new_transactions')
    for user in new_transactions:
        min_price = float('inf')
        selected_bundler = None
        for bundler in s['bundlers']:
            if len(bundler.bundle) < bundler.bundleSize:  # If bundler has space
                if bundler.posted_price_history[-1] < min_price and user.value > bundler.posted_price_history[-1]:
                    min_price = bundler.posted_price_history[-1]
                    selected_bundler = bundler
        if selected_bundler is not None:
            selected_bundler.bundle.append(selected_bundler.posted_price_history[-1])  # Append the posted price of the selected bundler to the bundle
        #else:
        #    print("No bundler available for user with value:", user.value)
    return ('posted_prices', [bundler.posted_price_history[-1] for bundler in s['bundlers']])


def update_state(params, step, sL, s, _input):
    bundlers = s['bundlers']
    for bundler in bundlers:
        profit = bundler.calculate_profit()
        previous_profit = s['bundler_profits'][bundler.name]['profit']
        s['bundler_profits'][bundler.name]['profit'] += profit
        
        if len(bundler.bundle) == bundler.bundleSize:  # If bundler's bundle is full
            if profit <= previous_profit:
                bundler.posted_price_history.append(bundler.posted_price_history[-1] * 1.01)  # Increase posted price by 1%
            else:
                if profit < 0:
                    bundler.posted_price_history.append(bundler.posted_price_history[-1] * 1.005)  # Increase posted price by 0.5% for negative profit
                else:
                    bundler.posted_price_history.append(bundler.posted_price_history[-1] * 0.995)  # Decrease posted price by 0.5% for positive profit
        else:  # If bundle size is not full
            if profit <= 0:  # If profit is negative
                bundler.posted_price_history.append(bundler.posted_price_history[-1] * 0.995)  # Decrease posted price by 0.5%
            else:
                bundler.posted_price_history.append(bundler.posted_price_history[-1] * 1.005)  # Increase posted price by 0.5%
                
    return ('bundlers', bundlers)




# Initialize bundlers
bundler_names = ['a', 'b', "c"]

omegas = [random.uniform(3, 4) for _ in bundler_names] #For random omegas
# Create bundlers using initial prices
bundlers = [Bundler(name, omega) for name, omega in zip(bundler_names, omegas)]
initial_prices = [omega + 1 for omega in omegas]

# Update posted_price_history for each bundler
for bundler, price in zip(bundlers, initial_prices):
    print(bundler.omega)
    bundler.posted_price_history.append(price)

# Create initial state
initial_state = {
    'users': [],  # List of users
    'bundlers': bundlers,  # List of bundlers
    'bundler_profits': {bundler.name: { 'profit': 0} for bundler in bundlers},  # Bundler profits using bundler names as keys
    'posted_prices': initial_prices  # Initial posted prices for each bundler
}

# Simulation parameters
simulation_parameters = {
    'T': range(500),  # Time steps
    'N': 1,  # Number of Monte Carlo runs
    'M': {},  # Parameters
}

psub = [
    {
        'policies': {
            "new_transactions": generate_users,
        },
        'variables': {
            "posted_prices": user_choose_bundler, 
            "bundlers": update_state,
        },
    }
]
# Simulation configuration
config = Configuration(
    initial_state=initial_state, 
    partial_state_update_blocks=psub,
    sim_config=simulation_parameters,
    user_id=0,
    model_id=0,
    simulation_id=0,
    subset_id=0,
    subset_window=deque([0, 0]),
    state_history=True
)

# Execution context
exec_mode = ExecutionMode()
exec_context = ExecutionContext(exec_mode.single_proc)

# Create executor
executor = Executor(exec_context, [config])

# Execute simulation
raw_result, tensor, sessions = executor.execute()

# Plot posted prices for each bundler
df = pd.DataFrame(raw_result)
bundler_names = ['a', 'b', 'c']  # List of bundler names

# Initialize a dictionary to store the aggregated bundle counts for each bundler
bundle_counts_by_bundler = {bundler_name: Counter() for bundler_name in bundler_names}

# Iterate over each result in raw_result
for result in raw_result:
    # Iterate over each bundler
    for bundler_name in bundler_names:
        bundler = next((bundler for bundler in result['bundlers'] if bundler.name == bundler_name), None)
        if bundler:
            bundle_count = len(bundler.bundle)
            bundle_counts_by_bundler[bundler_name][bundle_count] += 1

# Define colors for each bundler
colors = ['blue', 'green', 'red']

# Get unique bundle sizes
bundle_sizes = sorted(set(size for bundler_counts in bundle_counts_by_bundler.values() for size in bundler_counts.keys()))

# Plot the grouped bar chart for each bundle size
bar_width = 0.1  # Width of the bars
bar_positions = np.arange(len(bundle_sizes))  # X positions for the bars

for i, bundler_name in enumerate(bundler_names):
    bundle_counts_data = [bundle_counts_by_bundler[bundler_name][size] for size in bundle_sizes]
    print(bundle_counts_data)
    plt.bar(bar_positions + i * bar_width, bundle_counts_data, bar_width, color=colors[i], alpha=0.5, label=f'Bundler {bundler_name}')

# Set labels and title for the plot
plt.xlabel('Bundle Size')
plt.ylabel('Frequency')
plt.title('Frequency of Bundle Sizes for Each Bundler')
plt.xticks(bar_positions + bar_width * (len(bundler_names) - 1) / 2, bundle_sizes)
plt.legend(title='Bundler', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()

# Plot profit for each bundler
for bundler_name in bundler_names:
    bundler_profit_data = [result['bundler_profits'][bundler_name]['profit'] for result in raw_result]
    plt.plot(bundler_profit_data, label=f'Profit - Bundler {bundler_name}')

plt.title('Profit Over Time')
plt.xlabel('Time Step')
plt.ylabel('Profit')
plt.legend()
plt.show()

# Plot posted price
for i, bundler_name in enumerate(bundler_names):
    plt.plot(df['timestep'], df['posted_prices'].apply(lambda x: x[i]), label=f'Bundler {bundler_name}')

plt.xlabel('Time Step')
plt.ylabel('Posted Price')
plt.title('Posted Prices Over Time for Each Bundler')
plt.legend()
plt.grid(True) 
plt.show()

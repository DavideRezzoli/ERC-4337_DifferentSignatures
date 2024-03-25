import numpy as np
from cadCAD.configuration import Configuration
from cadCAD import configs
from cadCAD.engine import ExecutionMode, ExecutionContext, Executor
from collections import deque, Counter
import matplotlib.pyplot as plt
import pandas as pd
import random

class User:
    def __init__(self, value,transaction_type):
        self.value = value  # Value of the user 
        self.transaction_type = transaction_type  # Type of transaction (Schnoor or BLS)

class Bundler:
    def __init__(self, name, omega,transaction_type):
        self.name = name  # Name of the bundler
        self.omega = omega  # Marginal off-chain cost
        self.transaction_type = transaction_type
        self.posted_price_history = []  # List to store the history of posted prices
        self.penalty_lambda = 0.05  # Penalty lambda for negative profits
        self.bundle = []  # List to store the accepted transactions
        self.bundleSize = 20

    def posted_price(self, M):
        # Posted price function
        return M

    def calculate_profit(self):
        if len(self.bundle) == 0:
            return 0
        else:
            total_transaction_value = sum(self.bundle)
            transaction_cost = self.omega * len(self.bundle)
            profit = total_transaction_value - transaction_cost - 20
            if profit < 0:
                penalty = abs(profit) * self.penalty_lambda
                return profit - penalty
            else:
                return profit
    def __str__(self):
        # Return a string representation of the bundler
        return f'Bundler_{self.name}'

# Function to generate users
def generate_users(params, step, sL, s):
    new_transactions = []  # Initialize an empty list to store new transactions
    num_transactions = np.random.randint(50, 60)  # Generate random number of transactions between 30 to 49 (to ensure variety)
    for bundler in s['bundlers']:
        bundler.bundle = []  # Clear the bundler's bundle
        #print("Initial posted price", bundler.posted_price_history[-1])

    for _ in range(num_transactions):
        new_user_value = np.random.uniform(3, 7)  # Random Value (User's bid)
        transaction_type = random.choice(['Schnoor', 'BLS'])  # Randomly choose transaction type
        new_user = User(new_user_value,transaction_type)
        new_transactions.append(new_user)  # Add the new transaction to the list
    return {'new_transactions': new_transactions}

def user_choose_bundler(params, step, sL, s, _input):
    new_transactions = _input.get('new_transactions')
    for user in new_transactions:
        min_price = float('inf')
        eligible_bundlers = []  # List to keep track of all eligible bundlers

        # Identify eligible bundlers based on your criteria
        for bundler in s['bundlers']:
            if len(bundler.bundle) < bundler.bundleSize:  # If bundler has space
                if (bundler.transaction_type == 'Both' or bundler.transaction_type == user.transaction_type): #Comment out if no need of signature
                    last_posted_price = bundler.posted_price_history[-1]
                    if last_posted_price < min_price and user.value > last_posted_price:
                        min_price = last_posted_price
                        eligible_bundlers = [bundler]  # Reset list with new minimum price bundler
                    elif last_posted_price == min_price:
                        eligible_bundlers.append(bundler)  # Add bundler to the list as it matches the current min price

        # Select one eligible bundler randomly if there are multiple
        selected_bundler = random.choice(eligible_bundlers) if eligible_bundlers else None

        if selected_bundler:
            # Append the posted price of the selected bundler to the bundle
            selected_bundler.bundle.append(selected_bundler.posted_price_history[-1])

    # Return the updated posted prices
    return ('posted_prices', [bundler.posted_price_history[-1] for bundler in s['bundlers']])


def update_state(params, step, sL, s, _input):
    bundlers = s['bundlers']
    
    for bundler in bundlers:
        profit = bundler.calculate_profit()
        previous_profit = s['bundler_profits'][bundler.name]['profit']
        s['bundler_profits'][bundler.name]['profit'] += profit
        
        new_price = bundler.posted_price_history[-1]  # Default to last price to start
        
        if len(bundler.bundle) == bundler.bundleSize:  # If bundler's bundle is full
            if profit > 0: #If already getting better profit than before i increase my price of a slight amount
                new_price *= 1.003
            else:  # If profit is not improved increase 0.3%
                new_price *= 1.005
        elif len(bundler.bundle) >= bundler.bundleSize / 2: # more than half of the bundle full
            if profit > 0:  # If profit is positive, Decrease a bit the price to reach more users
                new_price *= 0.999
            else: # if negative i have to increase a bit the price to increase my earning
                new_price *= 1.001
        else: # Less than 50% of the bundle is full
            if profit <= 0:
                new_price *= 0.998
            else: # have to increase my price since my posted price is too low
                new_price *= 1.002


        floor_price = bundler.omega + 1 # Minimum price i am okay accepting against my cost
        # Check against floor price before finalizing new price
        bundler.posted_price_history.append(max(new_price, floor_price))

    return ('bundlers', bundlers)





# Initialize bundlers
bundler_names = ['a', 'b', "c"]
transaction_types = ['Schnoor', 'BLS', 'Both']
omegas = [random.uniform(3, 3.5) for _ in bundler_names] #For random omegas
# Create bundlers using initial prices
bundlers = [Bundler(name, 2, transaction_type) for name, omega, transaction_type in zip(bundler_names, omegas, transaction_types)]
initial_prices = [omega + 0.5 for omega in omegas]

# Update posted_price_history for each bundler
for bundler, price in zip(bundlers, initial_prices):
    #print(bundler.omega)
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
# Initialise the subplot function using number of rows and columns 
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
bar_width = 0.1 # Width of the bars
bar_positions = np.arange(len(bundle_sizes))  # X positions for the bars
# Set up the figure and axes
fig, axs = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True)

# Plot the second subplot - Profit Over Time
axs[0].set_title('Profit Over Time')
for bundler_name in bundler_names:
    bundler_profit_data = [result['bundler_profits'][bundler_name]['profit'] for result in raw_result]
    axs[0].plot(bundler_profit_data, label=f'Profit - Bundler {bundler_name}')
axs[0].set_xlabel('Time Step')
axs[0].set_ylabel('Profit')
axs[0].legend()
axs[0].grid(True)

# Plot the third subplot - Posted Prices Over Time for Each Bundler
axs[1].set_title('Posted Prices Over Time for Each Bundler')
for i, bundler_name in enumerate(bundler_names):
    axs[1].plot(df['timestep'], df['posted_prices'].apply(lambda x: x[i]), label=f'Bundler {bundler_name}')
axs[1].set_xlabel('Time Step')
axs[1].set_ylabel('Posted Price')
axs[1].legend()
axs[1].grid(True)

# Adjust layout
#plt.tight_layout()
plt.subplots_adjust(hspace=0.3)
# Show the plot
plt.show()

for i, bundler_name in enumerate(bundler_names):
    bundle_counts_data = [bundle_counts_by_bundler[bundler_name][size] for size in bundle_sizes]
    #print(bundle_counts_data)
    plt.bar(bar_positions + i * bar_width, bundle_counts_data, bar_width, alpha=0.5, label=f'Bundler {bundler_name}')

# Set labels and title for the plot
plt.xlabel('Bundle Size')
plt.ylabel('Frequency')
plt.title('Frequency of Bundle Sizes for Each Bundler')
plt.xticks(bar_positions + bar_width * (len(bundler_names) - 1) / 2, bundle_sizes)
plt.legend(title='Bundler', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()

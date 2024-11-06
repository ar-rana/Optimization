import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Dataset 1: Resource Allocation Dataset
# Attributes: CPU_Requirement, Memory_Requirement, Storage_Requirement, Network_Bandwidth
# Objective: Cost
num_samples = 100
cpu_requirement = np.random.uniform(1, 16, num_samples)  # CPU cores
memory_requirement = np.random.uniform(1, 64, num_samples)  # Memory in GB
storage_requirement = np.random.uniform(10, 1000, num_samples)  # Storage in GB
network_bandwidth = np.random.uniform(10, 500, num_samples)  # Network bandwidth in Mbps
# Generate cost as a function of the requirements with added noise
cost = 0.1 * cpu_requirement + 0.05 * memory_requirement + 0.02 * storage_requirement + 0.01 * network_bandwidth
cost += np.random.normal(0, 0.5, num_samples)  # Adding low noise

resource_allocation_data = pd.DataFrame({
    "CPU_Requirement": cpu_requirement,
    "Memory_Requirement": memory_requirement,
    "Storage_Requirement": storage_requirement,
    "Network_Bandwidth": network_bandwidth,
    "Cost": cost
})


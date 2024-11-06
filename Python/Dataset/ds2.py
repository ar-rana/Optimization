import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Dataset 2: Load Balancing Dataset
# Attributes: Task_Size, Server_CPU_Available, Server_Memory_Available, Server_Network_Capacity
# Objective: Response_Time
task_size = np.random.uniform(1, 50, num_samples)  # Task size in processing units
server_cpu_available = np.random.uniform(2, 32, num_samples)  # CPU in cores
server_memory_available = np.random.uniform(8, 128, num_samples)  # Memory in GB
server_network_capacity = np.random.uniform(50, 1000, num_samples)  # Network bandwidth in Mbps
# Generate response time as a function of the attributes with low error
response_time = 1.5 * task_size / (server_cpu_available + 0.1) + 0.2 * task_size / (server_memory_available + 0.1)
response_time += 0.1 * task_size / (server_network_capacity + 1)
response_time += np.random.normal(0, 0.2, num_samples)  # Adding low noise

load_balancing_data = pd.DataFrame({
    "Task_Size": task_size,
    "Server_CPU_Available": server_cpu_available,
    "Server_Memory_Available": server_memory_available,
    "Server_Network_Capacity": server_network_capacity,
    "Response_Time": response_time
})

# Save the datasets to CSV files for easy download
resource_allocation_data_path = "/mnt/data/resource_allocation_data.csv"
load_balancing_data_path = "/mnt/data/load_balancing_data.csv"

resource_allocation_data.to_csv(resource_allocation_data_path, index=False)
load_balancing_data.to_csv(load_balancing_data_path, index=False)

resource_allocation_data_path, load_balancing_data_path

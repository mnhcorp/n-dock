# Configuration file for n_dock project

config = {
    'data_path': '/path/to/your/data',  # Replace with the actual path to your data
    # Add other configuration parameters as needed
}

def get_config():
    return config

def update_config(key, value):
    global config
    config[key] = value

# Example usage:
# from n_dock.config import get_config, update_config
#
# # Get the entire config
# cfg = get_config()
#
# # Update a specific config value
# update_config('data_path', '/new/path/to/data')

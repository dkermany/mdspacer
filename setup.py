import os

def find_root_path(current_path, marker='.git'):
    while not os.path.exists(os.path.join(current_path, marker)):
        current_path, _ = os.path.split(current_path)
        if current_path == '':
            raise FileNotFoundError("Could not find the marker")
    return current_path

def read_env_file(file_path):
    """Read a .env file and return its contents as a dictionary."""
    env_vars = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#') or not line.strip():
                continue
            key, value = line.strip().split('=', 1)
            env_vars[key] = value
    return env_vars

def write_env(file_path, env_vars):
    """Write a dictionary of environment variables to a .env file."""
    with open(file_path, 'w') as file:
        for key, value in env_vars.items():
            file.write(f'{key}={value}\n')

if __name__ == "__main__":
    root_path = find_root_path(__file__)

    env_path = os.path.join(root_path, ".env")
    env_vars = read_env_file(env_path)
    env_vars["ROOTPATH"] = root_path
    env_vars["BONEPATH"] = "/data/dkermany_data/Bone_Project/"

    # Write the changes back to the .env file
    write_env(env_path, env_vars)


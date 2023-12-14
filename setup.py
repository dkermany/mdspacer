import os

def find_root_path(current_path, marker='.git'):
    while not os.path.exists(os.path.join(current_path, marker)):
        current_path, _ = os.path.split(current_path)
        if current_path == '':
            raise FileNotFoundError("Could not find the marker")
    return current_path


if __name__ == "__main__":
    root_path = find_root_path(__file__)
    os.environ["ROOTPATH"] = root_path
    os.environ["BONEPATH"] = "/data/dkermany_data/Bone_Project/"



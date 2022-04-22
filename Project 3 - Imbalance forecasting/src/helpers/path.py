import pathlib

def get_project_root_path(look_for_string: str = "Project 3") -> pathlib.Path:
    path = pathlib.Path.cwd()
    if path.name.startswith(look_for_string):
        return path
    
    for parent in path.parents:
        if parent.name.startswith(look_for_string):
            return parent
    
    raise ValueError(f"Couldn't find string {look_for_string} in any parent of {path}")


ROOT_PATH = get_project_root_path("Project 3")
DATA_PATH = ROOT_PATH / "data"

if __name__ == "__main__":
    print(ROOT_PATH)
    print(DATA_PATH)

import yaml

def write_yaml(data: dict):
    path = data['path']
    file = open(path + "/" + "config.yml", "w")
    yaml.dump(data, file)
    file.close()
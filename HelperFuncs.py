import os


def get_all_file_extensions(directory):
    extensions = set()
    for root, dirs, files in os.walk(directory):
        for file in files:
            _, extension = os.path.splitext(file)
            extensions.add(extension)
    return extensions


if __name__ == "__main__":
    directory = "slama"
    extensions = get_all_file_extensions(directory)
    print(extensions)

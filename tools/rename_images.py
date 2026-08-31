
import os
import sys

def rename_files_in_directory(directory, old_str, new_str):
    for filename in os.listdir(directory):
        old_path = os.path.join(directory, filename)
        new_filename = filename.replace(old_str, new_str)
        new_path = os.path.join(directory, new_filename)

        # Přeskočíme položky, které nejsou soubory
        if not os.path.isfile(old_path):
            continue

        # Přejmenování
        os.rename(old_path, new_path)
        print(f"{filename} -> {new_filename}")

assert len(sys.argv) == 3, sys.argv
directory = sys.argv[1]
new_str = sys.argv[2]

rename_files_in_directory(directory, "im", new_str)

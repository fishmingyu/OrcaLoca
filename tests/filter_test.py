import os

for _, dirs, _ in os.walk("./output"):
    for dir in dirs:
        # walk through the directory
        count = 0
        for _, _, files in os.walk(f"./output/{dir}"):
            for file in files:
                # walk through the files
                # check if a file starts with "searcher_" and ends with ".json"
                if file.startswith("searcher_") and file.endswith(".json"):
                    count += 1
        if count == 0:
            print(f"{dir} json not found")

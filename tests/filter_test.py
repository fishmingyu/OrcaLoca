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


for _, dirs, _ in os.walk("./log"):
    for dir in dirs:
        # walk through the directory
        for _, _, files in os.walk(f"./log/{dir}"):
            for file in files:
                # walk through the files
                # check if a file named Orcar.search_agent.log
                if file == "Orcar.search_agent.log":
                    with open(f"./log/{dir}/{file}") as f:
                        # check if the file has the string "OrcarAgent: Done"
                        if "Current search queue size: 0" in f.read():
                            break
                        else:
                            print(f"early stop in {dir}")

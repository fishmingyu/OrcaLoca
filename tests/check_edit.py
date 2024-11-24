import os


def check_output():
    # target directory ./output/
    # iterate over all subdirectories

    problem_list = []

    for _, dirs, files in os.walk("./output/"):
        for dir in dirs:
            # check if .patch file exists, we don't now the name of the file
            # so we iterate over all files in the directory
            for _, _, files in os.walk(f"./output/{dir}"):
                if_patch_exists = False
                for file in files:
                    # check if file is a .patch file
                    if file.endswith(".patch"):
                        if_patch_exists = True
                        break
                if not if_patch_exists:
                    problem_list.append(dir)

    if len(problem_list) == 0:
        print("All good!")
    else:
        print(f"Problems with the following directories: total {len(problem_list)}")
        # sort the list
        problem_list.sort()
        for problem in problem_list:
            print(problem)


if __name__ == "__main__":
    check_output()

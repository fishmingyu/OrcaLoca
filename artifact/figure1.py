import csv

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# from matplotlib.collections import LineCollection

# import seaborn as sns

# style have whitegrid, darkgrid, dark, white, ticks
# sns.set(style="white")

percent_resolved = []
percent_function = []
percent_file = []

with open("lite_results_paper.csv", "r", newline="") as f:
    reader = csv.reader(f)
    next(
        reader
    )  # If there is a header row, skip it. Remove if you do NOT have headers.

    for row in reader:
        # row might look like:
        #   [ "Blackbox", "49", "147", "63.333333", "190", "81.333333", "244" ]
        # So, for example:
        #   row[1] -> "% Resolve"
        #   row[3] -> "% Function"
        #   row[5] -> "% File"

        percent_resolved.append(float(row[1]))
        percent_function.append(float(row[3]))
        percent_file.append(float(row[5]))

print("Resolved %:", percent_resolved)
print("Function %:", percent_function)
print("File %:", percent_file)


plt.rcParams["font.family"] = "Times New Roman"

colors = ["#936bf8", "#f96363", "#fdc55b"]

all_data = [percent_resolved, percent_function, percent_file]
labels = [r"% Resolved", r"% Function", r"% File"]


fig, ax = plt.subplots(figsize=(7, 3))

positions = [1, 2, 3]

violin_parts = ax.violinplot(all_data, positions=positions, widths=0.8, showmeans=True)


for i, body in enumerate(violin_parts["bodies"]):
    path = body.get_paths()[0]
    vertices = path.vertices
    body.set_hatch("///")
    center_x = positions[i]
    new_vertices = []

    for x, y in vertices:
        if x > center_x:
            x = center_x
        new_vertices.append([x, y])

    path.vertices = np.array(new_vertices)

    body.set_edgecolor(colors[i])
    body.set_facecolor("none")
    # body.set_edgecolor('black')
    body.set_alpha(0.7)


for item in ["cmins", "cmaxes", "cbars"]:
    lines = violin_parts[item]
    lines.set_color("darkgrey")
    lines.set_linestyle("--")
    lines.set_linewidth(1)

    segments = lines.get_segments()
    for i, seg in enumerate(segments):
        new_seg = []
        for xx, yy in seg:
            if xx > positions[i]:
                xx = positions[i]
            new_seg.append([xx, yy])
        segments[i] = new_seg
    lines.set_segments(segments)


cmeans = violin_parts["cmeans"]
cmeans.set_color("black")
cmeans.set_linewidth(2)
cmeans.set_linestyle("-")
cmeans.set_alpha(1.0)
cmeans.set_zorder(3)


for i, d in enumerate(all_data):
    x_text = positions[i] + 0.1
    mean_val = np.mean(d)
    std_val = np.std(d)

    ax.text(
        x_text,
        mean_val + 4,
        f"{mean_val:.2f}%",
        ha="left",
        va="center",
        fontsize=17,
    )


ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=15)
ax.set_ylabel("Percentage", fontsize=15)
plt.yticks(fontsize=13)

# from matplotlib.patches import Patch

legend_patch = [
    mpatches.Patch(facecolor="none", edgecolor=color, hatch="///", label=label)
    for color, label in zip(colors, labels)
]
ax.legend(handles=legend_patch, fontsize=14)

# plt.title("Distribution and Average of File Match Rate ", fontsize=20)
plt.savefig("motivation.pdf", format="pdf", bbox_inches="tight")
# plt.tight_layout()
# plt.show()

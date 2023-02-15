import lib
from collections import OrderedDict
from convergence_speed_table import get_k_trehsold, get_stats_for_run, columns, render_columns, get_history_average
import matplotlib.pyplot as plt

data = OrderedDict()

data["N=0, run 1"] = lib.get_runs(["vq_vae_baseline_magic_bad"])
data["N=0, run 2"] = lib.get_runs(["vq_vae_baseline_magic"])

r = lib.get_runs(["vq_vae_original_with_1"])
r = [a for a in r if a.config["vq_vae.neihborhood"]=="none" and a.config["vq_vae.grid_dim"] == 1]
data["N=1"] = r


THRESHOLD = 1.1
tresh = get_k_trehsold(data["N=0, run 2"], THRESHOLD)

res = {}
for n, r in data.items():
    res[n] = get_stats_for_run(r, tresh, THRESHOLD)


print("\\begin{tabular}{l " + " ".join(["c"] * (len(columns))) + "} ")
print("\\toprule")
print("Model & " + " & ".join(columns) + " \\\\")
print("\\midrule")
for t in data.keys():
    print(t  + render_columns(res[t]) + " \\\\")

print("\\bottomrule")
print("\\end{tabular}")


fig = plt.figure(figsize=[4,2])
plots = []
for n, r in data.items():
    _, steps, hist = get_history_average(r, "train/loss")
    steps = [s for s in steps if s<=1500]
    hist = hist[:len(steps)]
    plots.append(plt.plot(steps, [p.mean for p in hist]))
    plt.fill_between(steps, [p.mean - p.std for p in hist], [p.mean + p.std for p in hist], alpha=0.3, linewidth=0)

plt.legend([p[0] for p in plots], data.keys())
plt.ylim(0,1.5)
plt.xlim(0,1500)
plt.xlabel("Iterations")
plt.ylabel("Loss")
fig.savefig("covergence.pdf", bbox_inches='tight', pad_inches = 0.01)
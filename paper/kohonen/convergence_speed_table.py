import lib
from lib.common import group, calc_stat
from lib.stat_tracker import Stat, StatTracker
from collections import OrderedDict
import math

# print(len(orig_runs))

THRESHOLD=1.1

def convergence_speed_of_run_and_best_loss(run, treshold):
    h = run.history(keys=["validation/mean_loss", "iteration"], pandas=False)
    steps = [p["iteration"] for p in h]
    losses = [p["validation/mean_loss"] for p in h]

    best_loss = min(losses)
    target = best_loss * treshold

    order = sorted(range(len(steps)), key=steps.__getitem__)
    for i in order:
        if losses[i] <= target:
            return steps[i], best_loss
    assert False


def convergence_speed_and_best_loss(runs, treshold):
    t = lib.StatTracker()
    l = lib.StatTracker()
    for r in runs:
        steps, loss = convergence_speed_of_run_and_best_loss(r, treshold)
        t.add(steps)
        l.add(loss)

    return t.get(), l.get()


def get_ordered_history(run, keys=[]):
    h = run.history(keys=["iteration"]+keys, pandas=False, samples=1000000000)
    steps = [p["iteration"] for p in h]
    order = sorted(range(len(steps)), key=steps.__getitem__)
    return [h[o] for o in order]


def get_history_average(runs, key="validation/mean_loss"):
    h = get_ordered_history(runs[0], [key])
    steps = [p["iteration"] for p in h]

    max_step = steps[-1]
    sums = {k: StatTracker() for k in steps}
    for r in runs:
        h = get_ordered_history(r, [key])
        for i, p in enumerate(h):
            if i>= len(steps):
                max_step = min(max_step, steps[-1])
                break
            assert p["iteration"] == steps[i]

            sums[p["iteration"]].add(p[key])
        max_step = min(max_step, h[-1]["iteration"])

    res = [sums[s].get() for s in steps if s <= max_step]

    return [r.mean for r in res], steps, res


def get_intersection(runs, tresh):
    avg, steps, _ = get_history_average(runs)
    for a, s in zip(avg, steps):
        if a <= tresh:
            return s
    return float("inf")

def get_stats_for_run(r, k_trehsold, relative_threshold):
    s, l = convergence_speed_and_best_loss(r, relative_threshold)
    return {
        "loss": l * 1000,
        "conv. speed": s / 1000,
        "K": get_intersection(r, k_trehsold) / 1000
    }


def get_k_trehsold(baseline, relative_threshold):
    avg, _, _ = get_history_average(baseline)
    return min(avg) * relative_threshold


def is_baseline(run):
    c = run.config
    return (c["vq_vae.quantizer"] == "vq_vae_original" and c["vq_vae.neihborhood"] == "none" and
            c["vq_vae.num_embeddings"] == 512 and c.get("vq_vae.magic_counter_init", 0) == 1)


def find_baseline(groups):
    res = None
    for n, runs in groups.items():
        if is_baseline(runs[0]):
            assert res is None, "Multiple baselines found"
            res = n
    assert res is not None, "Baseline not found"
    return res


def filter_2d_or_original(r):
    is_original = r.config["vq_vae.quantizer"] == "hard_som" and r.config["vq_vae.neihborhood"] == "none"
    return (r.config["vq_vae.grid_dim"] == 1) == is_original

def get_config(run, name):
    # Defaults for old runs
    if name in run.config:
        return run.config[name]

    if name == "vq_vae.magic_counter_init":
        return 0

    assert False, f"No default specification for {name}, but run {run.id} is missing it"

def get_stats(runs, relative_threshold, group_list, filter_fn=filter_2d_or_original, allow_sizes={}, rename_baseline=True):
    print(f"N runs before filter {len(runs)}")
    runs = list(filter(filter_fn, runs))
    print(f"N runs after filter {len(runs)}")
    groups = group(runs, group_list, get_config)
    print(list(groups.keys()))

    v0 = list(groups.values())[0]
    for v in groups.values():
        if len(v) != len(v0) and len(v) not in allow_sizes:
            print("Inconsistent group sizes.")
            for k, v in groups.items():
                print(f"  {k}: {len(v)}")
            assert False

    for n, rlist in groups.items():
        rzero = None
        for r in rlist:
            if r.summary["iteration"] not in {29980, 30000}:
                print(f"Invalid iteration count for config {n}")
                assert False

            if rzero is None:
                rzero = r
            else:
                for k, v in r.config.items():
                    other = rzero.config.get(k)
                    if k not in {"sweep_id_for_grid_search", "job_id", "test_batch_size"} and (k != "vq_vae.count_unit" or r.config["vq_vae.neihborhood"] != "none") and other is not None and other != v:
                        print(f"Configuration mismatch within group '{n}' on parameter '{k}'")
                        assert False



    baseline_name = find_baseline(groups)

    base_vq = groups[baseline_name]
    if rename_baseline:
        del groups[baseline_name]

        groups = groups.copy()
        groups["baseline"] = base_vq

    tresh = get_k_trehsold(base_vq, relative_threshold)

    res = {}
    for n, r in groups.items():
        res[n] = get_stats_for_run(r, tresh, relative_threshold)
    return res


columns = ["loss", "conv. speed", "K"]

def printnum(num):
    if isinstance(num, int) or (math.isfinite(num) and abs(round(num) - num) < 1e-6):
        return str(int(num))
    else:
        return f"{num:.2f}"

def render_val(val):
    if isinstance(val, Stat):
        return f"${printnum(val.mean)} \\pm {printnum(val.std)}$ "
    else:
        return f"${printnum(val)}$ "

def render_val_f(val):
    if isinstance(val, Stat):
        return f"${val.mean:.1f} \\pm {val.std:.1f}$ "
    else:
        return f"${val:.1f}$ "

def render_columns(r):
    line = ""
    for c in columns:
        line += f"& {render_val(r[c])} "
    return line

name_map = {
    "vq_vae.quantizer_hard_som": "orig",
    "vq_vae.neihborhood_gaussian": "gauss",
    "vq_vae.neihborhood_hard": "hard",
    "vq_vae.neihborhood_none": "none",
    "vq_vae.grid_dim_1": "1D",
    "vq_vae.grid_dim_2": "2D",
    "vq_vae.num_embeddings_1024": "1024",
    "vq_vae.num_embeddings_2048": "2048",
    "vq_vae.num_embeddings_512": "512",
    "vq_vae.quantizer_hardsom_noupdate_zero": "no-update",
    "vq_vae.magic_counter_init_0": "N=0",
    "vq_vae.magic_counter_init_1": "N=1",
    "vq_vae.count_unit_1": "$\\Delta$=1",
    "vq_vae.count_unit_0.1": "$\\Delta$=0.1",
    "vq_vae.count_unit_0.01": "$\\Delta$=0.01",
}


def print_big_table(datasets):
    all_models = set()
    for stats in datasets.values():
        for k in stats.keys():
            all_models.add(k)

    all_models = list(sorted(all_models))

    print("\\begin{tabular}{l " + " ".join(["c"] * (len(columns) * (len(datasets)))) + "} ")
    print("\\toprule")
    print("&" + "& ".join([" \\multicolumn{"+str(len(columns))+"}{c}{"+n+"}" for n in datasets.keys()]) + "\\\\")
    print("".join(["\\cmidrule(lr){"+str(2+i*3)+"-"+str(1+(i+1)*3)+"}" for i in range(len(datasets))]))
    print("Model & " + " & ".join([" & ".join(columns) for _ in datasets]) + " \\\\")
    print("\\midrule")

    def remap_name(name: str) -> str:
        pieces = name.split("/")
        return "/".join(name_map.get(p, p) for p in pieces)

    for m in all_models:
        line = remap_name(m).replace("_", "\\_")
        for d in datasets.values():
            if m not in d:
                line += "".join([" & " for _ in columns])
                continue

            line += render_columns(d[m])

        print(f"{line} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


class Formatter:
    def get_headers(self):
        raise NotImplementedError()

    def get_model_name(self, config):
        raise NotImplementedError()

    def is_delimiter(self, old_name, new_name):
        raise NotImplementedError()

    def get_key_priorities(self):
        raise NotImplementedError()

    def get_loss_scale(self, ds_name):
        raise NotImplementedError()

    def get_config(self, config, key):
        return config[key]


def print_big_table_format2(datasets, formatter: Formatter):
    all_models = set()

    key_priorities = formatter.get_key_priorities()

    all_models = {}

    raw_groups= {k: group(v, key_priorities.keys(), get_config) for k, v in datasets.items()}
    for g, stats in raw_groups.items():
        for name, runs in stats.items():
            if name not in all_models:
                all_models[name] = runs[0].config

    def get_key(e):
        _, config = e
        return tuple(v(formatter.get_config(config, k)) for k, v in key_priorities.items())

    all_models = [(k, v) for k, v in all_models.items()]
    all_models = list(sorted(all_models, key=get_key))
    columns = ["Loss", "+10\\%", "+20\\%"]


    groups_10 = {k: get_stats(v, 1.1, key_priorities.keys(), filter_fn=lambda x: True, rename_baseline=False) for k, v in datasets.items()}
    groups_20 = {k: get_stats(v, 1.2, key_priorities.keys(), filter_fn=lambda x: True, rename_baseline=False) for k, v in datasets.items()}


    header = formatter.get_headers()

    print("\\begin{tabular}{l l " + " ".join(["c"] * (len(columns) * (len(datasets)))) + "} ")
    print("\\toprule")
    print(" ".join("& " for _ in header) + "& ".join([" \\multicolumn{"+str(len(columns))+"}{c}{"+n+"}" for n in datasets.keys()]) + "\\\\")
    print("".join(["\\cmidrule(lr){"+str(len(header)+1+i*3)+"-"+str(len(header)+(i+1)*3)+"}" for i in range(len(datasets))]))
    print(" & ".join(header) + " & " + " & ".join([" & ".join(columns) for _ in datasets]) + " \\\\")
    print("\\midrule")
    old_name = None
    for m in all_models:
        # line = remap_name(m).replace("_", "\\_")
        model_name = formatter.get_model_name(m[1])
        if formatter.is_delimiter(old_name, model_name):
            print("\\midrule")
        old_name = model_name

        line = model_name+""
        for d in datasets.keys():
            if m[0] not in groups_10[d]:
                line += "".join([" & " for _ in columns])
                continue

            line += " & ".join(["", render_val_f(groups_10[d][m[0]]["loss"] * formatter.get_loss_scale(d)), render_val_f(groups_10[d][m[0]]["conv. speed"]), render_val_f(groups_20[d][m[0]]["conv. speed"])])
        print(f"{line} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    group_list = ["vq_vae.quantizer", "vq_vae.neihborhood", "vq_vae.num_embeddings"]

    datasets = OrderedDict()


    runs = lib.get_runs(["vq_vae_original_with_1", "vq_vae_all_som"])
    datasets["cifar10"] = get_stats(runs, THRESHOLD, group_list)


    runs = lib.get_runs(["vqvae2_face_mixture", "vq_vae2_face_mixture_orig_1_more_seeds", "vq_vae2_face_mixture_orig_som_more_seeds", "vq_vae2_face_mixture_more_seeds_2"])
    datasets["face mixtures"] = get_stats(runs, THRESHOLD, group_list)


    runs= lib.get_runs(["vq_vae2_mine", "vq_vae2_orig_1", "vq_vae2_mine_more_seeds", "vq_vae2_orig_1_more_seeds", "vq_vae2_imagenet_original_som", "vq_vae2_imagenet_original_som_one_more_seed"])
    datasets["imagenet"] = get_stats(runs, THRESHOLD, group_list, allow_sizes={5,2})

    print_big_table(datasets)
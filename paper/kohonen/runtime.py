import lib

runs = [r for r in lib.get_runs(["vq_vae_runtime"])]

groups = lib.common.group(runs, ["vq_vae.neihborhood"])

mean_runtime = {k: sum([r.summary["_runtime"] for r in v])/len(v) for k, v in groups.items()}
print(mean_runtime["vq_vae.neihborhood_hard"] / mean_runtime["vq_vae.neihborhood_none"])

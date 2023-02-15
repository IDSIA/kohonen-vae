import lib
import functools
import matplotlib.pyplot as plt
from convergence_speed_table import get_history_average
from collections import OrderedDict

def filter_baseline(r):
    c = r.config
    return (c["vq_vae.quantizer"] == "hard_som" and c["vq_vae.neihborhood"] == "none" and
            c["vq_vae.num_embeddings"] == 512 and c.get("vq_vae.magic_counter_init", 0) == 1 and
            c["vq_vae.grid_dim"] == 1)

def filter_neigborhood(r, neighborhood: str):
    c = r.config
    return (c["vq_vae.quantizer"] == "hard_som" and c["vq_vae.neihborhood"] == neighborhood and
            c["vq_vae.num_embeddings"] == 512 and c.get("vq_vae.magic_counter_init", 0) == 1 and
            c["vq_vae.grid_dim"] == 2)


def get_filtered_hist_average(r, n):
    # Bug workaround: I accidentally averaged all the validation points into the perplexity, which makes every point
    # after 1000s off, causing spikes (testing is every 1000 iterations). Remove those
    _, steps, hist = get_history_average(r, n)

    sout, hout = [], []
    skip_next = False
    for s, h in zip(steps, hist):
        if s % 1000 != 0:
            sout.append(s)
            hout.append(h)
    return sout, hout

perplexity_name_map = {
    "/perplexity": "perplexity",
    "/perplexity1": "perplexity_top",
    "/perplexity2": "perplexity_bottom",
}

def plot(runs, prefix):
    plots = OrderedDict()
    plots["Baseline"] = list(filter(filter_baseline, runs))
    plots["Gaussian"] = list(filter(functools.partial(filter_neigborhood, neighborhood="gaussian"), runs))
    plots["Hard"] = list(filter(functools.partial(filter_neigborhood, neighborhood="hard"), runs))

    assert len(plots["Baseline"]) in {10, 5}
    for v in plots.values():
        assert len(v) == len(plots["Baseline"])

    r = plots["Baseline"][0]
    perplexity_names = [k for k in r.summary.keys() if "perplexity" in k]

    for n in perplexity_names:
        plist = []
        fig = plt.figure(figsize=[4,2])
        for r in plots.values():
            steps, hist = get_filtered_hist_average(r, n)
            plist.append(plt.plot(steps, [p.mean for p in hist]))
            plt.fill_between(steps, [p.mean - p.std for p in hist], [p.mean + p.std for p in hist], alpha=0.3, linewidth=0)


        plt.legend([p[0] for p in plist], plots.keys())
        plt.xlabel("Iterations")
        plt.ylabel("Perplexity")
        plt.xlim((0, 30000))
        fig.savefig(f"{prefix}_{perplexity_name_map[n]}.pdf", bbox_inches='tight', pad_inches = 0.01)


runs = lib.get_runs(["vq_vae_original_with_1"])
plot(runs, "cifar")

runs = lib.get_runs(["vq_vae2_orig_1_more_seeds", "vq_vae2_orig_1", "vq_vae2_imagenet_original_som", "vq_vae2_imagenet_original_som_one_more_seed"])
plot(runs, "imagenet")

runs = lib.get_runs(["vqvae2_face_mixture", "vq_vae2_face_mixture_orig_1_more_seeds", "vq_vae2_face_mixture_orig_som_more_seeds", "vq_vae2_face_mixture_more_seeds_2"])
plot(runs, "face_mixture")
import lib
from collections import OrderedDict
from convergence_speed_table import filter_2d_or_original, print_big_table_format2, Formatter



datasets = OrderedDict()


runs = lib.get_runs(["vq_vae_shirnk_speed_n1"])
runs = list(filter(filter_2d_or_original, runs))

runs_1d = lib.get_runs(["vq_vae_original_with_1"])
runs_1d = list(filter(lambda r: r.config["vq_vae.grid_dim"] == 1, runs_1d))
runs = runs + runs_1d

datasets["CIFAR-10"] = runs

runs = lib.get_runs(["vq_vae2_face_mixture_count_unit"])
runs = list(filter(filter_2d_or_original, runs))

runs_1d = lib.get_runs(["vqvae2_face_mixture", "vq_vae2_face_mixture_orig_1_more_seeds", "vq_vae2_face_mixture_more_seeds_1d", "vq_vae2_face_mixture_more_seeds_1d_2"])
runs_1d = list(filter(lambda r: r.config["vq_vae.grid_dim"] == 1 and r.config["vq_vae.quantizer"] == "hard_som", runs_1d))
runs = runs + runs_1d

datasets["CelebA-HQ/AFHQ"] = runs


class AblationTableFormatter(Formatter):
    def get_headers(self):
        return ["Neighbours", "$\\tau$"]

    def get_model_name(self, config):
        name = config["vq_vae.neihborhood"].capitalize()
        if config["vq_vae.neihborhood"] != "none":
            name += f" {config['vq_vae.grid_dim']}D"
        return f"{name} & {config['vq_vae.count_unit']}"

    def is_delimiter(self, old_name, new_name):
        if old_name is None:
            return False

        return old_name.split("&")[0] != new_name.split("&")[0]

    def get_key_priorities(self):
        key_priority = OrderedDict()
        key_priority["vq_vae.neihborhood"] = lambda x: ["none", "hard", "gaussian"].index(x)
        key_priority["vq_vae.grid_dim"] = lambda x: x
        key_priority["vq_vae.count_unit"] = lambda x: -x

        # ["vq_vae.neihborhood", "vq_vae.count_unit", "vq_vae.grid_dim"]
        return key_priority

    def get_loss_scale(self, ds_name):
        return 10 if ds_name=="CelebA-HQ/AFHQ" else 1


print_big_table_format2(datasets, AblationTableFormatter())

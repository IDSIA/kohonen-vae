import lib
from collections import OrderedDict
from convergence_speed_table import Formatter, print_big_table_format2, is_baseline


class FixedZeroTableFormatter(Formatter):
    def get_headers(self):
        return ["Variant", "Neigh.", "N"]

    def get_model_name(self, config):
        q_name_map = {
            "hard_som": "orig",
            "vq_vae_original": "orig",
            "hardsom_noupdate_zero": "no-update",
            "gd_som": "GD"
        }
        name = q_name_map[config["vq_vae.quantizer"]]
        neighborhood = config["vq_vae.neihborhood"].capitalize()
        return f"{name} & {neighborhood} & {config.get('vq_vae.magic_counter_init', 0)}"

    def is_delimiter(self, old_name, new_name):
        if old_name is None:
            return False

        return old_name.split("&")[0] != new_name.split("&")[0]

    def get_key_priorities(self):
        key_priority = OrderedDict()
        key_priority["vq_vae.quantizer"] = lambda x: ["gd_som", "hard_som", "vq_vae_original"].index(x)
        key_priority["vq_vae.neihborhood"] = lambda x: ["none", "hard", "gaussian"].index(x)
        key_priority["vq_vae.magic_counter_init"] = lambda x: x
        return key_priority

    def get_loss_scale(self, ds_name):
        return 10 if ds_name=="CelebA-HQ/AFHQ" else 1

    def get_config(self, config, key):
        if (key not in config) and key == "vq_vae.magic_counter_init":
            return 0

        return config[key]


datasets = OrderedDict()

runs = [r for r in lib.get_runs(["vq_vae_all_gd"])]
runs += [r for r in lib.get_runs(["vq_vae_original_with_1"]) if is_baseline(r) and r.config["vq_vae.grid_dim"]==1]


datasets["CIFAR-10"] = runs

print_big_table_format2(datasets, FixedZeroTableFormatter())
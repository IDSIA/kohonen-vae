import sys
sys.path.append('../..')

from typing import Tuple
from models.vq_vae import VQVAE, SOMGeometry, HardNeighborhood, Grid, EmptyNeigborhood, GaussianNeighborhood, HardSOM
from models.vq_vae2 import VQVAE2, Quantize
import numpy as np

import framework
import torch

import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from tqdm import tqdm

import os
import matplotlib.pyplot as plt

os.makedirs("image_out", exist_ok=True)

# vq_vae2_slow_decay.pth
# checkpoint = torch.load("vq_vae2_slow_decay.pth")

# this is the good looking codebook vqvae2
# checkpoint = torch.load("model-30000.pth")
# checkpoint = torch.load("model_vqvae1.pth")
# checkpoint = torch.load("vqvae1_codebook_32_2d.pth")
# ./vqvae1_codebook_32.pth
# checkpoint = torch.load("model_vqvae2_orig.pth")
# checkpoint = torch.load("vqvae1_original_32.pth")
# checkpoint = torch.load("vq_vae_gaussian_0.01.pth")
# checkpoint = torch.load("vq_vae_hard_0.01.pth")
# checkpoint = torch.load("vq_vae_my_hard.pth")
# checkpoint = torch.load("vq_vae_my_gaussian.pth")
# checkpoint = torch.load("vq_vae_hard_delta=1.pth")
# checkpoint = torch.load("vq_vae_hard_delta=0.1.pth")
# checkpoint = torch.load("vq_vae_gaussian_delta=0.1.pth")
# checkpoint = torch.load("vq_vae_gaussian_delta=1.pth")
# checkpoint = torch.load("vq_vae_gaussian_delta=0.01_2.pth")
# checkpoint = torch.load("vq_vae_hard_delta=0.01_2.pth")
# checkpoint = torch.load("vq_vae_orig_none.pth")

# This is the hard used for macaron and the dog
checkpoint = torch.load("vq_vae2_hard_imagenet.pth")
# checkpoint = torch.load("vq_vae2_none_imagenet.pth")

# checkpoint = torch.load("vq_vae2_gaussian_imagenet3.pth")

args = checkpoint['run_invariants']['args']
args = framework.data_structures.dotdict.create_recursive_dot_dict(args)


def create_geometry(args) -> SOMGeometry:
    dim = args.vq_vae.grid_dim
    if args.vq_vae.neihborhood == "hard":
        neighborhood = HardNeighborhood(args.vq_vae.count_unit)
    elif args.vq_vae.neihborhood == "none":
        neighborhood = EmptyNeigborhood()
        dim = 1
        # dim = 2
        # neighborhood = HardNeighborhood(args.vq_vae.count_unit)
    elif args.vq_vae.neihborhood == "gaussian":
        neighborhood = GaussianNeighborhood(args.vq_vae.count_unit,
                                            base=args.vq_vae.gaussaian_base)
    else:
        raise ValueError(f"Invalid neighborhood: {args.vq_vae.neihborhood}")

    return SOMGeometry(
        Grid(dim),
        neighborhood
    )


def create_model(args, geometry) -> torch.nn.Module:
    if args.task in {"vq_vae2", "vq_vae2_face_mixture"}:
        return VQVAE2(
            n_res_channel=args.vq_vae.num_residual_hiddens,
            channel=args.vq_vae.num_hiddens,
            n_res_block=args.vq_vae.num_residual_layers,
            n_embed=args.vq_vae.num_embeddings,
            embed_dim=args.vq_vae.embedding_dim,
            decay=args.vq_vae.decay,
            commitment_cost=args.vq_vae.commitment_cost,
            quantizer=args.vq_vae.quantizer,
            som_geometry=geometry,
            counter_init=args.vq_vae.get("magic_counter_init", 0.0)
        )
    elif args.task in {"vq_vae", "vq_vae_face_mixture"}:
        return VQVAE(
            args.vq_vae.num_hiddens, args.vq_vae.num_residual_layers,
            args.vq_vae.num_residual_hiddens, args.vq_vae.num_embeddings,
            args.vq_vae.embedding_dim, args.vq_vae.commitment_cost,
            decay=args.vq_vae.decay,
            quantizer=args.vq_vae.quantizer,
            som_geometry=geometry, magic_counter_init=args.vq_vae.get("magic_counter_init", 0.0))
    else:
        raise ValueError(f"Unsupported task: {args.task}")


def get_mean_std(args) -> Tuple[np.ndarray, np.ndarray]:
    if args.task == "vq_vae2":
        return (
            np.asfarray([[[0.485]], [[0.456]], [[0.406]]], dtype=np.float32) * 255.0,
            np.asfarray([[[0.229]], [[0.224]], [[0.225]]], dtype=np.float32) * 255.0
        )
    elif args.task == "vq_vae":
        return (
            np.asfarray([[[125.3]], [[123.0]], [[113.9]]], dtype=np.float32),
            np.asfarray([[[63.0]], [[62.1]], [[66.7]]], dtype=np.float32)
        )
    elif args.task in {"vq_vae_face_mixture", "vq_vae2_face_mixture"}:
        return (
            np.asfarray([[[127.5]], [[127.5]], [[127.5]]], dtype=np.float32),
            np.asfarray([[[127.5]], [[127.5]], [[127.5]]], dtype=np.float32)
        )
    else:
        raise ValueError(f"Unsupported task {args.task}")


def get_img_size(args):
    if args.task in {"vq_vae2", "vq_vae2_face_mixture", "vq_vae_face_mixture"}:
        return (256, 256)
    elif args.task == "vq_vae":
        return (32, 32)
    else:
        raise ValueError(f"Unsupported task {args.task}")

geometry = create_geometry(args)
# Normally this should be not done, but the implementations without grid won't initialize it, which is a problem
# for the mapping later.
geometry.grid.init(args.vq_vae.num_embeddings)


model = create_model(args, geometry)
model.load_state_dict(checkpoint['model'])
model.cuda()
model.eval()

id_to_name = {}
logged_tensors = {}
overwrite_quantization = {}

# diff_t, quant_t, perplexity1, id_t


def log_hook(module, inputs, outputs):
    name = id_to_name[id(module)]
    diff, quant, perplexity, ids = outputs
    ids_nice = ids.view(quant.shape[:-1])
    logged_tensors[name] = ids_nice

    if name in overwrite_quantization:
        new_ids = overwrite_quantization[name]

        new_ids = new_ids.type_as(ids).view_as(ids)
        new_quant = module.embed_code(new_ids).view_as(quant)

        return diff, new_quant, perplexity, new_ids


def register_hooks(model):
    for name, mod in model.named_modules():
        if isinstance(mod, (Quantize, HardSOM)):
            id_to_name[id(mod)] = name
            mod.register_forward_hook(log_hook)


def normalize(args, img):
    mean, std = get_mean_std(args)
    return (img - mean) / std


def unnormalize(args, img):
    mean, std = get_mean_std(args)
    img = img * std + mean
    return img.clamp(0, 255)

def resize_img(img, args):
    size = get_img_size(args)
    img = torch.tensor(img)
    return transforms.Resize(size)(img)

def run(args, model, img, overwrite = {}):
    global logged_tensors
    global overwrite_quantization

    img = resize_img(img, args)
    img = normalize(args, img)
    logged_tensors = {}
    overwrite_quantization = overwrite
    with torch.no_grad():
        img = model(img[None].cuda())[0].cpu()
    quantized = logged_tensors
    logged_tensors = {}
    return unnormalize(args, img), quantized


def load_image(fname: str) -> np.ndarray:
    img = np.array(Image.open(fname).convert("RGB"))
    return np.transpose(img, (2, 0, 1))


def save_image(fname: str, image: np.ndarray):
    image = np.transpose(image, (1, 2, 0))
    image = Image.fromarray(np.uint8(image))
    image.save(fname)



register_hooks(model)

img = load_image("cat.jpg")
res, quantized = run(args, model, img)
save_image("cat_out.jpg", res)

# # del quantized["quantize_b"]
# quant_hack = {}
# for n, q in quantized.items():
#     qs = geometry.grid.to_semantic_space(q)
#     assert (geometry.grid.from_semantic_space(qs) == q).all()
#     assert q.dtype == qs.dtype
#     assert q.device == qs.device
#     qs = qs + 1
#     qs = geometry.grid.clamp_to_limits(qs)

#     quant_hack[n] = geometry.grid.from_semantic_space(qs)

# res, _ = run(args, model, img, quant_hack)
# save_image("cat_out_shifted.jpg", res)


# os.makedirs("image_out", exist_ok=True)
# quant_hack = {k: v.clone() for k, v in quantized.items()}

def save_grid(fname, fix_const, model, img):
    images = []

    quant_hack = quantized.copy()

    for i in tqdm(range(args.vq_vae.num_embeddings)):
        for v in quant_hack.values():
            v.fill_(i)
        # quant_hack["quantize_t"].fill_(234)
        fix_const(quant_hack)
        res, _ = run(args, model, img, quant_hack)
        images.append(res)

    assert geometry.grid.dim in {1, 2}

    s = images[0].shape[1:]
    g = geometry.grid.shape
    if geometry.grid.dim == 1:
        g = g + [1]

    out = torch.zeros([3, (s[0] + 1) * g[0], (s[1] + 1) * g[1]])

    for i, img in enumerate(images):
        coord = geometry.grid._id_to_coord[i]
        if len(coord) == 1:
            coord = [coord[0], 0]
        out[:, coord[0] * (s[0] + 1) : coord[0] * (s[0] + 1) + s[0],
            coord[1] * (s[1] + 1) : coord[1] * (s[1] + 1) + s[1]] = img

    save_image(fname, out)

for i in {0, 123, 498}:
    def fix_t(d):
        d["quantize_t"].fill_(i)

    def fix_b(d):
        d["quantize_b"].fill_(i)

    semantic_coords = geometry.grid._id_to_coord[i].cpu().tolist()
    save_grid(f"image_out/vqvae2_all_b_t={semantic_coords}.jpg", fix_t, model, img)
    save_grid(f"image_out/vqvae2_all_t_b={semantic_coords}.jpg", fix_b, model, img)

# geometry.grid.shape


# for v in quant_hack.values():
#     v[0,:4].fill_(323)
#     v[0,4:].fill_(490)
# res, _ = run(args, model, img, quant_hack)
# save_image(f"image_out/half_323_490.jpg", res)


# img = load_image("dog.jpg")
# res, _ = run(args, model, img)
# save_image("dog_out.jpg", res)


# res, _ = run(args, model, img, quantized)
# save_image("dog_to_cat_test.jpg", res)

def do_shift(quantized, shift):
    quant_hack = {}
    for n, q in quantized.items():
        qs = geometry.grid.to_semantic_space(q)
        assert (geometry.grid.from_semantic_space(qs) == q).all()
        assert q.dtype == qs.dtype
        assert q.device == qs.device
        qs = qs + shift
        qs = geometry.grid.clamp_to_limits(qs)

        quant_hack[n] = geometry.grid.from_semantic_space(qs)
    return quant_hack



# def concat_img_list(ilist):
#     padding = 4
#     w = sum([i.shape[-1] for i in ilist])
#     res = torch.zeros([ilist[0].shape[0], ilist[0].shape[1], w + (len(ilist) - 1)*padding], dtype=ilist[0].dtype)
#     for j, img in enumerate(ilist):
#         res[:, :, (ilist[0].shape[-1]+padding) * j : (ilist[0].shape[-1]+padding) * j + img.shape[-1]] = img.cpu()

#     return res


def plot_on_axis(axis, img):
    plt.sca(axis)
    plt.imshow(img.permute(1,2,0).clamp(0,255).type(torch.uint8), interpolation="none")
    plt.yticks([])
    plt.xticks([])

def save_image_list(fname, ilist, get_title):
    h = ilist[0].shape[1]
    w = ilist[0].shape[-1] * len(ilist)
    figsize = ( w/30, h/30)

    fig, ax = plt.subplots(1, len(ilist), figsize=figsize, squeeze=False)
    for i in range(len(ilist)):
        plot_on_axis(ax[0, i], ilist[i])
        plt.title(get_title(i), fontsize=50)

    fig.savefig(fname, bbox_inches='tight', pad_inches = 0.01)



def shift_img_vqvae2_multiple(fname, offset=10, model=model, prefix=""):
    name_base = ".".join(fname.split(".")[:-1])

    img = load_image(fname)
    res, quantized = run(args, model, img)

    save_image(f"image_out/{name_base}_reconstruction.jpg", res)

    blist = []
    tlist = []
    bothlist = []
    for i in range(-offset, offset + 1, 2):
        quant_hack = do_shift(quantized, 2*i)

        quant_hack2 = quant_hack.copy()
        quant_hack2["quantize_b"] = quantized["quantize_b"]
        res, _ = run(args, model, img, quant_hack2)
        tlist.append(res)

        quant_hack2 = quant_hack.copy()
        quant_hack2["quantize_t"] = quantized["quantize_t"]
        res, _ = run(args, model, img, quant_hack2)
        blist.append(res)

        res, _ = run(args, model, img, quant_hack)
        bothlist.append(res)

    def get_title(i):
        return f"Offset = {-offset + 2*i}"
    save_image_list(f"image_out/{prefix}{name_base}_tlist.pdf", tlist, get_title)
    save_image_list(f"image_out/{prefix}{name_base}_blist.pdf", blist, get_title)
    save_image_list(f"image_out/{prefix}{name_base}_bothlist.pdf", bothlist, get_title)



# shift_img_vqvae2_multiple("k_dog.jpg", prefix="hard_")
# shift_img_vqvae2_multiple("k_macaron.jpg", prefix="hard_")

# shift_img_vqvae2("k_dog.jpg",torch.tensor([1,3], dtype=torch.int64).cuda())





# Loading a second model won't work normally because of the hardcoded global variables, but in case of no grid and
# same normalization, it should work.

checkpoint_none = torch.load("vq_vae2_none_imagenet.pth")

args_none = checkpoint_none['run_invariants']['args']
args_none = framework.data_structures.dotdict.create_recursive_dot_dict(args_none)

model_none = create_model(args_none, geometry)
model_none.load_state_dict(checkpoint_none['model'])
model_none.cuda()
model_none.eval()

register_hooks(model_none)

# shift_img_vqvae2_multiple("k_dog.jpg", prefix="none_", model=model_none)
# shift_img_vqvae2_multiple("k_macaron.jpg", prefix="none_", model=model_none)


def plot_double_shift(fname: str, offset=3, prefix=""):
    name_base = ".".join(fname.split(".")[:-1])

    img = load_image(fname)


    res, quantized = run(args, model, img)

    shift_som = []
    for i in range(-offset, offset + 1, 1):
        quant_hack = do_shift(quantized, i)
        res, _ = run(args, model, img, quant_hack)
        shift_som.append(res)

    res, quantized = run(args, model_none, img)

    shift_vq = []
    for i in range(-offset, offset + 1, 1):
        quant_hack = do_shift(quantized, i)
        res, _ = run(args, model_none, img, quant_hack)
        shift_vq.append(res)

    fig = plt.figure(figsize=[(len(shift_vq)) + 1, 2])

    gs = fig.add_gridspec(2, len(shift_vq) + 1, height_ratios=[1,1])
    for j, d in enumerate([shift_som, shift_vq]):
        for i in range(len(d)):
            plot_on_axis(fig.add_subplot(gs[j, i+1]), d[i])
            if j == 0:
                plt.title(f"Offset={-offset+i}", fontsize=6)

    plot_on_axis(fig.add_subplot(gs[:, 0]), resize_img(img, args))
    plt.title("Input", fontsize=6)
    fig.savefig(f"image_out/{prefix}{name_base}_shift_som_no_som.pdf", bbox_inches='tight', pad_inches = 0.01)

plot_double_shift("k_dog.jpg")
plot_double_shift("k_macaron.jpg")
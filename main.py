from typing import Optional
import framework
import tasks
import os
import torch
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = False


def register_args(parser: framework.helpers.ArgumentParser):
    tasks.register_args(parser)
    parser.add_argument("-batch_size", default=128)
    parser.add_argument("-lr", default=1e-3)
    parser.add_argument("-wd", default=0.0)
    parser.add_argument("-test_interval", default=1000)
    parser.add_argument("-stop_after", default="None", parser=parser.int_or_none_parser)
    parser.add_argument("-task", default="tuple")
    parser.add_argument("-grad_clip", default="1.0", parser=parser.float_or_none_parser)
    parser.add_argument("-test_batch_size", default="None", parser=parser.int_or_none_parser)
    parser.add_argument("-test_pretrained", default=1)
    parser.add_argument("-optimizer", default="adam", choice=["adam", "adamw", "sgd", "adagrad"])
    parser.add_argument("-adam.betas", default="0.9,0.999", parser=parser.float_list_parser)
    parser.add_argument("-adam.eps", default=1e-8)
    parser.add_argument("-amp", default=False)
    parser.add_argument("-length_bucketed_sampling", default=False)
    parser.add_argument("-speedtest", default="none", choice=["none", "iter"])
    parser.add_argument("-reg", default=1.0)
    parser.add_argument("-test_only", default=False)
    parser.add_argument("-log_grad_norms", default=False)
    parser.add_argument("-n_microbatch", default="none", parser=parser.int_or_none_parser)


def initialize(restore: Optional[str] = None):
    helper = framework.helpers.TrainingHelper(wandb_project_name="lm",
                                              register_args=register_args, extra_dirs=["export", "model_weights", "tmp"],
                                              log_async=True, restore=restore)

    task = tasks.get_task(helper.args.task)

    task = task(helper)
    return helper, task


def main():
    helper, task = initialize()

    if helper.args.test_only:
        helper.log(task.validate())
    else:
        task.train()

        print("Training finished. Saving model...")
        task.save_weights()

    task.finish()
    helper.finish()


if __name__ == "__main__":
    main()

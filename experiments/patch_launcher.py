from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def options(self):
        opt = Options()
        opt.set(
            dataroot="./datasets/sr",
            dataset_mode="imagefolder",
            num_gpus=1, batch_size=8,
            # scale the image such that the short side is |load_size|, and
            # crop a square window of |crop_size|.
            preprocess="crop",
            load_size=None, crop_size=256,
            display_freq=1000, print_freq=50,
            save_freq=5000,
            # save_freq=400,
        ),

        return [
            opt.specify(
                name="patch_default",
                patch_use_aggregation=False,
                patch_size=128,
            ),
        ]

    def train_options(self):
        common_options = self.options()
        return [opt.specify(
            continue_train=True,
            evaluation_metrics="swap_visualization",
            evaluation_freq=1000,
        ) for opt in common_options]

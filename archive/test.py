import data_io as bio_io
import train_detector as td

ROOT = "/kaggle/input/competitions/biohub-cell-tracking-during-development/train"

samples = bio_io.list_samples(ROOT)
s = samples[0]

vol = bio_io.open_zarr_volume(s.zarr_path)

print("Sample:", s.name)
print("Volume shape:", vol.shape)
print("DOWNSAMPLE:", td.DOWNSAMPLE)

window, target, weight = td.build_training_window(
    s, t_start=0, window_size=2
)

print("Window:", window.shape)
print("Target:", target.shape)
print("Weight:", weight.shape)
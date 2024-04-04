# Introduction

Try to learn [ThaumatoAnakalyptor](https://github.com/schillij95/ThaumatoAnakalyptor) step by step.

# Personal Notes

### generate_half_sized_grid.py

#### downsample_folder_tifs

假設 volumes 資料夾裡有 00000.tif 到 00099.tif 的連續資料，`downsample_folder_tifs` 方法會根據 `downsample_factor` 跳著採樣這些資料，好比說採樣值為 2，則會挑 0, 2, 4, ..., 98 的 tif 檔，然後會再透過 `multiprocessing` 非同步的傳給 `downsample_image` 方法讓圖片的長寬縮小 `downsample_factor` 倍，也就是說整個 image stacks 長寬高都被採樣了某個倍數。

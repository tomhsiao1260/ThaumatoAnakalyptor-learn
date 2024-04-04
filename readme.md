# Introduction

Try to learn [ThaumatoAnakalyptor](https://github.com/schillij95/ThaumatoAnakalyptor) step by step.

# Personal Notes

### generate_half_sized_grid.py

#### downsample_folder_tifs

假設 volumes 資料夾裡有 00000.tif 到 00099.tif 的連續資料，`downsample_folder_tifs` 方法會根據 `downsample_factor` 跳著採樣這些資料，好比說採樣值為 2，則會挑 0, 2, 4, ..., 98 的 tif 檔，然後會再透過 `multiprocessing` 非同步的傳給 `downsample_image` 方法讓圖片的長寬縮小 `downsample_factor` 倍，也就是說整個 image stacks 長寬高都被採樣了某個倍數。結果會存在 `2dtifs_8um` 資料夾內。

#### generate_grid_blocks

根據 `2dtifs_8um` 裡的資料，產生三維大小為 500, 500, 500 的 grids 資料，並將結果存在 `2dtifs_8um_grids` 資料夾。過程中會先計算 `blocks_in_x, blocks_in_y, blocks_in_z` 表示在不同軸的 grid 個數，然後非同步的分派任務給 `process_block` 方法，逐一產生 `cell_yxz_....tif` 檔。另外，`standard_size` 參數紀錄了單一 grid 的檔案大小，若過去產生過 grid 了，則能透過這個參數跳過重新產生的過程，如果是第一次跑，這個數值是 -1，grid 會從頭生成。

### grid_to_pointcloud.py

#### umbilicus

`umbilicus.txt` 紀錄了一系列卷軸的中心座標 (y, z, x)，透過 `load_xyz_from_file` 方法解析成 numpy，再透過 `umbilicus` 方法沿著 z 軸內插，產生連續的資料點。

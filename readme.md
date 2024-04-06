# Introduction

Try to learn [ThaumatoAnakalyptor](https://github.com/schillij95/ThaumatoAnakalyptor) step by step.

# Personal Notes

### generate_half_sized_grid.py

#### downsample_folder_tifs

假設 volumes 資料夾裡有 00000.tif 到 00099.tif 的連續資料，`downsample_folder_tifs` 方法會根據 `downsample_factor` 跳著採樣這些資料，好比說採樣值為 2，則會挑 0, 2, 4, ..., 98 的 tif 檔，然後會再透過 `multiprocessing` 非同步的傳給 `downsample_image` 方法讓圖片的長寬縮小 `downsample_factor` 倍，也就是說整個 image stacks 長寬高都被採樣了某個倍數。結果會存在 `2dtifs_8um` 資料夾內。

#### generate_grid_blocks

根據 `2dtifs_8um` 裡的資料，產生三維大小為 500, 500, 500 的 grids 資料，並將結果存在 `2dtifs_8um_grids` 資料夾。過程中會先計算 `blocks_in_x, blocks_in_y, blocks_in_z` 表示在不同軸的 grid 個數，然後非同步的分派任務給 `process_block` 方法，逐一產生 `cell_yxz_....tif` 檔。另外，`standard_size` 參數紀錄了單一 grid 的檔案大小，若過去產生過 grid 了，則能透過這個參數跳過重新產生的過程，如果是第一次跑，這個數值是 -1，grid 會從頭生成。

### grid_to_pointcloud.py

umbilicus 為遵守 ply 格式所以座標系為 (y, z, x)，然後 corner_coords 為遵守 Grid Cells 格式所以座標系為 (y, x, z)，所以裡面有一些計算要注意這些座標系的差異，通常跟 point cloud 有關的都使用前者，跟 grid cells 有關的都使用後者。然後 tif stack 的座標系則是 (z, y, x)。

#### umbilicus

`umbilicus.txt` 紀錄了一系列卷軸的中心座標 (y, z, x)，透過 `load_xyz_from_file` 方法解析成 numpy，再透過 `umbilicus` 方法沿著 z 軸內插，產生連續的資料點。

#### umbilicus_xy_at_z

給一個 z 值，回傳在那個切面下外插出的卷軸中心座標 (y, z, x)。

#### skip_computation_block

判斷正方體中心點和卷軸中心點距離是否過大，以至於要略過計算。具體上，會根據 corner_coords 和 grid_block_size 所形成的正方體的中心點，與 umbilicus 卷軸中心點間的距離，來判斷是否大於 max_distance (True 或 False)。但如果 max_distance 為 -1，表示沒上限限制，會直接回傳 False 表示不略過。

#### save_surface_ply

ply 的儲存，有 vertices, normals, colors 三種資料，前兩者為 (y, z, x)，後者為 (r, g, b)。

#### grid_inference

為核心程式碼，有 `GridDataset`、`DataLoader`、`PointCloudModel` 等等的資料準備和計算過程。

#### grid_empty

計算在三維空間的一個正方體內是否都沒有載下來要處理的 Grid Cells 資料，都沒有則回傳 True，有要處理則回傳 False。`path_template` 為 Grid Cells 路徑的模板，`cords` 是正方體的最小值，也就是起始位置，`grid_block_size` 是正方體的邊長，`cell_block_size` 就是每個 Grid Cells 資料的邊長。

#### get_reference_vector

粗略計算 normals。只要給一個正方體的角落點，就會根據正方體的中心點，和卷軸的中心點距離，推算出 normal 的方向 (z, y, x)。

#### GridDataset

總而言之，資料集選擇的方法是透過一系列的大小為 200 的正方體緊密連接來定義的，你需要先給個起始點，然後會有個遞迴方法從這個點蔓延開來，去尋找整塊連續有下載下來的 grid cells 作為資料集。

具體來說，這是個 torch dataset，初始化過程主要在產生 blocks_to_process 屬性，是個 list，用來記錄需要被處理的正方體，以正方體的最小的點座標作為儲存方式 (角落座標)，每個正方體的邊長為 200 也就是 grid_block_size。篩選的方式寫在 blocks_to_compute 裡，首先是給定一個 start_coord，這是你想觀察的一個正方體的角落座標，然後整個方法會做下面三件事：

1. grid_empty：把這個正方體擴充一個 padding，並看看這個擴充的範圍內有沒有在資料夾內可以處理的資料。沒有的話則捨棄掉這個正方體。

2. skip_computation_block：另一個捨棄機制，如果這個正方體的中心點離卷軸中心太遠的話會捨棄，反之則會保留下來。

3. neighbor_coords：如果都沒被上面兩種機制篩選掉的話，會再從這個正方體的周圍再挑出 8 個正方體重複上面的事。

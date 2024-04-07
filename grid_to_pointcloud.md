## grid_inference

核心程式碼的切入點在 grid_inference，首先要先定義一個 torch 的資料集，稱作 GridDataset，再來是運算這些資料的模型 PointCloudModel，以及要怎麼記錄下這些運算結果的 MyPredictionWriter。

#### GridDataset

定義了怎麼訪問 volume 資料，是一系列邊長為 200 的正方體，稱作 grid blocks。所有的正方體的最小頂點資訊 (y, x, z) 會以 list 的形式存在 blocks_to_process 屬性內。其中 `__getitem__` 提供了訪問某筆資料的方法，主要會回傳一個 volume 資料，是一個 padding 後邊長為 300 的正方體。還會回傳其他附加資料，像是初估的法向量、角落的頂點座標等等。

定義好一個 dataset 後就可以放進 DataLoader 裡面，決定要怎麼批量訪問這些資料。

#### PointCloudModel

然後我們透過 torch lighting 定義一個 model，這個模型不是用來訓練，而是利用一系列的張量運算來 inference 一些結果。實作方法寫在 forward 內，輸入 x 會是從 dataset 那裡的 `__getitem` 方法取來的資料，輸出就是 inference 後的結果。

#### MyPredictionWriter

有了這些資訊後，我們就能透過 torch lighting 的 Trainer 來計算一些結果了，但在那之前，我們需要先定義一個 writer 來決定要怎麼處理或儲存計算後的結果。實作寫在 MyPredictionWriter，其中 write_on_batch_end 方法會在每個 batch 結束後被呼叫，其中 predict 這個參數會提供 PointCloudModel 的輸出結果 (是個批量的 list)。

#### Trainer

現在我們就可以定義一個 torch lighting 的 Trainer，以前面提到的 writer 進行初始化。然後把對應的 model, dataloader 丟進 trainer.predict 方法執行運算。

---

#### GridDataset details

資料集的準備有不少細節，首先是資料集是透過一系列的大小為 200 的正方體緊密連接來定義的，你需要先給個起始點，然後會有個遞迴方法從這個點蔓延開來，去尋找整塊連續有下載下來的 grid cells 作為資料集。

具體來說，這是個 torch dataset，初始化過程主要在產生 blocks_to_process 屬性，是個 list，用來記錄需要被處理的正方體，以正方體的最小的點座標作為儲存方式 (角落座標)，每個正方體的邊長為 200 也就是 grid_block_size。篩選的方式寫在 blocks_to_compute 裡，首先是給定一個 start_coord，這是你想觀察的一個正方體的角落座標，然後整個方法會做下面三件事：

1. grid_empty：把這個正方體擴充一個 padding，並看看這個擴充的範圍內有沒有在資料夾內可以處理的資料。沒有的話則捨棄掉這個正方體。

2. skip_computation_block：另一個捨棄機制，如果這個正方體的中心點離卷軸中心太遠的話會捨棄，反之則會保留下來。

3. neighbor_coords：如果都沒被上面兩種機制篩選掉的話，會再從這個正方體的周圍再挑出 8 個正方體重複上面的事。

umbilicus 為遵守 ply 格式所以座標系為 (y, z, x)，然後 corner_coords 為遵守 Grid Cells 格式所以座標系為 (y, x, z)，所以裡面有一些計算要注意這些座標系的差異，通常跟 point cloud 有關的都使用前者，跟 grid cells 有關的都使用後者。然後 tif stack 的座標系則是 (z, x, y)。

cell block: 邊長為 500 的原始資料檔
grid block: 邊長為 200 的計算用資料

#### umbilicus

`umbilicus.txt` 紀錄了一系列卷軸的中心座標 (y, z, x)，透過 `load_xyz_from_file` 方法解析成 numpy，再透過 `umbilicus` 方法沿著 z 軸內插，產生連續的資料點。

#### umbilicus_xy_at_z

給一個 z 值，回傳在那個切面下外插出的卷軸中心座標 (y, z, x)。

#### skip_computation_block

判斷正方體中心點和卷軸中心點距離是否過大，以至於要略過計算。具體上，會根據 corner_coords 和 grid_block_size 所形成的正方體的中心點，與 umbilicus 卷軸中心點間的距離，來判斷是否大於 max_distance (True 或 False)。但如果 max_distance 為 -1，表示沒上限限制，會直接回傳 False 表示不略過。

#### save_surface_ply

ply 的儲存，有 vertices, normals, colors 三種資料，前兩者為 (y, z, x)，後者為 (r, g, b)。

#### grid_empty

計算在三維空間的一個正方體內是否都沒有載下來要處理的 Grid Cells 資料，都沒有則回傳 True，有要處理則回傳 False。`path_template` 為 Grid Cells 路徑的模板，`cords` 是正方體的最小值，也就是起始位置，`grid_block_size` 是正方體的邊長，`cell_block_size` 就是每個 Grid Cells 資料的邊長。

#### get_reference_vector

粗略計算 normals。只要給一個正方體的角落點，就會根據正方體的中心點，和卷軸的中心點距離，推算出 normal 的方向 (z, y, x)。

#### load_grid

回傳一個 grid block 的 numpy，座標為 (z, x, y)。會跟據檔案裡不存在的部分會回傳 0，最大值為 1。

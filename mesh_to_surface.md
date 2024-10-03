## mesh_to_surface

萃取出 3d tif 資料裡，指定的 obj 裡 segment 平坦化後的結果，也就是 layers 資料夾裡的 65 張 tif。

會先定義一個 `MeshDataset`，載入 obj, tif 資料和一些相關座標。並透過 `PPMAndTextureModel` 產生平坦化的計算結果。另外，Dataset 裡有定義 `MyPredictionWriter`，用來存下計算結果。

### MeshDataset

`load_mesh` 負責處理 obj 載入，`load_grid` 負責載入 3d tif。`__getitem__` 搭配 `custom_collate_fn` 定義怎麼批量丟資料給模型的 `forward` 方法。

### PPMAndTextureModel

`forward` 執行了所有的核心計算。其資料輸入是由 Dataset 的 `__getitem__` 經過 `custom_collate_fn` 產生的，輸出為萃取出的數值 values 和對應在空間中的座標點 grid_points。

### MyPredictionWriter

模型產生的 prediction 結果 values, grid_points 後，會觸發 writer 的 `write_on_batch_end` 做後續處理，最後會在跑 `write_tif` 把 65 張 tif 存下來。
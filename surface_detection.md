## surface_detection

核心程式碼在 surface_detection 方法。第一部會先模糊化，濾掉高頻雜訊。

#### uniform_blur3d

回傳一個用來模糊化的 function，是一層 3D convolution，用均勻的 kernal 達到模糊化的效果。

#### sobel_filter_3d

做邊緣偵測相關運算，會回傳一個 (300, 300, 300, 3) 的 tensor，是透過三層 3D convolution 產生的，用的是 sobel 的 kernal 作邊緣檢測。

#### adjusted_vectors_interp

法向量資訊存在 adjusted_vectors_interp，大小為 (300, 300, 300, 3)。計算過程會先把 sobel 的結果採樣為 (30, 30, 30, 3) 的大小，然後透過 vector_convolution 推算出法向量，再透過 adjust_vectors_to_global_direction 的修正，把這些向量對齊到與卷軸中心發出來的向量方向相近。最後再把這些 (30, 30, 30, 3) 的向量透過 interpolate_to_original 內插，擴充回原來的大小 (300, 300, 300, 3)。

#### first_derivative

把 sobel 張量投影到 adjusted_vectors_interp 上，並記錄長度，存為 (300, 300, 300) 的張量，且數值被歸一到介於 -1 ~ 1 之間。也就是只保留在法向量方向上的 sobel 數值，即垂直於紙片方向的強度變化，會在邊緣處有顯著亮點。

#### second_derivative

同樣邏輯，把 first_derivative 拿去算 sobel，然後再投影到 adjusted_vectors_interp 上，並記錄長度，存為 (300, 300, 300) 的張量，且數值被歸一到介於 -1 ~ 1 之間。也就是只保留在法向量方向上的 sobel 數值，即垂直於紙片方向強度變化的變化率，會在 first_derivative 圖案上的邊緣處有顯著的亮點。

#### recto & verso

有了沿著法向量一次和二次微分的張量，就可以根據不同條件式萃取出紙張的正面 recto 和背面 verso。首先，一次微分很正的大致就是 recto，很負的大致就是 verso，然後這兩個區域再透過二次微分的強度不要太大的條件來提取出更窄的區間。

#### save_ply

有了 recto 和 verso，就能提取出這兩個面對應的 points 和 normals。然後再透過 extract_size_tensor 方法萃取出在 padding 前的點有那些，以及透過 corner_coords 資訊將這些點的原始位置還原，並存成 ply。其中 recto 會被放在 point_cloud_recto 資料夾，verso 會被放在 point_cloud_verso 資料夾，檔案會命名為 cell_yxz_A_B_C.ply 的形式，其中 A, B, C 是以 grid_block_size 的跨步 200 下去計算的。

## surface_detection

核心程式碼在 surface_detection 方法。第一部會先模糊化，濾掉高頻雜訊。

#### uniform_blur3d

回傳一個用來模糊化的 function，是一層 3D convolution，用均勻的 kernal 達到模糊化的效果。

#### sobel_filter_3d

做邊緣偵測相關運算，會回傳一個 (300, 300, 300, 3) 的 tensor，是透過三層 3D convolution 產生的，用的是 sobel 的 kernal 作邊緣檢測，產生的結果會採樣為 (30, 30, 30, 3) 的大小。

## surface_detection

核心程式碼在 surface_detection 方法。第一部會先模糊化，濾掉高頻雜訊。

#### uniform_blur3d

回傳一個用來模糊化的 function，是一層 3D convolution，用均勻的 kernal 達到模糊化的效果。

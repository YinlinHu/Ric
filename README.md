
**This repository has been merged into [ximgproc component of OpenCV](https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/include/opencv2/ximgproc/sparse_match_interpolator.hpp)** (Thanks to Tobias Senst)

Code for《Robust Interpolation of Correspondences for Large Displacement Optical Flow》 CVPR 2017

Although the code here is much faster than the paper version, it has worse performance compared with the raw paper version. For the compatibility of windows platform, we use the default variational post-processing of OpenCV (3.1.0+), rather than the EpicFlow post-processing as the paper version do. If you want to reproduce the paper result accurately, please use the post-processing of EpicFlow (Only Unix-like platforms).

Here is only a demonstration implementation, and compiled on windows. For Unix-like platforms, some simple compiler configurations are needed. The directory "win32" saves binaries for windows platform.

Yinlin
2018.04.25

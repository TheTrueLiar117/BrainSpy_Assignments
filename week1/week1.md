**Manual Implementaion of Convolve function **

**Experiments with Image**

Performed gaussian blur and deduced the importance of using it for edge detection.

**Number of Methods/Kernels Used**
A total of 4 kernels were used naming:
Sobel(3x3 and 5x5)
Scharr(3x3 and 5x5)

**Comparison between different methods**

Firstly, i performed basic sobel and scharr with high threshold being 0.3 and low threshold being 0.1. Although there isn't much difference to be seen but scharr performed a little better.
Comparing the 3x3 and 5x5 kernels of the same types, it can be seen that the 5x5 one is better at detecting thin edges, this is because it addresses the noise in the image more efficiently.
Later on, increasing the high threshold ultimately decreases the amount of edge detection and vice versa.

After experimenting with the values, i found the values of HT and LT to be 0.15 and 0.15 using scharr 5x5 method


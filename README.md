# nn-image-compression
These scripts train an autoencoder-based neural network to compress and decompress images. The size of the compressed file is determined by the number of neurons in the bottleneck layer, and the compression rate is equivalent to the size of the bottleneck versus the size of the input image.

![Autoencoder Architecture](autoencoder_diagram.png?raw=true)

Results:

*Original image - 784 pixels, 8 bits per pixel*

![Original](results/autoencoder_20190806_185428_in_128.png?raw=true)

*Reconstructed image - from 128 values, ~1.3 bits per pixel*

![Reconstructed 128](results/autoencoder_20190806_185440_out_128.png?raw=true)

*Reconstructed image - from 64 values, ~0.65 bits per pixel*

![Reconstructed 64](results/autoencoder_20190806_185857_out_64.png?raw=true)

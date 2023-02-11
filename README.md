# Image-retrieval

## Image retrieval is one of the core problems which can be broken down into two sub tasks

- Represenation or Encondings
- Search engine

### Representation 
- In the image representation, one tries to encode an image compactly into a global or local features (the term features ,embeddings or encondings are used interchangeably) which captures crucial content in the image. The encoded features are then used to build the search space or a matrix of encodings.

### Image search problem
- In search problem, a database constructed using the encoded feature(s) of index images (which are also referred as gallery or reference images) is used to find the nearest neighbors matching to the input or query image. A simple example of search: finding k-Nearest-Neighbors of a query-embedding in a reference-embeddings database. 

### Approach using Deep learning to learn Representation

- Use 4 layers Convolution Autoencoders to learn low dimensional rich hidden representation of different type of images.

- Addition of noise in the image database to help the the model learn rich features from the image by removing the noise from images.

- Images are encoded in a hidden representation of 0’s and 1’s and stored in a matrix know as 'matrix of encondings'.

### Search a query image

- Query image is passed to the Conv model to extract it’s hidden representation,which is XORed with the repository matrix to get a score of similarity of the hidden representation with each of the image representation in the matrix

- The top 3 images with the highest score are the closest image results for a query image

- The closest matchable image found through XOR between the query image representation and each image hidden representation in the repository matrix to get a score of similarity.


### Results 

![results](https://user-images.githubusercontent.com/23450113/218266781-76952ebe-699a-4e58-b6f5-d11262af1fa3.png)

![results_shoes](https://user-images.githubusercontent.com/23450113/218266782-cf0637f0-bc59-4d5b-adfc-4defd7af7fee.png)


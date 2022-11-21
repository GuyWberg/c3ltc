# c3-LTC

A Sage implementation of Locally Testable Codes with constant rate, distance and locality based on the construction from [Dinur et al](https://arxiv.org/abs/2111.04808). 

### Setup

Requirements:
- Python >= 3.9.
- [Sage](https://doc.sagemath.org/html/en/installation/index.html). 
- a C++ compiler supporting C++11 standard.
- [Jupyter](https://jupyter.org/install).
- You also need to have make tools. On macOS run:
```brew install autoconf automake libtool```
On ubunto run:
```sudo apt-get install autotools-dev```

After installing the requirements, run the following:
- Run ``sh setup_spasm.sh`` in the cloned directory.  
- Then run Jupyter notebook with Sage kernel (``sage -n jupyter``) and open the `c3LTC.ipynb` notebook.  

This implementation uses (spasm)[https://github.com/cbouilla/spasm] library for sparse finite fields matrix computation to improve the runtime. 

### Example
```
G = PSL(2,7)
C_a = ReedSolomonCode(GF(7), Integer(6), Integer(4))
C_b = ReedSolomonCode(GF(7), Integer(6), Integer(4))
A = random_generators(G,6)
B = random_generators(G,6)
c3ltc = c3LTC(C_a, C_b, G, A, B)
```

The function `c3LTC(c3ltc,vertex)` gets the following parameters:

- `C_a` - Sage code object.
- `C_b` - Sage code object.
- `G` - Sage group object.
- `A` - a list of Sage group elements.
- `B` - a list of Sage group elements.


### Local view display

```local_view(c3ltc,vertex)```

Shows the local view of square numbers ("labels") for a vertex.

- `c3ltc` - c3LTC object
- `vertex` - number from 1 to the number of vertices in the graph (not 0!).

```show_common(c3ltc, vertex1, vertex2, side)```

Shows the local views of square numbers ("labels") for the two vertices with highlighted common row/column.

- `c3ltc` - c3LTC object
- `vertex1` - number from 1 to the number of vertices in the graph.
- `vertex2` - number from 1 to the number of vertices in the graph.
- `side` - "left"/"right" (depends on vertex1 being left (accosted with rows) neighbor of vertex2 or right (associated with columns) neighbors)

```local_view_in_word(c3ltc,word,vertex)```

Shows the local view matrix in a word.


- `c3ltc` - c3LTC object
- `word` - an array of field elements of a size that's compatible with c3ltc's length.
- `vertex` - number from 1 to the number of vertices in the graph.

```show_square(c3ltc,N)```

Shows the vertex numbers that participate in square `N`.

- `c3ltc` - c3LTC object
- `N` - number from 0 to the number of squares in the graph. 

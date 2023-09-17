# Overview
Python3 implementation of ANNOY (Approximate Nearest Neighbors Oh Yeah), with an Apache Spark (PySpark) implementation to distribute vector index over multiple nodes. All vector operations are done with numpy.


## How it works
Split index into approximately equal partitions and construct an ANNOY tree for each. When querying, query each tree (map) and then consolidate result vectors by how close they are to query vector.
Each worker node hosts the tree IN MEMORY (as a numpy array). So while the entire index does not need to fit in memory, each partition does.

## Vector Distance Measurements
Currently only supports euclidean distance (`np.linalg.norm`). I might add more later ~~depending on how lazy I'm feeling~~.

## References
Eric Bernhardsson's (ANNOY algorithm creator) post on how ANNOY works: https://erikbern.com/2015/10/01/nearest-neighbors-and-vector-models-part-2-how-to-search-in-high-dimensional-spaces.html

Pyspark Examples: https://github.com/spark-examples/pyspark-examples

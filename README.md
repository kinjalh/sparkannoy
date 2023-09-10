# Overview
Python3 implementation of ANNOY (Approximate Nearest Neighbors Oh Yeah), with an Apache Spark (PySpark) implementation to distribute vector index over multiple nodes. All vector operations are done with numpy since it's fast and I like fast things.

## Distributed Index (PySpark)
Split index into approximately equal partitions and construct an ANNOY tree for each. When querying, query each tree (map) and then consolidate result vectors by how close they are to query vector.

## Vector Stuff
Currently only supports euclidean distance (`np.linalg.norm`). I might add more later depending on ~~how lazy I'm feeling~~ demand from users.

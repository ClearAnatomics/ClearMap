Documentation of the Vasculature Graph Toolbox
==============================================
This is a summary of the properties that can typically be found in vascular graphs, built
using the GraphGt library in ClearMap2.
The graphs have the following properties:

Graph properties:
-----------------
n_pixels: number of pixels in the graph (i.e. number of nodes in the graph before pruning)

#. edge_geometry_annotation (n_pixels x 1, dtype=int64, each entry is an annotation ID)
#. edge_geometry_coordinates (n_pixels x 3, dtype=float64, each entry is a coordinate (typically in voxels, i.e. assuming isotropic voxels))
#. edge_geometry_coordinates_MRI (n_pixels x 3, dtype=float64, each entry is a coordinate, (in MRI units))
#. edge_geometry_coordinates_atlas (n_pixels x 3, dtype=float64, each entry is a coordinate, in atlas units)
#. edge_geometry_distance_to_surface (n_pixels x 1, dtype=float64, each entry is a distance to the surface of the voxel)
#. edge_geometry_radii (n_pixels x 1, dtype=float64, each entry is a radius, typically in voxels, i.e. assuming isotropic voxels)
#. edge_geometry_radii_atlas (n_pixels x 1, dtype=float64, each entry is a radius, in atlas units)
#. edge_geometry_type (string, either 'graph' or 'edge')
#. shape (3, dtype=int64, the shape of the original image (typically extracted from the skeleton image))


Vertex properties
-----------------
#. annotation (n_vertices x 1, dtype=int64, each entry is an annotation ID)
#. annotation_metaregions
#. annotation_structures
#. coordinates (n_vertices x 3, dtype=float64, each entry is a coordinate (typically in voxels, i.e. assuming isotropic voxels))
#. coordinates_MRI (n_vertices x 3, dtype=float64, each entry is a coordinate, (in MRI units))
#. coordinates_atlas (n_vertices x 3, dtype=float64, each entry is a coordinate, in atlas units)
#. distance_to_surface (n_vertices x 1, dtype=float64, each entry is a distance to the surface of the voxel. Typically in voxels, i.e. assuming isotropic voxels)
#. radii (n_vertices x 1, dtype=float64, each entry is a radius, typically in voxels, i.e. assuming isotropic voxels)
#. radii_atlas (n_vertices x 1, dtype=float64, each entry is a radius, in atlas units)
#. artery (n_vertices x 1, dtype=bool, each entry indicates whether the vertex is an artery)
#. vein (n_vertices x 1, dtype=bool, each entry indicates whether the vertex is a vein)
#. pressure (n_vertices x 1, dtype=float64, each entry is a blood pressure value)

Edge properties:
----------------
#. distance_to_surface  (n_edges x 1, dtype=float64, each entry is a distance to the surface of the voxel. Typically in voxels, i.e. assuming isotropic voxels)
#. edge_geometry_indices (n_edges x 2, dtype=int64, each entry is a pair of indices into the vertex list, indicating the start and end of the edge)
#. length (n_edges x 1, dtype=float64, each entry is the length of the edge, typically in voxels, i.e. assuming isotropic voxels)
#. radii (n_edges x 1, dtype=float64, each entry is a radius, typically in voxels, i.e. assuming isotropic voxels)
#. radii_atlas (n_edges x 1, dtype=float64, each entry is a radius, in atlas units)
#. artery (n_edges x 1, dtype=bool, each entry is a boolean indicating whether the edge is an artery)
#. vein (n_edges x 1, dtype=bool, each entry is a boolean indicating whether the edge is a vein)
#. flow (n_edges x 1, dtype=float64, each entry is the blood flow of the edge)
#. veloc (n_edges x 1, dtype=float64, each entry is the blood velocity of the edge)


Edge merging mappings:
----------------------
This is a description of the way to map each property when merging edges.
The mapping can typically be a mean, median, sum, max operation,
concatenation or taking the first or last value.
The mapping is specified as a dictionary, with the key being the property name,
and the value being the mapping function.

The following properties are mapped using the mean function:
  - Edges: distance_to_surface, radii, radii_atlas
  - Vertices: distance_to_surface, radii, radii_atlas

The following properties are mapped using the sum function:
  - length

The following properties are mapped using the first function:
  - Edges: edge_geometry_indices
  - Vertices: ann

Undecided:
    annotation, annotation_metaregions, annotation_structures
    edge_geometry_indices


No merging:
    Coordinates, coordinates_MRI, coordinates_atlas, shape, edge_geometry_type
import numpy as np

from ClearMap.Alignment import Annotation as ano


def get_sub_regions(reg_list, region_list, key='id'):
    """
    Get regions that share part of the name

    Parameters
    ----------
    reg_list
    region_list

    Returns
    -------

    """
    regions = []
    main_region_name = ano.find(region_list[0][0], key=key)['name']
    for region in reg_list.keys():
        reg_name = ano.find_name(region, key=key)
        if main_region_name in reg_name:
            for se in reg_list[region]:
                reg_name = ano.find_name(se, key=key)
                regions.append(reg_name)
    return regions


# def group_region(regions, struct_name, features, extra_reg_ids, struct_acronyms=None):
#     print(struct_name)
#     inds = []
#     for i, r in enumerate(regions):
#         id_, level = r[0]
#         n = ano.find(id_, key='id')['acronym']
#         if struct_acronyms is None:
#             struct_acronyms = [struct_name.upper()]
#         for acro in struct_acronyms:
#             if acro in n:
#                 print(n)
#                 inds.append(i)
#     grouped_regions = np.mean(features[:, inds, :], axis=1)
#     grouped_regions = np.expand_dims(grouped_regions, axis=1)
#     features = np.delete(features, inds, axis=1)
#     features = np.concatenate((features, grouped_regions), axis=1)
#     regions = np.delete(regions, inds, axis=0)
#     regions = np.concatenate((regions, np.expand_dims(np.array(extra_reg_ids), axis=0)))
#     return features, regions


def get_region_volume(region_leaves, atlas):  # FIXME: move to annotation
    return sum([np.sum(atlas == leaf[0]) for leaf in region_leaves])

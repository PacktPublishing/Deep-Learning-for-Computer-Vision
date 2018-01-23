def create_annoy(target_features):
    t = AnnoyIndex(layer_dimension)
    count = 0
    for idx, target_feature in enumerate(target_features):
        t.add_item(idx, target_feature)
    t.build(10)
    t.save(MODEL.path+'annoy.ann')
annoy_index = AnnoyIndex(10)
annoy_index.load(MODEL.path + 'annoy.ann')
matches = annoy_index.get_nns_by_vector(query_feature, 20)
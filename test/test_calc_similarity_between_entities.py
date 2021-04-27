from eval.quality import calc_similarity_between_entities

gold = (2, 5)
pred = (3, 5)

similarity, tp, fp, fn = calc_similarity_between_entities(gold, pred)
print(similarity, tp, fp, fn)

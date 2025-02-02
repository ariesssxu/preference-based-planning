from omnigibson.utils.bddl_utils import OBJECT_TAXONOMY
category = "apple"      # or any other category
synset = OBJECT_TAXONOMY.get_synset_from_category(category)
abilities = OBJECT_TAXONOMY.get_abilities(synset)
print(abilities)
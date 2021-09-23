"""Compute top-K statistics on results."""

import pickle
import os

tempt_dict_pickles = os.listdir("./results")
if len(tempt_dict_pickles) == 0:
    raise ValueError

m = len(tempt_dict_pickles) # see NERF paper

tempt_dicts = []
for i, filename in enumerate(tempt_dict_pickles):
    with open(os.path.join("./results", filename), 'rb') as file:
        tempt_dicts.append(pickle.load(file))

TOP_K = 5
n_reactions = len(tempt_dicts[0]["pred"])
for k in range(1, TOP_K+1):

    acc = 0
    for i in range(n_reactions):

        valid_preds = 0
        for tempt_dict in tempt_dicts:

            if tempt_dict["pred"][i] is not None:
                valid_preds += 1

            if valid_preds > k: break

            # if tempt_dict["predicted_rxns"][i]:
            #     acc += 1
            #     break

    print("TOP", k, "accuracy =", acc / n_reactions)
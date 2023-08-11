import pandas as pd   
import numpy as np
import copy

def build_emission_params(path, k):
    with open(path, mode="r") as fp:
        data = fp.read()
        lines = data.split("\n")
        emission = {}
        for line in lines:
            if line == "":
                continue
            [word, tag] = line.rsplit(" ",1)
            if tag not in emission.keys():
                emission[tag] = {}
            if word not in emission[tag].keys():
                emission[tag][word] = 0
            emission[tag][word] += 1
        emission = pd.DataFrame(emission).fillna(0)
        emission.loc["#UNK#",:] = 1
        for col in emission.columns:
            emission[col] = emission[col]/emission[col].sum()
        return emission
    
def build_transition_params(path):
    with open(path, mode="r") as fp:
        data = fp.read()
        lines = data.split("\n")
        transition = {}
        prev_tag = "START"
        for line in lines:
            if line == "":
                if prev_tag == "START":
                    continue
                tag = "STOP"
            else:
                [word, tag] = line.rsplit(" ",1)
            if prev_tag not in transition.keys():
                transition[prev_tag] = {}
            if tag not in transition[prev_tag].keys():
                transition[prev_tag][tag] = 0
            transition[prev_tag][tag] += 1
            if tag == "STOP":
                prev_tag = "START"
            else:
                prev_tag = tag
        if prev_tag != "START":
            if prev_tag not in transition.keys():
                transition[prev_tag] = {}
            if "STOP" not in transition[prev_tag].keys():
                transition[prev_tag]["STOP"] = 0
            transition[prev_tag]["STOP"] += 1
        transition = pd.DataFrame(transition).fillna(0)
        for col in transition.columns:
            transition[col] = transition[col]/transition[col].sum()
        return transition

def build_position_params(path):
    with open(path, mode="r") as fp:
        data = fp.read()
        sequences = data.split("\n\n")
        sequences = [sequence.split("\n") for sequence in sequences]
        transition = {}
        for sequence in sequences:
            if "" in sequence:
                continue
            prev_tag = "START"
            for i in range(len(sequence)+1):
                if i == len(sequence):
                    tag = "STOP"
                else:
                    [word, tag] = sequence[i].rsplit(" ", 1)
                if prev_tag not in transition.keys():
                    transition[prev_tag] = {}
                if tag not in transition[prev_tag].keys():
                    transition[prev_tag][tag] = []
                transition[prev_tag][tag].append(i/len(sequence))
                prev_tag = tag
        mean = copy.deepcopy(transition)
        spread = copy.deepcopy(transition)
        for v in transition.keys():
            for u in transition[v].keys():
                mean[v][u] = np.mean(mean[v][u])
                spread[v][u] = np.std(spread[v][u])
        mean = pd.DataFrame(mean)
        spread = pd.DataFrame(spread)
        return mean, spread
    
def pos_viterbi_alg(transition, emission, sequence, mean, mean_bias, spread, spread_bias):
    tree = []
    pred_path = []
    for i in range(len(sequence)+2):
        tree.append({})
        if i == 0:
            tree[i]["START"] = [None, 1]
        else:
            pos = (i-1)/len(sequence)
            for idx in transition.index:
                trans_values = {}
                if i == len(sequence)+1 and idx != "STOP":
                    continue
                elif i == len(sequence)+1:
                    emit_value = 1
                else:
                    if sequence[i-1] not in emission.index:
                        emit = "#UNK#"
                    else:
                        emit = sequence[i-1]
                    if idx == "STOP":
                        emit_value = 0
                    else:
                        emit_value = emission.loc[emit,idx]
                for state in tree[i-1].keys():
                    if state == "STOP":
                        continue
                    else:
                        pos_bias = 1-(np.abs(pos-mean.loc[idx, state])*mean_bias*(1-spread.loc[idx, state]*spread_bias))
                        if pos_bias < 0 or np.isnan(pos_bias):
                            pos_bias = 0
                        trans_value = tree[i-1][state][1]*emit_value*transition.loc[idx,state]*pos_bias
                    trans_values[state] = trans_value
                tree[i][idx] = [max(trans_values, key=trans_values.get), max(trans_values.values())]
    pred_state = "STOP"
    level = len(tree)-1
    pred_path.append(pred_state)
    while level > 0:
        pred_path.append(tree[level][pred_state][0])
        pred_state = tree[level][pred_state][0]
        level -= 1
    pred_path.reverse()
    return pred_path

def pos_viterbi(train_path, test_path, mean_bias, spread_bias):
    transition = build_transition_params(train_path)
    emission = build_emission_params(train_path, 1)
    mean, spread = build_position_params(train_path)
    with open(test_path, mode="r") as fp:
        data = fp.read()
        sequences = data.split("\n\n")
        sequences = [sequence.split("\n") for sequence in sequences]
        pred_paths = []
        out = ""
        for sequence in sequences:
            if "" in sequence:
                continue
            pred_path = pos_viterbi_alg(transition, emission, sequence, mean, mean_bias, spread, spread_bias)
            pred_paths.append(pred_path)
            for i in range(len(sequence)):
                out += f"{sequence[i]} {pred_path[i+1]}\n"
            out += "\n"
        [dir, file] = test_path.rsplit("/", 1)
        [name, ext] = file.rsplit(".", 1)
        path_out = f"{dir}/{name}.p4.out"
        with open(path_out, mode="w") as fp_out:
            fp_out.write(out)
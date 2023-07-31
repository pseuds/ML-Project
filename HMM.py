class HMM:
    def __init__(self, filepath):
        self.initialise_training(filepath)
        self.k = 0

    def initialise_training(self, filepath):
        f = open(filepath, "r", encoding="utf8")
        lines = f.readlines()

        # contains all x and y in sequence
        self.x_list = []
        self.y_list = []

        # contains list of all encountered words
        self.known_words = []
        # contains list of all encountered tags (assumes training set contains all possible tags :C)
        self.tags_list = []
        # {y tag: {count: no of y tags, x_list: list of words with y tag},
        # prevtag_count: {prevtag: no of times previous tag transitioned to y tag} }
        self.count_y_dict = {
            'START': {'count': 1, 'prevtag_count':{}},
            'STOP': {'count': 0, 'prevtag_count':{}},
        }

        self.emissions_dict = {}
        self.transitions_dict = {}
        self.policy_dict = {}

        # START tag
        prev_tag = 'START' 

        for line in lines: 
            l = line.split()
            try:
                if prev_tag == "START": self.count_y_dict["START"]['count'] += 1
                word = ' '.join(l[:-1])
                if not word in self.known_words: 
                    self.known_words.append(word)
                if not l[-1] in self.tags_list: 
                    self.tags_list.append(l[-1])
                    self.count_y_dict[l[-1]] = {}
                    self.count_y_dict[l[-1]]['count'] = 0
                    self.count_y_dict[l[-1]]['x_list'] = []
                    self.count_y_dict[l[-1]]['prevtag_count'] = {}
                self.x_list.append(word)
                self.y_list.append(l[-1])
                self.count_y_dict[l[-1]]['count'] += 1
                self.count_y_dict[l[-1]]['x_list'].append(word)
                try: self.count_y_dict[l[-1]]['prevtag_count'][prev_tag] += 1
                except: self.count_y_dict[l[-1]]['prevtag_count'][prev_tag] = 0
                prev_tag = l[-1]
            except:
                # empty lines will reach here
                try: self.count_y_dict['STOP']['prevtag_count'][prev_tag] += 1
                except: self.count_y_dict['STOP']['prevtag_count'][prev_tag] = 1
                self.count_y_dict["STOP"]['count'] += 1
                prev_tag = 'START'
        f.close()

    def emission_v1(self, x, y):
        count_yx = 0
        count_y = 0
        for i in range(len(self.x_list)):
            if self.y_list[i] == y: count_y+=1
            if self.y_list[i] == y and self.x_list[i] == x: count_yx+=1
        return count_yx/count_y

    def emission_v2(self, x, y):
        count_yx = 0
        count_y = self.count_y_dict[y]['count']
        for x_ in self.count_y_dict[y]['x_list']:
            if x_ == x: count_yx += 1
        if x == "#UNK#": return self.k/(count_y+self.k)
        else: return count_yx/(count_y+self.k)

    def transition(self, new_y, old_y):
        # print("new_y:", new_y, "old_y:", old_y)
        # print("self.count_y_dict[new_y]['prevtag_count']:", self.count_y_dict[new_y]['prevtag_count'])
        try: count_oldtonew = self.count_y_dict[new_y]['prevtag_count'][old_y]
        except: count_oldtonew = 0
        count_old = self.count_y_dict[old_y]['count']
        return count_oldtonew/count_old

    def predict_tag(self, x):
        highest_e = -1
        best_tag = None
        e_dd = {}
        for tag in self.tags_list:
            e_val = self.emission_v2(x, tag)
            if e_val > highest_e: 
                highest_e = e_val
                best_tag = tag
            e_dd[tag] = e_val
        return best_tag
    
    def evaluate(self, testfilepath):
        f = open(testfilepath, "r", encoding="utf8")
        w = open(f"Data/{testfilepath[5:7]}/dev.p1.out", "w", encoding="utf8")
        lines = f.readlines()
        for line in lines:
            if line == "\n": 
                w.write(line)
                continue
            line = line[:-1]
            if not line in self.known_words:
                self.k+=1
                result_tag = self.predict_tag("#UNK#")
            else:
                result_tag = self.predict_tag(line)
            w.write(f"{line} {result_tag}\n")
        w.close()
        f.close()
    
    def get_emission(self, y, x):
        try: result = self.emissions_dict[(y,x)]
        except: 
            result = self.emission_v2(x,y)
            self.emissions_dict[(y,x)] = result
        return result

    def get_transition(self, old_y, new_y):
        try: result = self.transitions_dict[(new_y, old_y)]
        except:
            result = self.transition(new_y, old_y)
            self.transitions_dict[(new_y, old_y)] = result
        return result

    def get_policy(self, step, y):
        if step == 0:
            return 1 if y == "START" else 0
        else: 
            try: 
                return self.policy_dict[step][y]
            except: 
                return (0, None)

    # finds the best prev tag and sets it
    def set_policy(self, step, current_tag, word):
        best_value = -1
        best_prevtag = None
        for prev_tag in self.tags_list:
            # print("step-1:",step-1, "prev_tag:",prev_tag)
            # print("self.get_policy(step-1, prev_tag):",self.get_policy(step-1, prev_tag))
            # print("self.get_transition(prev_tag, current_tag):",self.get_transition(prev_tag, current_tag))
            # print("self.get_emission(current_tag, word):",self.get_emission(current_tag, word))
            value = self.get_policy(step-1, prev_tag)[0] * self.get_transition(prev_tag, current_tag) * self.get_emission(current_tag, word)
            if value > best_value:
                best_value = value
                best_prevtag = prev_tag
        # self.policy_dict[step] = {}
        self.policy_dict[step][current_tag] = (best_value, best_prevtag)

    def viterbi(self, x_sequence):
        # Base case, where step is 0

        # ----- Moving forward recursively -----

        # init step 1
        current_step = 1
        self.policy_dict[current_step]={}
        for tag in self.tags_list:
            self.policy_dict[current_step][tag] = (self.get_policy(0,"START") * self.get_transition("START", tag) * self.get_emission(tag, x_sequence[current_step-1]), "START")
        current_step += 1

        # print("Init step 1 done.")
        # print(self.policy_dict[1])
        # breakpoint()

        # from 2nd word onwards
        for word in x_sequence[1:]:
            self.policy_dict[current_step] = {}
            for current_tag in self.tags_list:
                self.set_policy(current_step, current_tag, word)
            current_step += 1
        
        print("Backtracking begins.")
        # Backtracking
        current_step -= 1
        optimal_state_sequence = []
        next_tag = "STOP"
        for _ in range(len(x_sequence)):
            best_tag = None
            best_value = -1
            # iterate to get the best tag
            for tag in self.tags_list:
                try: 
                    this_value = self.get_policy(current_step, tag)[0] * self.get_transition(tag, next_tag)
                    if this_value > best_value:
                        # print("best_tag updated to", tag)
                        best_value = this_value
                        best_tag = tag
                except: 
                    print("Error. best_tag not retrieved")
                    print("self.get_policy[current_step][tag][0]:",self.get_policy[current_step][tag][0])
                    print("self.get_transition(tag, next_tag):",self.get_transition(tag, next_tag))
                    this_value = self.get_policy(current_step, tag)[0] * self.get_transition(tag, next_tag)
                    pass
            optimal_state_sequence.append(best_tag)
            current_step -= 1
            next_tag = best_tag
        optimal_state_sequence.reverse()
        return optimal_state_sequence
    
    def evaluate_viterbi(self, testfilepath):
        f = open(testfilepath, "r", encoding="utf8")
        w = open(f"Data/{testfilepath[5:7]}/dev.p2.out", "w", encoding="utf8")
        lines = f.readlines()
        sentence = []
        for line in lines:
            if line != "\n":
                sentence.append(line[:-1])
            else:
                w.write(line)
                optimal_tags = self.viterbi(sentence)
                for tag, word in zip(optimal_tags, sentence):
                    w.write(f"{word} {tag}\n")
        w.close()
        f.close()
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
        # contains list of all tags
        self.tags_list = ["B-positive", "B-neutral", "B-negative", "I-positive", "I-neutral", "I-negative", "O"]

        # count_y_dict = {y tag: { count: no of y tags, x_list: list of words with y tag }
        self.count_y_dict = {
            'START': {'count': 0},
            'STOP': {'count': 0}
        }
        for y in self.tags_list:
            self.count_y_dict[y] = {'count': 0, 'x_list': []}

        # initialise emissions dict, stores count of emissions
        self.emissions_count_dict = {}
        for y in self.tags_list: 
            self.emissions_count_dict[y] = {}

        # initialise transitions dict, stores count of transition occurences
        self.transition_count_dict = {}
        self.transition_count_dict['START'] = {}
        for y in self.tags_list: 
            self.transition_count_dict['START'][y] = 0
        self.transition_count_dict['START']['STOP'] = 0
        for y1 in self.tags_list:
            self.transition_count_dict[y1] = {}
            for y2 in self.tags_list:
                self.transition_count_dict[y1][y2] = 0
            self.transition_count_dict[y1]['STOP'] = 0

        self.policy_dict = {}

        # START tag
        prev_tag = 'START' 

        # for each line in training set
        for line in lines: 
            l = line.split()
            try:
                if prev_tag == "START": self.count_y_dict["START"]['count'] += 1
                word = ' '.join(l[:-1])
                current_tag = l[-1]
                # keep track of known words
                if not word in self.known_words: 
                    self.known_words.append(word)
                self.x_list.append(word)
                self.y_list.append(current_tag)
                # count no of times tag appears in training set, and add corresponding word to x_list
                self.count_y_dict[current_tag]['count'] += 1
                self.count_y_dict[current_tag]['x_list'].append(word)
                self.transition_count_dict[prev_tag][current_tag] += 1
                try: self.emissions_count_dict[current_tag][word] += 1
                except: self.emissions_count_dict[current_tag][word] = 1
                # set current tag to previous tag for next iteration
                prev_tag = current_tag
            except:
                # empty lines will reach here
                current_tag = "STOP"
                self.count_y_dict["STOP"]['count'] += 1
                self.transition_count_dict[prev_tag]["STOP"] += 1
                prev_tag = 'START' # for the next sentence
        f.close()

    # Part 1.1 : estimates the emission parameters from the training set using MLE
    def emission_v1(self, x, y):
        count_yx = self.emissions_count_dict[y][x]
        count_y = self.count_y_dict[y]['count']
        return count_yx/count_y

    # Part 1.2 : with consideration of #UNK# words
    def emission_v2(self, x, y):
        count_y = self.count_y_dict[y]['count']
        if x == "#UNK#": return self.k/(count_y+self.k)
        else: 
            try: count_yx = self.emissions_count_dict[y][x]
            except: count_yx = 0
            return count_yx/(count_y+self.k)

    # Part 2.1 : estimates the transition parameters from the training set using MLE
    def transition(self, new_y, old_y):
        count_oldtonew = self.transition_count_dict[old_y][new_y]
        count_old = self.count_y_dict[old_y]['count']
        return count_oldtonew/count_old

    # Part 1.3 : Implement a simple sentiment analysis system that produces the tag
    def predict_tag(self, x):
        highest_e = -1
        best_tag = None
        for tag in self.tags_list:
            e_val = self.emission_v2(x, tag)
            if e_val > highest_e: 
                highest_e = e_val
                best_tag = tag
        return best_tag
    
    # Part 1.4 : Write outputs using predict_tag() into dev.p1.out
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
    
    # get emission params of word x and tag y
    def get_emission(self, y, x):
        if not x in self.known_words:
            x = "#UNK#"
        return self.emission_v2(x,y)

    # get transition params of y_i-1 to y_i
    def get_transition(self, old_y, new_y):
        return self.transition(new_y, old_y)

    # get policy
    def get_policy(self, step, y):
        # Base case, where step is 0
        if step == 0:
            return 1 if y == "START" else 0
        else: 
            return self.policy_dict[step][y]

    # finds the best prev tag and sets it
    def set_policy(self, step, current_tag, word):
        best_value = -1
        best_tag = None
        for prev_tag in self.tags_list:
            value = self.get_policy(step-1, prev_tag)[0] * self.get_transition(prev_tag, current_tag) * self.get_emission(current_tag, word)
            if value > best_value:
                best_value = value
                best_tag = prev_tag
        self.policy_dict[step][current_tag] = (best_value, best_tag)

    # finds the k best prev tags and sets them
    def set_policy_kth(self, step, current_tag, word, kth):
        kth_ls = [(-1, None, None) for _ in range(kth)]
        for prev_tag in self.tags_list:
            for i in range(kth):
                value = self.get_policy(step-1, prev_tag)[i][0] * self.get_transition(prev_tag, current_tag) * self.get_emission(current_tag, word)
                self.insert_sort(kth_ls, (value, prev_tag, i))
        self.policy_dict[step][current_tag] = kth_ls

    # inserts and maintains sorted array, returns same array if value is less than all elements 
    def insert_sort(self, array, value):
        if value[0] > array[0][0]: array[0] = value
        array.sort(key=lambda x:x[0])

    # Part 2.2 : viterbi algorithm
    def viterbi(self, x_sequence):
        self.policy_dict = {}
        self.k = 1

        # Moving forward recursively
        # init step 1
        current_step = 1
        self.policy_dict[current_step] = {}
        word = x_sequence[0]
        for tag in self.tags_list:
            if not word in self.known_words: 
                word = "#UNK#"
            self.policy_dict[current_step][tag] = (self.get_transition("START", tag) * self.get_emission(tag, word), 'START')
        current_step += 1

        # from 2nd word onwards
        for word in x_sequence[1:]:
            if not word in self.known_words: 
                word = "#UNK#"
            self.policy_dict[current_step] = {}
            for current_tag in self.tags_list:
                self.set_policy(current_step, current_tag, word)
            current_step += 1

        # from last word to STOP
        best_value = -1
        best_tag = None
        self.policy_dict[current_step]={}
        for last_tag in self.tags_list:
            value = self.get_policy(current_step-1, last_tag)[0] * self.get_transition(last_tag, 'STOP')
            if value > best_value:
                best_value = value
                best_tag = last_tag
        self.policy_dict[current_step]['STOP'] = (best_value, best_tag)
        
        # Backtracking
        optimal_state_sequence = []
        next_tag = "STOP"
        for _ in range(len(x_sequence)):
            prev_tag = self.get_policy(current_step, next_tag)[1]
            optimal_state_sequence.append(prev_tag)
            current_step -= 1
            next_tag = prev_tag
        optimal_state_sequence.reverse()
        return optimal_state_sequence
    
    # Part 2.2 : write outputs using Viterbi algorithm
    def evaluate_viterbi(self, testfilepath):
        f = open(testfilepath, "r", encoding="utf8")
        w = open(f"Data/{testfilepath[5:7]}/dev.p2.out", "w", encoding="utf8")
        lines = f.readlines()
        sentence = []
        for line in lines:
            # if line is not empty, add the word to the sentence
            if line != "\n": sentence.append(line[:-1])
            # if line is empty, run viterbi on the sentence
            else:
                optimal_tags = self.viterbi(sentence)
                for tag, word in zip(optimal_tags, sentence):
                    w.write(f"{word} {tag}\n")
                sentence = []
                w.write("\n")
        w.close()
        f.close()

    # Part 3 : Viterbi algorithm to find kth best outputs
    def viterbi_kth(self, x_sequence, kth):
        self.policy_dict = {}
        self.k = 1

        # Moving forward recursively 
        # init step 1
        current_step = 1
        self.policy_dict[current_step] = {}
        word = x_sequence[0]
        for tag in self.tags_list:
            kth_ls = [(-1, None, None) for _ in range(kth)]
            if not word in self.known_words: 
                word = "#UNK#"
            pi_value = self.get_transition("START", tag) * self.get_emission(tag, word)
            self.insert_sort(kth_ls, (pi_value, "START", 0))
            self.policy_dict[current_step][tag] = kth_ls
        current_step += 1

        # from 2nd word onwards:
        for word in x_sequence[1:]:
            if not word in self.known_words:
                word = "#UNK#"
            self.policy_dict[current_step] = {}
            for current_tag in self.tags_list:
                self.set_policy_kth(current_step, current_tag, word, kth)
            current_step += 1
        
        # from last word to STOP
        kth_ls = [(-1, None, None) for _ in range(kth)]
        self.policy_dict[current_step] = {}
        for last_tag in self.tags_list:
            for i in range(kth):
                value = self.get_policy(current_step-1, last_tag)[i][0] * self.get_transition(last_tag, 'STOP')
                self.insert_sort(kth_ls, (value, last_tag, i))
        self.policy_dict[current_step]['STOP'] = kth_ls

        kth_best_sequences = []
        # Backtracking
        for i in range(kth):
            ith_best_seq = []
            next_tag = 'STOP'
            for _ in range(len(x_sequence)):
                prev_tag = self.get_policy(current_step, next_tag)[i][1]
                next_i = self.get_policy(current_step, next_tag)[i][2]
                ith_best_seq.append(prev_tag)
                current_step -= 1
                next_tag = prev_tag
                i = next_i
            ith_best_seq.reverse()
            kth_best_sequences.append(ith_best_seq)
        # reverse, so that best sequence starts at index 0
        kth_best_sequences.reverse()
        return kth_best_sequences 

    # Part 3 : write 2nd and 8th best output using Viterbi algorithm
    def evaluate_viterbi_kth(self, testfilepath):
        f = open(testfilepath, "r", encoding="utf8")
        w2 = open(f"Data/{testfilepath[5:7]}/dev.p3.2nd.out", "w", encoding="utf8")
        w8 = open(f"Data/{testfilepath[5:7]}/dev.p3.8th.out", "w", encoding="utf8")
        lines = f.readlines()
        sentence = []
        for line in lines:
            # if line is not empty, add the word to the sentence
            if line != "\n": sentence.append(line[:-1])
            # if line is empty, run viterbi on the sentence
            else:
                best_sequences = self.viterbi_kth(sentence,8)
                second_best = best_sequences[1]
                eighth_best = best_sequences[7]
                for tag2, tag8, word in zip(second_best, eighth_best, sentence):
                    w2.write(f"{word} {tag2}\n")
                    w8.write(f"{word} {tag8}\n")
                sentence = []
                w2.write("\n")
                w8.write("\n")
        w2.close()
        w8.close()
        f.close()
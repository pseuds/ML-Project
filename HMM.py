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
            'START': {'prevtag_count':{}},
            'STOP': {'prevtag_count':{}},
        }

        # START tag
        prev_tag = 'START' 

        for line in lines: 
            l = line.split()
            try:
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
                except: self.count_y_dict['STOP']['prevtag_count'][prev_tag] = 0
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
        count_oldtonew = self.count_y_dict[new_y]['prevtag_count'][old_y]
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
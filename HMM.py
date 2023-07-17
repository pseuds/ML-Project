class HMM:
    def __init__(self, filepath):
        self.initialise_training(filepath)
        self.k = 0

    def initialise_training(self, filepath):
        f = open(filepath, "r", encoding="utf8")
        lines = f.readlines()

        self.x_list = []
        self.y_list = []
        self.known_words = []
        self.tags_list = []
        self.count_y_dict = {}

        for line in lines: 
            l = line.split()
            try:
                if not l[0] in self.known_words: self.known_words.append(l[0])
                if not l[1] in self.tags_list: 
                    self.tags_list.append(l[1])
                    self.count_y_dict[l[1]] = {}
                    self.count_y_dict[l[1]]['count'] = 0
                    self.count_y_dict[l[1]]['x_list'] = []
                self.x_list.append(l[0])
                self.y_list.append(l[1])
                self.count_y_dict[l[1]]['count'] += 1
                self.count_y_dict[l[1]]['x_list'].append(l[0])
            except:
                pass
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
        # for i in range(len(self.x_list)):
        #     # debug
        #     # print("y",y,"self.y_list[i]",self.y_list[i])
        #     if self.y_list[i] == y: count_y+=1
        #     if self.y_list[i] == y and self.x_list[i] == x: count_yx+=1
        # debug
        # print("count_yx",count_yx,"count_y",count_y)
        for x_ in self.count_y_dict[y]['x_list']:
            if x_ == x: count_yx += 1

        if x == "#UNK#": return self.k/(count_y+self.k)
        else: return count_yx/(count_y+self.k)

    def predict_tag(self, x):
        highest_e = -1
        best_tag = None
        # debug
        e_dd = {}
        for tag in self.tags_list:
            e_val = self.emission_v2(x, tag)
            if e_val > highest_e: 
                highest_e = e_val
                best_tag = tag
            # debug
            e_dd[tag] = e_val
        # print(e_dd)
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

    def calculate_precision(self, goldfilepath, testfilepath):
        # TODO: NOT SURE IF CORRECT
        f_gold = open(goldfilepath, "r", encoding="utf8")
        f_test = open(testfilepath, "r", encoding="utf8")
        lines_gold = f_gold.readlines()
        lines_test = f_test.readlines()
        correct_count = 0
        total_count = 0
        for gl, tl in zip(lines_gold, lines_test):
            if gl == "\n":
                continue
            g = gl.split()
            t = tl.split()
            # total_count += 1
            # if t[1] != "O":
            total_count += 1
            if g[1] == t[1]: 
                correct_count += 1
        f_gold.close()
        f_test.close()
        return correct_count/total_count
    
    def calculate_recall(self, goldfilepath, testfilepath):
        # TODO: I DON'T KNOW
        f_gold = open(goldfilepath, "r", encoding="utf8")
        f_test = open(testfilepath, "r", encoding="utf8")
        lines_gold = f_gold.readlines()
        lines_test = f_test.readlines()
        correct_count = 0
        total_count = 0
        for gl, tl in zip(lines_gold, lines_test):
            if gl == "\n":
                continue
            g = gl.split()
            t = tl.split()
            # total_count += 1
            # if t[1] != "O":
            total_count += 1
            if g[1] == t[1]: 
                correct_count += 1
        f_gold.close()
        f_test.close()
        return correct_count/total_count
    
    def calculate_f_score(self, goldfilepath, testfilepath):
        prec = self.calculate_precision(goldfilepath, testfilepath)
        recall = self.calculate_recall(goldfilepath, testfilepath)
        return 2 / ( (1/prec) + (1/recall) ) 






# Testing Testing
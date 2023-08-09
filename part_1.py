from HMM import HMM

if __name__ == '__main__':
    print("Processing ES...")
    # For ES dataset
    ES_set = HMM("Data/ES/train")
    ES_set.evaluate("Data/ES/dev.in")

    print("Processing RU...")
    # For RU dataset
    RU_set = HMM("Data/RU/train")
    RU_set.evaluate("Data/RU/dev.in")
    
    print("Done.")
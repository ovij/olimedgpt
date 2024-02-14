from meld import *
from examples import D


def get_qa_pairs(D,openai_model):
    """
    Get list of QA pair objects and calcuate MELD thresholds in each
    """
    qapairs = []
    for q,a in D:
        qapair = MELDTestCase(q,a,client=client,openai_model=openai_model)
        qapair.test()
        qapairs.append(qapair)
    return qapairs

def get_Z(qapairs,Y):
    """
    Given list of qapairs with MELD, calculate % above threshold Y
    """
    Z=0
    for qapair in qapairs:
        qapair.get_MELD_threshold(Y)
        if qapair.meld_threshold:
            Z+=1
    return Z/len(D)

if __name__ == "__main__":

    OPENAI_MODEL = "gpt-4"
    qapairs = get_qa_pairs(D,OPENAI_MODEL)

    Y = 0.95
    Z = get_Z(qapairs,Y)
    print(f"{100-(100*Z)}% of test cases pass MELD check")

    # To check specific test cases access the different objects stored
    first_example = qapairs[0]
    q = first_example.q
    q2 = first_example.q2
    g = first_example.g
    l = first_example.l
    print(f"Question was (q): '{q}'\nthis was split with a second part (q2): '{q2}'\ngpt predicted g: '{g}'\nwhich was a LD distance ratio of l = {l}")




def split_by_MRR(f_path):
    with open(f_path) as fr:
        data = fr.readlines()
    mrr_sort = []
    for d in data[1::2]:
        [index, _, mrr] = d.split(" ")
        mrr_sort.append([int(index), float(mrr)])
    mrr_sort.sort(key=lambda x: x[1], reverse=True)
    split_size = int(len(mrr_sort) / 2)
    memorized_index = [x[0] for x in mrr_sort[:split_size]]
    not_memorized_index = [x[0] for x in mrr_sort[-split_size:]]
    all_index = memorized_index + not_memorized_index
    return all_index , memorized_index, not_memorized_index

def split_by_Acc(f_path):
    '''
    Count as memorized only when the gold is ranked first / chosen
    '''
    with open(f_path) as fr:
        data = fr.readlines()
    memorized_index = [x for x in range(len(data)) if int(data[x].strip()) == 1]
    not_memorized_index = [x for x in range(len(data)) if int(data[x].strip()) != 1]
    all_index = memorized_index + not_memorized_index
    return all_index , memorized_index, not_memorized_index
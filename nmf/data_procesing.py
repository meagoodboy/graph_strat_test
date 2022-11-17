def process_data(data, clinical,clin_param):
    
    data = data[~data.index.duplicated(keep='first')]
    pat = list(set(clinical.index).intersection(set(data.index)))
    data = data.loc[pat,:]
    clinical = clinical.loc[pat,:]
    y = list(set(clinical[clin_param]))
    y = [str(i) for i in y]
    y.sort()
    print(y)
    clin_pat = []
    for i in clinical.index:
        if clinical.loc[i,clin_param] in y:
            clin_pat.append(i)
    
    data = data.loc[clin_pat,:]
    clinical = clinical.loc[clin_pat,:]
    feature = []
    for i in clinical[clin_param]:
        for j in range(len(y)):
            if i == y[j]:
                feature.append(j+1)
                
    return data , feature
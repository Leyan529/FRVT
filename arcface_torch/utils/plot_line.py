
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dir = "WebFace42M_resnet_269_2022_9_10"
model = "resnet_269"
ep = 20

folder = ["IJBB", "IJBC"]
x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]

dtypes = np.dtype(
    [(str(lab), float) for lab in  x_labels ]
)




for f in folder:
    print(f)
    df = pd.DataFrame(np.empty(0, dtype=dtypes))
    if model == "resnet_269":        
        if f == "IJBB":
            data = ({
                '1e-06' :[41.39, 42.07, 44.82, ],
                '1e-05':[91.61, 90.69, 90.59],
                '0.0001':[95.25, 95.04, 95.12],
                '0.001':[96.86, 96.71, 96.79],
                '0.01':[98.02, 97.90, 97.99],
                '0.1':[98.84, 99.01, 98.92],
            })
            df = pd.DataFrame(data)
        
        elif f == "IJBC":
            data = ({
                '1e-06' : [91.39, 88.23, 89.06],
                '1e-05':  [95.04, 94.16, 94.20],
                '0.0001': [96.86, 96.61, 96.68],
                '0.001':  [97.98, 97.86, 97.93],
                '0.01': [98.70, 98.65, 98.71],
                '0.1': [99.22, 99.33, 99.32],
            })
            df = pd.DataFrame(data)

    for epoch in range(6, ep+1):   
        try:     
            fn = f"./work_dirs/{dir}/epoch_{epoch}/{f}/result.txt"
            file = open(fn, "r")
            line = file.readlines()[3].strip().split("|")[2:-1]
            line = [float(f.replace(" ", "")) for f in line ]
            new_row = { str(lab): line[idx] for idx, lab in enumerate(x_labels)}
            df = df.append(new_row, ignore_index=True)  
        except Exception as e:
            print(e)

    # print(df)   

    plt.figure(figsize=(12, 6), dpi=130)
    # Loss curve
    for label in x_labels:
        plt.plot(df[str(label)].values)
    plt.title(f)
    plt.legend([str(label) for label in x_labels], bbox_to_anchor =(1.0, 1.0), title="FPR threshold")
    plt.xticks(list(range(0, ep+1 - 6)), [f"epoch_{i}" for i in range(6, ep+1)] ,
       rotation=20)  # Set text labels and properties.
    
    plt.ylabel("Recall(%)")
    plt.xlabel("Epoch")

    plt.savefig(f'./work_dirs/{dir}/{f}.png', dpi=130)
    plt.show()

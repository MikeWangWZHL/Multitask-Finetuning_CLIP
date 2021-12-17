from __future__ import print_function
import time
import numpy as np
import pandas as pd
# from sklearn.datasets.mldata import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# %matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import torch

def main(dev_set, test_set, dataset, method, using_dataset, color_number = 10, if_use_embedding = False, pca_dim = 50, perplexity = 40):

    if if_use_embedding:
        dev_dict = {key:value['zeroshot_image_embedding'].numpy() for key,value in dev_set[using_dataset].items()}
        test_dict = {key:value['zeroshot_image_embedding'].numpy() for key,value in test_set[using_dataset].items()}
    else:
        dev_dict = {key:value['image_tensors'].numpy().reshape(30,-1) for key,value in dev_set[using_dataset].items()}
        test_dict = {key:value['image_tensors'].numpy().reshape(30,-1) for key,value in test_set[using_dataset].items()}

    dev_X = None
    dev_y = None
    for key,value in dev_dict.items():
        if dev_X is None:
            dev_X = value.copy()
            labels = np.array([key for i in range(value.shape[0])])
            dev_y = labels
        else:
            dev_X = np.vstack((dev_X,value))
            labels = np.array([key for i in range(value.shape[0])])
            dev_y = np.concatenate((dev_y,labels))

    test_X = None
    test_y = None
    for key,value in test_dict.items():
        if test_X is None:
            test_X = value.copy()
            labels = np.array([key for i in range(value.shape[0])])
            test_y = labels
        else:
            test_X = np.vstack((test_X,value))
            labels = np.array([key for i in range(value.shape[0])])
            test_y = np.concatenate((test_y,labels))

    label2index = {}
    for key in dev_set[using_dataset].keys():
        label2index[key] = len(label2index)
    # print(label2index)

    # for (X,y) in [(dev_X,dev_y), (test_X,test_y)]:

    X,y = dev_X,dev_y
    feat_cols = ['pixel'+str(i) for i in range(X.shape[1]) ]
    df_dev = pd.DataFrame(X,columns=feat_cols)
    # df['label'] = df['y'].apply(lambda i: str(i))
    df_dev['label'] = y
    df_dev['y'] = df_dev['label'].apply(lambda label: label2index[label])

    X,y = test_X,test_y
    feat_cols = ['pixel'+str(i) for i in range(X.shape[1]) ]
    df_test = pd.DataFrame(X,columns=feat_cols)
    df_test['label'] = y
    df_test['y'] = df_test['label'].apply(lambda label: label2index[label])
    X, y = None, None
    print(df_test.head)
    print('Size of the dataframe: {}'.format(df_test.shape))

    '''PCA'''
    # pca = PCA(n_components=3)
    # pca_result = pca.fit_transform(df[feat_cols].values)
    # df['pca-one'] = pca_result[:,0]
    # df['pca-two'] = pca_result[:,1] 
    # df['pca-three'] = pca_result[:,2]
    # print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    # plt.figure(figsize=(8,5))
    # sns.scatterplot(
    #     x="pca-one", y="pca-two",
    #     hue="y",
    #     palette=sns.color_palette("hls", 9),
    #     data=df.loc[:,:],
    #     legend="full",
    #     alpha=0.3
    # )
    # plt.show()

    '''t-SNE with PCA'''
    plt.figure(figsize=(16,8))

    dfs = [df_dev, df_test]
    for i in range(len(dfs)):
        df = dfs[i]
        time_start = time.time()
        
        pca = PCA(n_components=pca_dim)
        pca_result = pca.fit_transform(df[feat_cols].values)
        print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))

        tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity, n_iter=500)
        tsne_pca_results = tsne.fit_transform(pca_result)
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

        df['t-SNE-1'] = tsne_pca_results[:,0]
        df['t-SNE-2'] = tsne_pca_results[:,1]

        if i == 0:
            ax = plt.subplot(1, 2, 1)
            ax.title.set_text(f'{dataset} dev')
        else:
            ax = plt.subplot(1, 2, 2)
            ax.title.set_text(f'{dataset} test')
        sns.scatterplot(
            x="t-SNE-1", y="t-SNE-2",
            hue="label",
            palette=sns.color_palette("hls", color_number),
            data=df,
            legend="full",
            alpha=0.8,
            # ax = ax
        )
    if if_use_embedding:
        plt.savefig(f't-SNE-{dataset}-{method}-embedding-pac{pca_dim}-perp{perplexity}.png',dpi=300)
    else:
        plt.savefig(f't-SNE-{dataset}-datapoint-pac{pca_dim}-perp{perplexity}.png',dpi=300)
    # plt.show()


if __name__ == '__main__':

    """
        example code for plotting image embedding using t-SNE;
        you will need to run /src/get_embedding.py to get image embedding and store it in /src/visualization/embedding 
    """
    multitask = torch.load("./embedding/multitask_finetuned_embedding.pt")
    fomaml = torch.load("./embedding/fomaml_finetuned_embedding.pt")
    zeroshot = torch.load("./embedding/zeroshot_embedding.pt")
    
    color_number = 10
    pca_dim = 10
    perplexity = 40

    main(multitask, multitask, "Clevr_Counting", "multitask", "counting", color_number = color_number, if_use_embedding=True, pca_dim = 10, perplexity = 40)
    main(fomaml, fomaml, "Clevr_Counting", "fomaml","counting", color_number = color_number,if_use_embedding=True, pca_dim = 10, perplexity = 40)
    main(zeroshot, zeroshot, "Clevr_Counting", "zeroshot","counting", color_number = color_number,if_use_embedding=True, pca_dim = 10, perplexity = 40)
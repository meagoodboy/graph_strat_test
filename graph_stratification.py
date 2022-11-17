import os
import yaml
import sys

import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from torchvision.transforms import ToTensor
from torch_geometric.loader import DataLoader
from tqdm.auto import trange, tqdm
from torch_geometric import seed_everything
import torchvision

from GAE.encoder_models import *
from GAE.node_decoder_models import *
from GAE.test_train_validate import *
from GAE.utils import *
from GAE.wrappers import *
from clustering.clustering import *
from GAE.do_nmf import *
from GAE.data_processing.process_data import *

import wandb
wandb.login()

if __name__ == '__main__':
    
    config_file = str(sys.argv[1])
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device : ", device)

    omics_file = str(config['main']['omics_file'])
    network_file = str(config['main']['network_file'])
    survival_file = str(config['main']['survival_file'])
    log_save = str(config['main']['log_save'])

    learning_rate = float(config['main']['learning_rate'])
    num_features = int(config['main']['num_features'])
    out_channels = int(config['main']['out_channels'])
    num_epochs = int(config['main']['num_epochs'])
    batch_size = int(config['main']['batch_size'])
    rank = int(config['main']['rank'])


    cancer = str(config['meta']['cancer'])
    omic = str(config['meta']['omic'])
    gmodels = config['model']['gmodel']
    optim = str(config['model']['optim'])
    loss_funct = str(config['model']['lossfn'])
    wandb_project = str(config['model']['wandb'])

    seed_everything(36912)

    data = SingleOmicData(network_file, omics_file, 1)
    num_nodes = len(data.node_order)

    train_size = int(0.8 * len(data))
    x,y = torch.utils.data.random_split(data, lengths=[train_size, len(data) - train_size], generator=torch.Generator())

    train_loader = DataLoader(x, shuffle=True, batch_size=batch_size,num_workers=8)
    val_loader = DataLoader(y, shuffle=True, batch_size=batch_size, num_workers=8)
    encode_loader = DataLoader(data, shuffle=False, batch_size=1)

    final_result = pd.DataFrame(columns = ["Model_Name","Enocding_Type","Clustering_Algorithm","Clusters","Silhouette_Score","Survival_Pval","Cluster_score"])

    if not os.path.exists(log_save):
        os.makedirs(log_save)
    final_res_save = log_save +  optim + "_" + str(learning_rate) + "_results.csv"
    print("Saving the clustering results in : ",final_res_save)

    for gmodel in gmodels:

        ## creating save folders
        try:
            savefolder = "./new_res/" + cancer + "/" + omic + "/" + gmodel + "/" +  optim + "_" + str(learning_rate) +"/"
            savemodels = "./new_mod/" + cancer + "/" + omic + "/" + gmodel + "/" + optim + "_" +  str(learning_rate) +"/"
            savename = cancer + "_" + omic + "_" + gmodel + "_" + optim + "_" + str(learning_rate) +"_"

            savename = savemodels + savename 
            summaryin = savemodels + "runs"
            bestmodel = savename + "bestmodel.pt"
            finalmodel = savename + "model.pt" 
            configf = savename + "config.yml" 
            fencsave = savefolder + "final.csv"
            bencsave = savefolder + "best.csv"

            if not os.path.exists(savefolder):
                os.makedirs(savefolder)
            if not os.path.exists(savemodels):
                os.makedirs(savemodels)

            clear_cache()

            wandb.init(project=wandb_project)

            cfg = wandb.config
            cfg.update({"epochs" : num_epochs, "batch_size": batch_size, "lr" : learning_rate,"optim" : optim,"data_type" : omic , "cancer" : cancer,"save":savefolder,"model_type":gmodel},allow_val_change=True)

            model = GAEM( encoder = get_encoder(gmodel,in_channels = num_features, out_channels = out_channels),
                node_decoder = L2Linear(out_channels = out_channels, num_nodes = num_nodes, batch_size=batch_size) )

            print(model)
            print("total parameters in the model : ", calculate_num_params(model))
            print("total parameters in the encoder : ", calculate_num_params(model.encoder))
            print("total parameters in the node_decoder : ", calculate_num_params(model.node_decoder))
            print("total parameters in the decoder : ", calculate_num_params(model.decoder))

            model = model.to(device)

            optmizer = None
            if optim == "ADAM":
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            else:
                print("no optimiser selected")
                break

            lossfn = None
            if loss_funct == "focal":
                lossfn = torchvision.ops.focal_loss.sigmoid_focal_loss
            else:
                print("no loss selected")
                break


            all_val_loss = []
            for e in range(num_epochs):

                print("Epoch : ", e + 1,  " / " , num_epochs)

                train_loss = train(model,train_loader,optimizer, device,lossfn )
                val_loss = validate(model,val_loader,device, lossfn)
                all_val_loss.append(val_loss)
                wandb.log({"validation loss" : val_loss,"train loss" : train_loss})

                if e > 10:
                    if val_loss < min(all_val_loss):
                        torch.save(model.state_dict(), bestmodel)
                        print("Saved best model weights")

                if (e+1) % 20 == 0:
                    print("Saving Model")
                    torch.save(model.state_dict(), finalmodel)

            wandb.finish()

            final_vectors = encode(model,encode_loader,device)
            final_vectors = np.array(final_vectors).reshape(len(data.patients), -1)

            final_df = pd.DataFrame(final_vectors, index=data.patients, columns = data.node_order)
            final_df.to_csv(fencsave)
            
            try:
                stat_final = do_all_clustering(final_df, survival_file)
                a = pd.DataFrame( data = ([gmodel,"final", i[0] , i[1] ,i[2] ,i[3] ,i[4] ] for i in stat_final ), columns = [ "Model_Name" ,"Enocding_Type" ,"Clustering_Algorithm" ,"Clusters" ,"Silhouette_Score" ,"Survival_Pval", "Cluster_score"])
                final_result = pd.concat([final_result,a])
            except Exception as e:
                print(e)
                print("error in dataframe creation of final df ")

            ## nmf
            try: 
                alsnmf, nmf_loss = do_alsnmf(final_df, rank, num_epochs)
                print("alsnmf_loss : ", nmf_loss)            
                stat_final = do_all_clustering(alsnmf, survival_file)
                a = pd.DataFrame( data = ([gmodel,"alsnmf", i[0] , i[1] ,i[2] ,i[3] ,i[4] ] for i in stat_final ), columns = [ "Model_Name" ,"Enocding_Type" ,"Clustering_Algorithm" ,"Clusters" ,"Silhouette_Score" ,"Survival_Pval", "Cluster_score"])
                final_result = pd.concat([final_result,a])
            except:
                print("error in dataframe creation of alsnmf df ")

            try:
                nmf, nmf_loss = do_nmf(final_df, rank, num_epochs)
                print("nmf_loss : ", nmf_loss)
                stat_final = do_all_clustering(nmf, survival_file)
                a = pd.DataFrame( data = ([gmodel,"nmf", i[0] , i[1] ,i[2] ,i[3] ,i[4] ] for i in stat_final ), columns = [ "Model_Name" ,"Enocding_Type" ,"Clustering_Algorithm" ,"Clusters" ,"Silhouette_Score" ,"Survival_Pval", "Cluster_score"])
                final_result = pd.concat([final_result,a])
            except:
                print("error in dataframe creation of nmf df ")

            try:
                fpdnmf, nmf_loss = do_fpdnmf(final_df, rank, num_epochs)
                print("fpdnmf_loss : ", nmf_loss)
                stat_final = do_all_clustering(fpdnmf, survival_file)
                a = pd.DataFrame( data = ([gmodel,"fpdnmf", i[0] , i[1] ,i[2] ,i[3] ,i[4] ] for i in stat_final ), columns = [ "Model_Name" ,"Enocding_Type" ,"Clustering_Algorithm" ,"Clusters" ,"Silhouette_Score" ,"Survival_Pval", "Cluster_score"])
                final_result = pd.concat([final_result,a])
            except:
                print("error in dataframe creation of fpdnmf df ")
            
            try:
                pnmf, nmf_loss = do_pnmf(final_df, rank, num_epochs)
                print("pnmf_loss : ", nmf_loss)
                stat_final = do_all_clustering(pnmf, survival_file)
                a = pd.DataFrame( data = ([gmodel,"pnmf", i[0] , i[1] ,i[2] ,i[3] ,i[4] ] for i in stat_final ), columns = [ "Model_Name" ,"Enocding_Type" ,"Clustering_Algorithm" ,"Clusters" ,"Silhouette_Score" ,"Survival_Pval", "Cluster_score"])
                final_result = pd.concat([final_result,a])
            except:
                print("error in dataframe creation of pnmf df ")
            
            try:
                gnmf, nmf_loss = do_gnmf(final_df, rank, num_epochs)
                print("gnmf_loss : ", nmf_loss)
                stat_final = do_all_clustering(gnmf, survival_file)
                a = pd.DataFrame( data = ([gmodel,"gnmf", i[0] , i[1] ,i[2] ,i[3] ,i[4] ] for i in stat_final ), columns = [ "Model_Name" ,"Enocding_Type" ,"Clustering_Algorithm" ,"Clusters" ,"Silhouette_Score" ,"Survival_Pval", "Cluster_score"])
                final_result = pd.concat([final_result,a])
            except:
                print("error in dataframe creation of gnmf df ")
            
            
            final_result.to_csv(final_res_save)
            
        except Exception as e: 
            print("error in gmodel : " , gmodel)
            print(e)

    print("Done")

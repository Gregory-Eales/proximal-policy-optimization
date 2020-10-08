import torch
import os

def save_model(model, name="model_params"):
    torch.save(model.state_dict(), "{}.pt".format(name))

def create_exp_dir(experiment_name):

    exp_path = "experiments/{}".format(experiment_name)

    log_path = exp_path+"/logs"
    graph_path = exp_path+"/graphs"
    param_path = exp_path+"/params"
    result_path = exp_path+"/results"

    try:
        os.mkdir(exp_path)
        os.mkdir(log_path)
        os.mkdir(graph_path)
        os.mkdir(param_path)
        os.mkdir(result_path)

    except:
        pass

    return exp_path

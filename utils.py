import torch

def load_soft_emb_without_conj(template, save_name, device):
    premise_mode = (save_name.split('/')[0]).split("-")[1]
    emb_arg = save_name.split('/')[1].split(".")
    emb_path = ".".join([emb_arg[0], "fs", emb_arg[2], emb_arg[3], "log"])

    soft_type = torch.load('soft_embedding/type/{}.pth'.format(emb_path), map_location=device)[1:4]
    soft_subclass = torch.load('soft_embedding/subClassOf/{}.pth'.format(emb_path), map_location=device)[1:4]
    soft_subprop = torch.load('soft_embedding/subPropertyOf/{}.pth'.format(emb_path), map_location=device)[1:4]
    soft_domain = torch.load('soft_embedding/domain/{}.pth'.format(emb_path), map_location=device)[1:4]
    soft_range = torch.load('soft_embedding/range/{}.pth'.format(emb_path), map_location=device)[1:4]

    if "rdfs2" in save_name:
        if premise_mode[0] == '2':
            template.soft_embedding.weight.data[1:4] = soft_domain
            template.soft_embedding.weight.data[4:7] = soft_type
        else:
            template.soft_embedding.weight.data[1:4] = soft_type
    elif "rdfs3" in save_name:
        if premise_mode[0] == '2':
            template.soft_embedding.weight.data[1:4] = soft_range
            template.soft_embedding.weight.data[4:7] = soft_type
        else:
            template.soft_embedding.weight.data[1:4] = soft_type
    elif "rdfs5" in save_name:
        if premise_mode[0] == '2' and premise_mode[1] == '2':
            template.soft_embedding.weight.data[1:4] = soft_subprop
            template.soft_embedding.weight.data[4:7] = soft_subprop
            template.soft_embedding.weight.data[7:10] = soft_subprop
        elif premise_mode[0] == '2' and premise_mode[1] != '2':
            template.soft_embedding.weight.data[1:4] = soft_subprop
            template.soft_embedding.weight.data[4:7] = soft_subprop
        else:
            template.soft_embedding.weight.data[1:4] = soft_subprop
    elif "rdfs7" in save_name:
        if premise_mode[0] == '2':
            template.soft_embedding.weight.data[1:4] = soft_subprop
    elif "rdfs9" in save_name:
        if premise_mode[0] == '2' and premise_mode[1] == '2':
            template.soft_embedding.weight.data[1:4] = soft_subclass
            template.soft_embedding.weight.data[4:7] = soft_type
            template.soft_embedding.weight.data[7:10] = soft_type
        elif premise_mode[0] == '2' and premise_mode[1] != '2':
            template.soft_embedding.weight.data[1:4] = soft_subclass
            template.soft_embedding.weight.data[4:7] = soft_type
        elif premise_mode[0] != '2' and premise_mode[1] == '2':
            template.soft_embedding.weight.data[1:4] = soft_type
            template.soft_embedding.weight.data[4:7] = soft_type
        else:
            template.soft_embedding.weight.data[1:4] = soft_type
    elif "rdfs11" in save_name:
        if premise_mode[0] == '2' and premise_mode[1] == '2':
            template.soft_embedding.weight.data[1:4] = soft_subclass
            template.soft_embedding.weight.data[4:7] = soft_subclass
            template.soft_embedding.weight.data[7:10] = soft_subclass
        elif premise_mode[0] == '2' and premise_mode[1] != '2':
            template.soft_embedding.weight.data[1:4] = soft_subclass
            template.soft_embedding.weight.data[4:7] = soft_subclass
        else:
            template.soft_embedding.weight.data[1:4] = soft_subclass
    return template
    
def load_soft_emb_with_conj(template, save_name, device):
    premise_mode = (save_name.split('/')[0]).split("-")[1]
    emb_arg = save_name.split('/')[1].split(".")
    emb_path = ".".join([emb_arg[0], "fs", emb_arg[2], emb_arg[3], "log"])

    soft_type = torch.load('soft_embedding/type/{}.pth'.format(emb_path), map_location=device)[1:4]
    soft_subclass = torch.load('soft_embedding/subClassOf/{}.pth'.format(emb_path), map_location=device)[1:4]
    soft_subprop = torch.load('soft_embedding/subPropertyOf/{}.pth'.format(emb_path), map_location=device)[1:4]
    soft_domain = torch.load('soft_embedding/domain/{}.pth'.format(emb_path), map_location=device)[1:4]
    soft_range = torch.load('soft_embedding/range/{}.pth'.format(emb_path), map_location=device)[1:4]

    if "rdfs2" in save_name:
        if premise_mode[0] == '2' and premise_mode[1] == '2':
            template.soft_embedding.weight.data[1:4] = soft_domain
            template.soft_embedding.weight.data[6:9] = soft_type
        elif premise_mode[0] == '2' and premise_mode[1] != '2':
            template.soft_embedding.weight.data[1:4] = soft_domain
            template.soft_embedding.weight.data[5:8] = soft_type
        else:
            template.soft_embedding.weight.data[2:5] = soft_type
    elif "rdfs3" in save_name:
        if premise_mode[0] == '2' and premise_mode[1] == '2':
            template.soft_embedding.weight.data[1:4] = soft_range
            template.soft_embedding.weight.data[6:9] = soft_type
        elif premise_mode[0] == '2' and premise_mode[1] != '2':
            template.soft_embedding.weight.data[1:4] = soft_range
            template.soft_embedding.weight.data[5:8] = soft_type
        else:
            template.soft_embedding.weight.data[2:5] = soft_type
    elif "rdfs5" in save_name:
        if premise_mode[0] == '2' and premise_mode[1] == '2':
            template.soft_embedding.weight.data[1:4] = soft_subprop
            template.soft_embedding.weight.data[5:8] = soft_subprop
            template.soft_embedding.weight.data[9:12] = soft_subprop
        elif premise_mode[0] == '2' and premise_mode[1] != '2':
            template.soft_embedding.weight.data[1:4] = soft_subprop
            template.soft_embedding.weight.data[5:8] = soft_subprop
        else:
            template.soft_embedding.weight.data[2:5] = soft_subprop
    elif "rdfs7" in save_name:
        if premise_mode[0] == '2':
            template.soft_embedding.weight.data[1:4] = soft_subprop
    elif "rdfs9" in save_name:
        if premise_mode[0] == '2' and premise_mode[1] == '2':
            template.soft_embedding.weight.data[1:4] = soft_subclass
            template.soft_embedding.weight.data[5:8] = soft_type
            template.soft_embedding.weight.data[9:12] = soft_type
        elif premise_mode[0] == '2' and premise_mode[1] != '2':
            template.soft_embedding.weight.data[1:4] = soft_subclass
            template.soft_embedding.weight.data[5:8] = soft_type
        elif premise_mode[0] != '2' and premise_mode[1] == '2':
            template.soft_embedding.weight.data[1:4] = soft_type
            template.soft_embedding.weight.data[5:8] = soft_type
        else:
            template.soft_embedding.weight.data[2:5] = soft_type
    elif "rdfs11" in save_name:
        if premise_mode[0] == '2' and premise_mode[1] == '2':
            template.soft_embedding.weight.data[1:4] = soft_subclass
            template.soft_embedding.weight.data[5:8] = soft_subclass
            template.soft_embedding.weight.data[9:12] = soft_subclass
        elif premise_mode[0] == '2' and premise_mode[1] != '2':
            template.soft_embedding.weight.data[1:4] = soft_subclass
            template.soft_embedding.weight.data[5:8] = soft_subclass
        else:
            template.soft_embedding.weight.data[2:5] = soft_subclass
    return template
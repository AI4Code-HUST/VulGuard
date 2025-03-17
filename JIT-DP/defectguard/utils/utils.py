from importlib.resources import files
import os, gdown
import json
import math

SRC_PATH = str(files('defectguard'))

IDS = {
    'deepjit': {
        'hyperparameters': '18TIg-2DhxI0Ou0vdUg5g8nx6c5GHEzWQ',
        'Java': '10XhImFwLt2VZ7Ra7nydN2bhBm3nEW6wY',
        'Java_dictionary': '1o2CgfK3KnvS4Ud2xZdXTEkKTaQ-z6Ioa',
        'Go': '1vf6LWG2BVkvgl5VOpxguAXYuIITtPOOs',
        'Go_dictionary': '1vZXzNTMsuh028pgBETRiPsNJv-k5ad5X',
        'C++': '17FJyecOmrII69Tb0mEnE9l0O-FgjN--b',
        'C++_dictionary': '1QGgZtaMfbvdCmp_-BW0HbrG_5mEhpixz',
        'C': '1v_MIMMj30xw3T_Yc_wl1xPTxMnpOTBM9',
        'C_dictionary': '1lc055RAd-dttq3DRyM5oAMjpYJ3iAO4S',
        'JavaScript': '1P4Zs35jGS9QCEk4mWX4o90C2C2hxkNtF',
        'JavaScript_dictionary': '1oSfvgufyBgG0oBMCzu_RlCjCkLsbEibp',
        'Python': '18drD9eS9R54S_eVF3bGC001MoKaCkcZ5',
        'Python_dictionary': '1P326mIDCwWRK_HEX4YcJoSGNgKDIruIW',
    },
    'simcom': {
        'hyperparameters': '18TIg-2DhxI0Ou0vdUg5g8nx6c5GHEzWQ',
        'sim_Java': '1jEvRx4OwScOHh35_5GhlrXdmpU8Hd1nr',
        'sim_C++': '1jlXULkT6spmvlAYp_g0kOaZjWsUj0sJy',
        'sim_Go': '14it5ddgzhVUo-EVBqNslp7fBtf-fSAqn',
        'sim_C': '1RGwQU4xE6QhvvS-wWR4VDlJaE4Bn1_zN',
        'sim_Python': '1b3oCvmpYtMEmbgSC_U40pgxZtrqKIC1a',
        'sim_JavaScript': '1vwXellUEyDoBJn2nMI_GwwiNaRuktQGU',
        'com_Java': '1dqMQ6eSuWgGObDAvb5S9SsPLhJLp8Kxv',
        'Java_dictionary': '1o2CgfK3KnvS4Ud2xZdXTEkKTaQ-z6Ioa',
        'com_Go': '1whs8xBN7DoYRcqVzsckatgsQRE2nzC7L',
        'Go_dictionary': '1vZXzNTMsuh028pgBETRiPsNJv-k5ad5X',
        'com_C++': '1-DNk8bJ8BJMNKRwrCJ444f4gK7bJKHKk',
        'C++_dictionary': '1QGgZtaMfbvdCmp_-BW0HbrG_5mEhpixz',
        'com_C': '1pw9lUJHrtiUJMFK9BkKG5MxMm1fQtbzr',
        'C_dictionary': '1lc055RAd-dttq3DRyM5oAMjpYJ3iAO4S',
        'com_Python': '1G6QgVSljuM3s-8HTdgCwb5Kxt9bxHqbh',
        'Python_dictionary': '1P326mIDCwWRK_HEX4YcJoSGNgKDIruIW',
        'com_JavaScript': '1mneU-7tTWNMXP3KswD05ChZWLCC135gL',
        'JavaScript_dictionary': '1oSfvgufyBgG0oBMCzu_RlCjCkLsbEibp',
    },
    'tlel': {
        'Java': '1vEg336LpFv_Ey_6NUCPfct8YnQnGKZY_',
        'C++': '1QzmBmfEaZpF6OHKo-aZ_b-1pE8FCp3nu',
        'Go': '1ALIGUy_PSMubr5Ei_--f_XKXmApBljvv',
        'Python': '1YSiNOo2rfRLEVLf3ICmd3Rd_-vGEUR9z',
        'JavaScript': '1INd_AgKPxV3BzLOcmDE2wAVHID53AFj0',
        'C': '1Gqvf2UsxFWsYggGahSjjz-GLzZQCssWB',
    },
    'lapredict': {
        'Java': '1_zgRMnWBNiaNCkmwFXRK-dc5g1LhUCzT',
        'C++': '1e8GZLwG0U1NV1lreDXX9-LZRj6MMAWTl',
        'Go': '1ZhMR1HuJW-x-IcndV_CvpG5VQlJpI16T',
        'Python': '1iybtJnuFa3_4u9bKgzJ4k4hHmpAh-fPg',
        'JavaScript': '1VUSi21ffjhqoVhmBYNPzrJjf7zoaPGac',
        'C': '1kVEuBnUqIYjd0PTW1RVdI6VnN9bEmtw5',
    },
    'lr': {
        'Java': '1kaoET2cUHT8R1taTWR9Mg2JWFE7l4S60',
        'C++': '1l568vKULAIugh4zBcPWoMOtLWY4vqC6y',
        'Go': '1Ly7tcC6z3xjZfNtwnd6WlznK17w7iTKS',
        'Python': '1cdjZuFdn69EHiBYyW0CV_j4SjUf6uwXa',
        'JavaScript': '1tKjPOEzspnpLU-l00AMPGI5DpCLJPCmx',
        'C': '1_Y8lNtJEjNvHz7hb_ya4NObzwbv7ahZ9',
    },
    'cc2vec': '',
    'jitline': '',
}

def sort_by_predict(commit_list):
    # Sort the list of dictionaries based on the "predict" value in descending order
    sorted_list = sorted(commit_list, key=lambda x: x['predict'], reverse=True)
    return sorted_list

def vsc_output(data):
    # Extract the commit hashes from "no_code_change_commit"
    no_code_change_commits = data.get("no_code_change_commit", [])
    
    # Extract the "deepjit" list
    deepjit_list = data.get("deepjit", [])
    
    # Create a dictionary with keys from "no_code_change_commit" and values as -1
    new_dict = [{'commit_hash': commit, 'predict': -1} for commit in no_code_change_commits]
    
    # Append the new dictionary to the "deepjit" list
    deepjit_list += (new_dict)
    
    # Update the "deepjit" key in the original data
    data["deepjit"] = deepjit_list

    return data

def check_threshold(data, threshold):
    output = {}
    for model, predicts in data.items():
        if model == "no_code_change_commit":
            continue
        else:
            output[model] = []
        for predict in predicts:
            if predict['predict'] >= threshold:
                output[model].append(predict)
    
    return output

def create_download_list(model_name, language):
    download_list = []
    dictionary = f'{language}_dictionary'
    version = f'{language}'

    if model_name == 'simcom':
        sim_language = f'sim_{version}'
        com_language = f'com_{version}'
        download_list.append(sim_language)
        download_list.append(com_language)
        download_list.append('hyperparameters')
        download_list.append(dictionary)
    elif model_name == 'cc2vec':
        cc2vec_language = f'cc2vec_{version}'
        dextended_language = f'dextended_{version}'
        download_list.append(cc2vec_language)
        download_list.append(dextended_language)
        download_list.append('hyperparameters')
        download_list.append(dictionary)
    elif model_name == 'deepjit':
        download_list.append(version)
        download_list.append('hyperparameters')
        download_list.append(dictionary)
    else:
        download_list.append(version)
    
    return download_list

def download_file(file_id, folder_path):
    if not os.path.isfile(folder_path):
        gdown.download(f'https://drive.google.com/file/d/{file_id}/view?usp=sharing', output=folder_path, fuzzy=True)

def download_folder(model_name, language):
    # Check if the file exists locally
    folder_path = f'{SRC_PATH}/models/metadata/{model_name}'

    if not os.path.exists(folder_path):
        # File doesn't exist, download it
        # Create the directory if it doesn't exist
        print(f"Directory: {folder_path}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # Download model's metadata
    download_list = create_download_list(model_name, language)
    for item in download_list:
        download_file(IDS[model_name][item], f'{folder_path}/{item}')

def yield_jsonl(file):  
    # Read the file and yield lines in equal parts
    with open(file, 'r') as f:
        for line in f:
            yield json.loads(line)

def open_jsonl(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data
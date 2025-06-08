import numpy as np

def padding_message(data, max_length):
    return [padding_length(line=d, max_length=max_length) for d in data]

def mapping_dict_msg(pad_msg, dict_msg):
    return np.array(
        [np.array([dict_msg[w.lower()] if w.lower() in dict_msg.keys() else dict_msg['<NULL>'] for w in line.split(' ')]) for line in pad_msg])

def padding_data(data, dictionary, params, type):
    if type == 'msg':
        pad_msg = padding_message(data=data, max_length=params['message_length'])
        pad_msg = mapping_dict_msg(pad_msg=pad_msg, dict_msg=dictionary)
        return pad_msg
    elif type == 'code':
        pad_code = padding_commit_code(data=data, max_line=params['code_line'], max_length=params['code_length'])
        pad_code = mapping_dict_code(pad_code=pad_code, dict_code=dictionary)
        return pad_code
    else:
        print('Your type is incorrect -- please correct it')
        exit()

def padding_data_point(data_point, dictionary, params, type):
    if type == 'msg':
        pad_msg = padding_length(line=data_point, max_length=params['message_length'])
        return np.array([dictionary[w.lower()] if w.lower() in dictionary.keys() else dictionary['<NULL>'] for w in pad_msg.split(' ')])
    elif type == 'code':
        pad_code = [[padding_length(line=line, max_length=params['code_length']) for line in data_point]]
        pad_code = padding_commit_code_line(pad_code, max_line=params['code_line'], max_length=params['code_length'])
        return np.array([np.array([dictionary[w.lower()] if w.lower() in dictionary else dictionary['<NULL>'] for w in l.split()]) for l in pad_code[0]])
    else:
        print('Your type is incorrect -- please correct it')
        exit()

def padding_commit_code(data, max_line, max_length):
    padding_length = padding_commit_code_length(data=data, max_length=max_length)
    return padding_commit_code_line(
        padding_length, max_line=max_line, max_length=max_length
    )

def padding_commit_code_length(data, max_length):
    return [padding_multiple_length(lines=commit, max_length=max_length) for commit in data]
        
def padding_multiple_length(lines, max_length):
    return [padding_length(line=l, max_length=max_length) for l in lines]

def padding_length(line, max_length):
    line_length = len(line.split())
    if line_length < max_length:
        return str(line + ' <NULL>' * (max_length - line_length)).strip()
    elif line_length > max_length:
        line_split = line.split()
        return ' '.join([line_split[i] for i in range(max_length)])
    else:
        return line
    
def padding_commit_code_line(data, max_line, max_length):
    new_data = []
    for d in data:
        if len(d) == max_line:
            new_data.append(d)
        elif len(d) > max_line:
            new_data.append(d[:max_line])
        else:
            num_added_line = max_line - len(d)
            for _ in range(num_added_line):
                d.append(('<NULL> ' * max_length).strip())
            new_data.append(d)
    return new_data

def mapping_dict_code(pad_code, dict_code):
    return np.array([np.array([np.array([dict_code[w.lower()] if w.lower() in dict_code else dict_code['<NULL>'] for w in l.split()]) for l in ml]) for ml in pad_code])
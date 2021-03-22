import numpy as np

def getData_2(sub_data):

    
    inputs = dict(
            probed_Ch = np.array(sub_data["probed_Ch"])[:, None],
            unprobed_Ch = np.array(sub_data["unprobed_Ch"])[:, None],
                 )                  
                
    dat = dict(
        
        probed_Ch = np.array(sub_data["probed_Ch"]),
        unprobed_Ch = np.array(sub_data["unprobed_Ch"]),

        correct=np.array(sub_data['correct']),
        answer=np.array(sub_data['answer']),
        inputs = inputs,
        probL=np.array(sub_data['probL']),
        y = np.array(sub_data['response'])
    )
    
    weights = {'bias' : 1, 'probed_Ch' : 1, 'unprobed_Ch':1}

    return dat, weights


def getData_4(sub_data):

    
    inputs = dict(
                probed_Ch_L = np.array(sub_data["probed_Ch_L"])[:, None],
                probed_Ch_R = np.array(sub_data["probed_Ch_R"])[:, None],
                probe_L = np.array(sub_data["probe_L"])[:, None],
                probe_R = np.array(sub_data["probe_R"])[:, None]
            )                  
                
    dat = dict(
        
        probed_Ch_L = np.array(sub_data["probed_Ch_L"]),
        probed_Ch_R = np.array(sub_data["probed_Ch_R"]),
        probe_L = np.array(sub_data["probe_L"]),
        probe_R = np.array(sub_data["probe_R"]),

        correct=np.array(sub_data['correct']),
        answer=np.array(sub_data['answer']),
        inputs = inputs,
        probL=np.array(sub_data['probL']),
        y = np.array(sub_data['response'])
    )
    
    weights = {'probed_Ch_L' : 1, 'probed_Ch_R' : 1, 'probe_L' : 1, 'probe_R' : 1}

    return dat, weights


def getData_6(sub_data):

    
    inputs = dict(
                probed_Ch_L = np.array(sub_data["probed_Ch_L"])[:, None],
                probed_Ch_R = np.array(sub_data["probed_Ch_R"])[:, None],
                unprobed_Ch_L = np.array(sub_data["unprobed_Ch_L"])[:, None],
                unprobed_Ch_R = np.array(sub_data["unprobed_Ch_R"])[:, None],
                probe_L = np.array(sub_data["probe_L"])[:, None],
                probe_R = np.array(sub_data["probe_R"])[:, None]
            )                  
                
    dat = dict(
        
        probed_Ch_L = np.array(sub_data["probed_Ch_L"]),
        probed_Ch_R = np.array(sub_data["probed_Ch_R"]),
        unprobed_Ch_L = np.array(sub_data["unprobed_Ch_L"]),
        unprobed_Ch_R = np.array(sub_data["unprobed_Ch_R"]),
        probe_L = np.array(sub_data["probe_L"]),
        probe_R = np.array(sub_data["probe_R"]),

        correct=np.array(sub_data['correct']),
        answer=np.array(sub_data['answer']),
        inputs = inputs,
        probL=np.array(sub_data['probL']),
        y = np.array(sub_data['response'])
    )
    
    weights = {'bias' : 0, 'probed_Ch_L' : 1, 'probed_Ch_R' : 1, 'unprobed_Ch_L' : 1, 'unprobed_Ch_R' : 1, 'probe_L' : 1, 'probe_R' : 1}

    return dat, weights

def getData_prevresp(sub_data):

    
    inputs = dict(
                probed_Ch_L = np.array(sub_data["probed_Ch_L"])[:, None],
                probed_Ch_R = np.array(sub_data["probed_Ch_R"])[:, None],
                probe_L = np.array(sub_data["probe_L"])[:, None],
                probe_R = np.array(sub_data["probe_R"])[:, None],
                prev_resp = np.array(sub_data["prev_resp"])[:, None]
            )                  
                
    dat = dict(
        
        probed_Ch_L = np.array(sub_data["probed_Ch_L"]),
        probed_Ch_R = np.array(sub_data["probed_Ch_R"]),
        probe_L = np.array(sub_data["probe_L"]),
        probe_R = np.array(sub_data["probe_R"]),
        prev_resp = np.array(sub_data["prev_resp"]),

        correct=np.array(sub_data['correct']),
        answer=np.array(sub_data['answer']),
        inputs = inputs,
        probL=np.array(sub_data['probL']),
        y = np.array(sub_data['response'])
    )
    
    return dat        
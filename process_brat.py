import pandas as pd
from hpobert.utils import view_all_entities_terminal
import os
import srsly

from os import listdir 
from os.path import isfile, join 
   
src_folder = 'data/hpo_ann/letters' # TODO: make these variables arguments 
des_folder = 'data/jsonl'
label = 'pnt'                       # TODO: generalize this

def process_batch(src_folder, des_folder):
    """Processing an entire input data folder. The function return a `.jsonl` file for each pair of (.ann, .txt) files 
    and a `all.jsonl` file contanining all annotation in the separate jsonl's file

    Args:
        src_folder ([str]): directory of input data files
        des_folder ([str]): directory of input data file
    """
    all_anns = []
    
    # determine documents to be process
    onlyfiles = [f for f in listdir(src_folder) if isfile(join(src_folder, f))]          
    filenames = [file.split('.')[0] for file in onlyfiles] # TODO: hanlde file that are not annotation (hidden files, other folders, missing one of (txt, ann) etc.)
    filenames_unique = set(filenames)   
    
    # process 
    for filename in filenames_unique:
        anns = process_single(filename, src_folder, des_folder)
        all_anns.extend(anns)
        
    # save all anns
    srsly.write_jsonl(os.path.join(des_folder, 'all.jsonl'), all_anns)
        
def process_single(filename, src_folder, des_folder, verbose=False):
    """Process single file. Converting (X.ann, X.txt) to X.jsonl where each line
    follows charracter-offset format which is a dictionary with fields:
        1. "text" field: the targeted text
        2. "spans" field with a list of NER annotations in the form of  {"start": <ch_idx>, "end": <ch_idx>,
        "label": <NER label name>}
    The function saves the resulted jsonl file and return the character-offset dictionary
    
    Args:
        filename ([str]): target file (exclude the extension)
        src_folder ([str]): directory of input data file
        des_folder ([str]): directory of ouput data file

    Returns:
        anns [Dict]: annotation dictionary in character-offset format
    """
    # Read input files
    file_txt = os.path.join(src_folder, filename+'.txt')
    with open(file_txt, 'r') as f: 
        lines = f.readlines()
        
    file_ann = os.path.join(src_folder, filename+'.ann')
    anns = pd.read_csv(file_ann, sep='\t', names=['id', '_type', 'entity'])

    anns['start'] = anns['_type'].apply(lambda x: x.split(' ')[-2])
    anns['end'] = anns['_type'].apply(lambda x: x.split(' ')[-1])
    anns['type'] = anns['_type'].apply(lambda x: x.split(' ')[0])
    anns = anns.drop(columns=['_type'])

    # remove all anns whose type is not 'EyeDisease'
    anns = anns.loc[anns['type'] == 'EyeDiseases'] # TODO: generalize this

    # sort according to start offset
    anns['start'] = anns['start'].astype('int') 
    anns['end'] = anns['end'].astype('int')
    anns_sorted = anns.sort_values(by=['start'], ignore_index=True)

    # Initialize index and output variables  
    N = len(anns)
    start_sent = 0  # start of the sententence in input text file
    end_sent = 0    # end of the sententence in input text file
    i_start = 0
    sents = []
    anns_offset = []
    
    # Processing
    start_ann = anns_sorted['start'][i_start] # initial start_ann
    for line in lines:
        end_sent = start_sent + len(line) - 1
        while start_ann >= start_sent and start_ann <= end_sent:
            sents.append(line)
            
            # Making character offset
            end_ann = anns_sorted['end'][i_start]   # start of entity in brat file
            start = start_ann - start_sent          # start of entity in the sentence
            end = end_ann - start_sent              # end of entity in the sentence
            entity = anns_sorted['entity'][i_start]
            ann_offset = {'text': line, 'spans': [{'start': int(start), 'end': int(end), 'label':label}]} # desired annotation format
            if verbose:
                print(view_all_entities_terminal(ann_offset['text'], ann_offset['spans']), end='')
                print(f"| {i_start} | {start_ann} | {entity}\n")
            anns_offset.append(ann_offset)
            
            # Update
            i_start += 1
            if i_start == N:
                print('End of anns list')
                break
            start_ann = anns_sorted['start'][i_start]

        start_sent =  end_sent + 1
    
    # Merge duplicated offsets
    anns = merge_offsets(anns_offset, True)
    if verbose:
        for i, ann in enumerate(anns):
            print(i)
            print(view_all_entities_terminal(ann['text'], ann['spans']))
    
    # Save to jsonl file
    srsly.write_jsonl(os.path.join(des_folder, filename+'.jsonl'), anns)
    return anns

def merge_offsets(anns, verbose=False):
    """Merge offsets when duplicated

    Args:
        anns (List[Dict]): List of character-offset annotation
    """
    for i, ann in enumerate(anns):
        if i==0: continue
        if ann['text'] == anns[i-1]['text']:
            anns[i-1]['spans'].extend(ann['spans'])
            if verbose:
                print('Detect repeated text at: ', i)
            anns.remove(ann)
    
    return anns
        

if __name__ == '__main__':
    process_batch(src_folder, des_folder)
    print('Done')
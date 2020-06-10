import json
from os.path import basename

with open('./via_latest.json') as f:
    text = f.read()
    Data = json.loads(text)

Project = Data['project']
    
vid_list = Project['vid_list']

print('-->', Data.keys())
print("-->", Project.keys())

# filenames = Data['file'].keys()

video2file_lookup = {}  # (actor, action, sid) -> (vid)
file2video_lookup = {}  # (vid) -> (actor, action, sid)

for vid in vid_list:
    file = basename(Data['file'][vid]['fname'])[4:-13].split('_')
    if len(file) == 3:
        actor, action, subaction = file
        subaction = int(subaction)
        if (actor == 'S11' and action == 'Phoning') or\
            (actor == 'S5' and action == 'Discussion') or\
            (actor == 'S6' and action == 'Waiting'):
            subaction -= 1
        
        video2file_lookup[actor, action, subaction] = vid
        file2video_lookup[vid] = (actor, action, subaction)
        
print('|meta|', len(Data['metadata']))
print('|vids|', len(file2video_lookup))

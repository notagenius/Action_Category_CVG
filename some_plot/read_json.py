import json

with open('via_latest.json') as json_file:
	data = json.load(json_file)
	for i in data['metadata']:
			print data['metadata'][i]['av']['1']
	

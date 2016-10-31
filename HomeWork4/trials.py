

import Data as d
import json
import  time

results =[]
start_time =time.time()
results.append(d.Data(1, 2, 3 ,4, 5.0, time.time() - start_time))
json_string = json.dumps([ob.__dict__ for ob in results])
print json_string

print results[0]
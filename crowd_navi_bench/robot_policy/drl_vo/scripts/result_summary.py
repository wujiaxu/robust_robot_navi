import json
import os 
import numpy as np

def get_navi_time_sr(file):
    # Open and read the JSON file
    with open(file, 'r') as f:
        data = json.load(f)
    nts = np.array(data["navigation_time"])
    se = data["success_episode"]
    return nts[se], len(se)/500.

if __name__ == "__main__":
    root = "/home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/test_log"
    test_logs = os.listdir(root)
    results = {}
    scenes = []
    targets = []
    results_data = {}
    for test_log in test_logs:
        if "backup" in test_log:continue
        # print(test_log.split("_vs_"))
        target  = test_log.split("_vs_")[0]
        if target not in results:
            results[target] ={}
            results_data[target] ={}
            targets.append(target)
        scene = test_log.split("_vs_")[1]
        if scene not in scenes:
            scenes.append(scene)
        log_file = os.path.join(root,test_log,"results.json")
        nts,sr = get_navi_time_sr(log_file)
        results[target][scene] = (sr,np.mean(nts),np.std(nts))
        results_data[target][scene]=nts
    for target in results_data:
        multi = []
        single = []
        all = []
        for scene in results_data[target]:
            if "c0.90" in scene:
                multi.append(results_data[target][scene])
            else:
                single.append(results_data[target][scene])
            all.append(results_data[target][scene])
        multi = np.concatenate(multi)
        single = np.concatenate(single)
        all = np.concatenate(all)

        print(target,"{} {} {} {} {} {}".format(np.mean(multi),np.mean(single),np.mean(all),np.std(multi),np.std(single),np.std(all)))
    with open("result_summary.json","w") as f:
        json.dump(results,f)


    
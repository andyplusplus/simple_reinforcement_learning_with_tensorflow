###################################################
# episode restriction
###################################################
episode_rate = 0.01
def get_episode_restriction(episodes):
    global episode_rate
    eps = int(episode_rate * episodes)
    eps += 1 # to avoid eps is zero
    return eps



###################################################
# get log dir
###################################################
def get_log_dir(log_name, root="tf_logs"):
    log_dir = "{}/{}/".format(root, log_name)
    return log_dir


# several log dirs in the same session
log_dir_index = 0
def get_log_dir_by_time(root="tf_logs"):
    global log_dir_index
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_dir = "{}/run-{}-{}/".format(root, now, log_dir_index)
    log_dir_index += 1
    return log_dir


###################################################
# print list
###################################################
def list_printer(list, items_per_line = 50):
    lines = int(len(list)/items_per_line)
    for i in range(lines):
        print(" ".join(str(list[i*items_per_line + x]) for x in range(items_per_line)))

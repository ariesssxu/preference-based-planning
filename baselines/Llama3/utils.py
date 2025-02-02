import textdistance

def chat(content, messages, role="user"):
    messages.append({"role": role, "content": content})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=30,
        messages=messages
    )

    chat_response = completion
    answer = chat_response.choices[0].message.content
    print(f'ChatGPT: {answer}')
    messages.append({"role": "assistant", "content": answer})
    return answer, messages

def parse_concat(logs, replace=None):
    logs_simplified = []
    pre_line = None
    for line in logs:
        line = line.strip()
        if line == pre_line or len(line) < 3:
            continue
        else:
            logs_simplified.append(line)
            pre_line = line
    if not replace:        
        return '\n'.join(logs_simplified)
    else:
        return replace.join(logs_simplified)
    
def parse_action(logs, replace=None):
    logs_simplified = []
    pre_line = None
    for line in logs:
        line = line.strip()
        if line == pre_line or len(line) < 3:
            continue
        else:
            logs_simplified.append(line)
            pre_line = line
    if not replace:        
        return '\n'.join(logs_simplified)
    else:
        return replace.join(logs_simplified)

def get_same_demo(gt, test_files):
    # as we have shuffled the test files, just return the first one
    for test_file in test_files:
        if test_file["preference"] == gt:
            with open(test_file["text"], 'r') as f:
                text_log = f.readlines()
            return text_log, test_file["preference"]

def compare(answer, gt, in_sequence):
    answer = answer.lower()
    gt = gt.lower()
    # for keyword in gt.split(" ")[:]:
    #     if keyword.lower() not in answer:
    #         false += 1
    #         print(f"False: {answer} vs {gt}")
    #         break
    for keyword in gt.split(" ")[:]:
        if keyword.lower() not in answer:
            print(f"False: {answer} vs {gt}")
            return False
    # if in_sequence:
    #     # check if the words occur in the same order
    #     gt_words = gt.split(" ")
    #     answer_words = answer.split(" ")
    #     idx = []
    #     for word in gt_words:
    #         idx.append(answer_words.index(word))
    #     if idx != sorted(idx):
    #         return False
    return True

def compute_levenshtein_distance(answer, gt):
    answer = answer.lower().split(",")
    answer = [action.strip().split(" ")[0] for action in answer]
    gt = gt.lower().split(",")
    gt = [action.strip().split(" ")[0] for action in gt]
    print(answer, gt)
    # all to index
    all_actions = list(set(answer + gt))
    action2index = {}
    for i, action in enumerate(all_actions):
        action2index[action] = i
    answer = [action2index[action] for action in answer]
    gt = [action2index[action] for action in gt]

    print(answer, gt)

    return textdistance.levenshtein.distance(answer, gt)
    
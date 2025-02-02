from tokencost import calculate_prompt_cost, calculate_completion_cost
import textdistance


def chat(client, model, content, messages, role="user"):
    messages.append({"role": role, "content": content})
    prompt_cost = calculate_prompt_cost(messages, "gpt-4-turbo")
    completion = client.chat.completions.create(
        model=model,
        max_tokens=30,
        messages=messages
    )
    completion_cost = calculate_completion_cost(completion, "gpt-4-turbo")
    chat_response = completion
    answer = chat_response.choices[0].message.content
    print(f'ChatGPT: {answer}')
    messages.append({"role": "assistant", "content": answer})
    return answer, messages, prompt_cost + completion_cost


def chat_vision(client, model, content, messages, role="user"):
    # prompt_cost = calculate_prompt_cost(messages, "gpt-4-vision-preview")
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=30
    )
    answer = completion.choices[0].message.content
    # completion_cost = calculate_completion_cost(completion, "gpt-4-vision-preview")
    print(f'ChatGPT-4v: {answer}')
    messages.append({"role": "assistant", "content": answer})
    return answer, messages, 0


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


def get_same_demo_text(gt, test_files):
    # as we have shuffled the test files, just return the first one
    for test_file in test_files:
        if test_file["preference"] == gt:
            with open(test_file["text"], 'r') as f:
                text_log = f.readlines()
            return text_log, test_file["preference"]


def get_same_demo_video(gt, test_files):
    # as we have shuffled the test files, just return the first one
    for test_file in test_files:
        if test_file["preference"] == gt:
            return test_file["camera"], test_file["preference"]


def get_same_demo_video_text(gt, test_files):
    # as we have shuffled the test files, just return the first one
    for test_file in test_files:
        if test_file["preference"] == gt:
            return test_file["camera"], test_file["preference"], test_file["text"]


def compare(answer, gt, in_sequence):
    answer = answer.lower().split(" ")
    answer = [word if len(word) and word[-1] != "." else word[:-1]
              for word in answer]
    gt = gt.lower()
    for keyword in gt.split(" ")[:]:
        if keyword.lower() not in answer:
            print(f"False: {answer} vs {gt}")
            return False
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

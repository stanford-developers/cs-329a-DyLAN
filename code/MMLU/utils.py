import json
import os
import re
import time
import pandas as pd
from prompt_lib import MMLU_QUESTION, COMPLEX_COT_EXAMPLES, TEMPERATURE, MAX_TOKENS
from together import Together
import backoff
from together.error import RateLimitError, APIError


class OutOfQuotaException(Exception):
    "Raised when the key exceeded the current quota"

    def __init__(self, key, cause=None):
        super().__init__(f"No quota for key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()


class AccessTerminatedException(Exception):
    "Raised when the key has been terminated"

    def __init__(self, key, cause=None):
        super().__init__(f"Access terminated key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()


# Initialize Together client
_together_client = None


def get_together_client():
    global _together_client
    if _together_client is None:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set")
        _together_client = Together(api_key=api_key)
    return _together_client


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) == 0:
                continue
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) >= 2
        return splits[0]
    else:
        return string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def parse_question_answer(df, ix):
    question = df.iloc[ix, 0]
    a = df.iloc[ix, 1]
    b = df.iloc[ix, 2]
    c = df.iloc[ix, 3]
    d = df.iloc[ix, 4]

    question = MMLU_QUESTION.format(question, a, b, c, d)

    answer = df.iloc[ix, 5]

    return question, answer


def get_mmlu_qa_pairs(csv_name):
    df = pd.read_csv(csv_name, header=None)
    ix = len(df)
    return [parse_question_answer(df, idx) for idx in range(ix)]


def get_math_qa_pairs(sub_dir, min_file, max_file):
    def find_math_answer(s):
        assert ('boxed' in s)
        # s = s.replace(",", "")
        ans = s.split('boxed')[-1]
        if (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        return a

    # in code/MMLU/utils.py

    from pathlib import Path
    import json

    def parse_single_qa_math(subdir, file_id):
        """
        Load a single MATH problem from JSON.

        Supports both:
          - original MATH-style JSON: { "problem", "solution", "level", "type" }
          - our csv->json style:      { "question", "answer", "level", "type" }
        """
        # file_id may be an int (1) or a string ("0001")
        if isinstance(file_id, int):
            fname = f"{file_id:04d}.json"
        else:
            # assume it's already the basename without extension (e.g. "0001")
            fname = f"{str(file_id).zfill(4)}.json"

        path = Path(subdir) / fname
        with path.open("r", encoding="utf-8") as f:
            problem_data = json.load(f)

        # Question text
        if "problem" in problem_data:
            question = problem_data["problem"]
        elif "question" in problem_data:
            question = problem_data["question"]
        else:
            raise KeyError(f"{path}: missing 'problem' / 'question' key")

        # Gold answer
        if "solution" in problem_data:
            answer = problem_data["solution"]
        elif "answer" in problem_data:
            answer = problem_data["answer"]
        else:
            raise KeyError(f"{path}: missing 'solution' / 'answer' key")

        # Level and type (these exist in your JSON already)
        level = problem_data.get("level", "")
        typ = problem_data.get("type", "")

        return question, level, typ, answer

    ret_list = []
    for subdir, dirs, files in os.walk(sub_dir):
        for file in files:
            file_num = int(os.path.splitext(file)[0])  # "0009.json" -> 9
            if min_file <= file_num <= max_file:
                # pass the numeric id so parse_single_qa_math builds "0009.json"
                question, prob_level, prob_type, answer = parse_single_qa_math(subdir, file_num)
            else:
                continue
            ret_list.append((question, answer))
    return ret_list


def is_equiv(str1, str2, verbose: bool = False) -> bool:
    """
    Task‑agnostic equivalence check used for both MMLU and MATH.

    For math‑style outputs, we first try to extract a canonical short
    answer from each string (using extract_math_answer). If both sides
    yield non‑empty answers, we compare those. Otherwise we fall back
    to normalized string equality.
    """
    if str1 is None or str2 is None:
        return False

    s1 = str(str1)
    s2 = str(str2)

    # --- 1) Try math-style answer extraction on both sides ---
    try:
        a1 = extract_math_answer(s1)
        a2 = extract_math_answer(s2)
        if verbose:
            print(f"[is_equiv] a1={a1!r}, a2={a2!r}")
        # If we got something non‑empty from both, trust that.
        if a1 and a2:
            return a1 == a2
    except Exception as e:
        if verbose:
            print(f"[is_equiv] extract_math_answer failed: {e!r}")

    # --- 2) Fallback: normalized raw‑string equality (MMLU, etc.) ---
    try:
        n1 = _strip_string(s1)
        n2 = _strip_string(s2)
        return n1 == n2
    except Exception:
        return s1.strip() == s2.strip()



def extract_math_answer(pred_str):
    if ('The answer is ' in pred_str):
        pred = pred_str.split('The answer is ')[-1].strip()
    elif ('the answer is ' in pred_str):
        pred = pred_str.split('the answer is ')[-1].strip()
    elif 'boxed' in pred_str:
        ans = pred_str.split('boxed')[-1]
        if len(ans) == 0:
            print(pred_str)
        if (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        pred = a

    else:
        pattern = '-?\d*\.?\d+'
        pred = re.findall(pattern, pred_str)
        if (len(pred) >= 1):
            # print(pred_str)
            pred = pred[-1]
        else:
            pred = ''
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred[-1] == "/":
            pred = pred[:-1]
    pred = _strip_string(pred)
    if 'boxed' in pred:
        ans = pred.split('boxed')[-1]
        if (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        pred = a
    return pred


@backoff.on_exception(backoff.expo, (RateLimitError, APIError), max_tries=5)
def generate_answer(answer_context, model):
    print("question context: ")
    print(answer_context)
    client = get_together_client()
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=answer_context,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            n=1
        )
    except RateLimitError as e:
        error_message = str(e)
        if "quota" in error_message.lower():
            raise OutOfQuotaException(os.getenv("TOGETHER_API_KEY"))
        elif "terminated" in error_message.lower() or "violation" in error_message.lower():
            raise AccessTerminatedException(os.getenv("TOGETHER_API_KEY"))
        else:
            raise e

    return completion.choices[0].message.content, completion.usage.prompt_tokens, completion.usage.completion_tokens


def parse_single_choice(reply):
    pattern = r'\(([ABCDabcd])\)'
    matches = re.findall(pattern, reply)

    solution = None
    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            break

    if solution is None:
        alter_pattern = r'([ABCDabcd])\)'
        alter_matches = re.findall(alter_pattern, reply)
        for match_str in alter_matches[::-1]:
            solution = match_str.upper()
            if solution:
                break

    return solution


def most_frequent(clist, cmp_func):
    counter = 0
    num = clist[0]

    for i in clist:
        current_frequency = sum(cmp_func(i, item) for item in clist)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num, counter


# ---- Soft tie-break judge helpers (k-way) ----
import json, re, random


def _extract_json_list(text):
    """
    Extract the first JSON-like list '[ ... ]' from text and parse it.
    Returns a Python list or None on failure.
    """
    if not isinstance(text, str):
        return None
    # strip code fences if any
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return None
    frag = m.group(0)
    try:
        data = json.loads(frag)
        if isinstance(data, list):
            return data
    except Exception:
        return None
    return None


def _sanitize_and_normalize(weights, k):
    """
    Convert to floats, clamp negatives to zero, and renormalize to sum 1.
    If anything goes wrong (size mismatch or all zeros), return uniform.
    """
    try:
        vals = [float(x) for x in weights]
    except Exception:
        return [1.0 / k] * k

    if len(vals) != k:
        return [1.0 / k] * k

    vals = [0.0 if (v is None or not (v == v)) else float(v) for v in vals]  # NaN->0
    vals = [max(0.0, v) for v in vals]
    s = sum(vals)
    if s <= 0.0:
        return [1.0 / k] * k
    return [v / s for v in vals]


def judge_importance_weights(responses, question, qtype, model):
    """
    Bias-reduced k-way judge:
      * Shuffle candidate order before prompting.
      * Ask LLM for a JSON weight vector [w1..wk] (sum to 1).
      * Map weights back to the original order.
    Returns (weights_in_original_order, prompt_tokens, completion_tokens).
    """
    assert isinstance(responses, list) and len(responses) >= 2
    k = len(responses)

    # 1) Shuffle to reduce position bias
    perm = list(range(k))
    random.shuffle(perm)
    shuffled = [responses[i] for i in perm]

    # 2) Build messages (JSON-only)
    from prompt_lib import construct_weight_judge_message
    messages = construct_weight_judge_message(shuffled, question, qtype)

    # 3) Call the same LLM backend as other calls (temperature 0)
    reply, ptok, ctok = generate_answer(messages, model)

    # 4) Parse and sanitize
    parsed = _extract_json_list(reply)
    if parsed is None:
        norm = [1.0 / k] * k
    else:
        norm = _sanitize_and_normalize(parsed, k)

    # 5) Map weights back to original order
    #    norm is aligned with 'shuffled' order; invert permutation
    inv = [0] * k
    for pos, orig in enumerate(perm):
        inv[orig] = pos
    restored = [0.0] * k
    for orig_idx in range(k):
        restored[orig_idx] = norm[inv[orig_idx]]

    return restored, ptok, ctok

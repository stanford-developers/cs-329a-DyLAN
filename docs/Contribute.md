# Contribute.md

## Environment Setup:

1. Assume you have Python 3.10

```shell
# From the repository root
python3.10 --version   # should print Python 3.10.x
python3.10 -m venv venv

# Activate the environment
source venv/bin/activate

# (optional but recommended) upgrade pip
python -m pip install -U pip
```

2. Follow README.md to install dependencies.

```shell
cd code
pip install -r requirements.txt
```

3. PyCharm: point the Project Interpreter at the `venv`

If you use PyCharm, configure it to use the virtualenv you created above.

**a. Open the project**

- `File → Open…` and select the repository root (the folder that contains `venv/`).

**b. Open Interpreter settings**

- **Windows/Linux:** `File → Settings…`
- **macOS:** `PyCharm → Preferences…`
- Then go to `Project: <your project> → Python Interpreter`.

**c. Select the `venv`**

- If PyCharm already lists the `venv` interpreter, select it and click **OK/Apply**.
- Otherwise, click **Add Interpreter** (or the ⚙️ gear icon → **Add…**) → **Existing environment** and browse to:
    - **macOS/Linux:** `<repo>/venv/bin/python`
    - **Windows:** `<repo>\venv\Scripts\python.exe`
- Click **OK**, then **Apply**.

4. Create a .env file using .env.example and add your API keys. DO NOT COMMIT YOUR API KEYS.
5. Test API key in Python Shell:

```python 
import os, openai
from dotenv import load_dotenv;

load_dotenv()
openai.api_base = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
openai.api_key = os.getenv("TOGETHER_API_KEY")
r = openai.ChatCompletion.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    temperature=0
)
print(r["choices"][0]["message"]["content"])
```

You should get something like this:

```
>>> r = openai.ChatCompletion.create(
...     model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
...     messages=[{"role": "user", "content": "Reply with exactly: OK"}],
...     temperature=0
... )
>>> print(r["choices"][0]["message"]["content"])
OK
>>> print(r)
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "OK",
        "role": "assistant",
        "tool_calls": []
      },
      "seed": 16791737139537046000
    }
  ],
  "created": 1760223489,
  "id": "oEovV3E-4Yz4kd-98d1f5e4eccf5580",
  "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
  "object": "chat.completion",
  "prompt": [],
  "usage": {
    "cached_tokens": 0,
    "completion_tokens": 2,
    "prompt_tokens": 40,
    "total_tokens": 42
  }
}
>>> 
```

6. 
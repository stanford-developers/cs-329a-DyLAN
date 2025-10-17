import ast
import json
import os
import random
import sys
from prettytable import PrettyTable
from LLMLP import LLMLP
from utils import *
from utils import setup_logging, InferenceTimer

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Put your query here
QUERY = r"""What 8 letter word can have a letter taken away and it still makes a word. Take another letter away and it still makes a word. Keep on doing that until you have one letter left. What is the word?"""

EXP_NAME = "trial_1"
# MODEL = "chatgpt0301"
# Use a cheaper Together model for debugging.
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

ACTIVATION = "listwise"
TYPE = "open-ended"
DIR_NAME = "trial"
USE_CONCURRENT = True  # Enable concurrent processing
MAX_WORKERS = 4  # Number of concurrent worker threads
ENABLE_LOGGING = True  # Enable detailed logging to file
LOG_FILE = None  # If None, use timestamp-based filename

# Here are the roles of the participants in the LLM-agent collaboration
# See prompt_lib.ROLE_MAP for the full list of roles
ROLES = ["Assistant", "Assistant", "Assistant", "Assistant"]

def set_rd_seed(seed):
    random.seed(seed)

def main():
    set_rd_seed(0)
    assert len(ROLES) > 0

    # Setup logging if enabled
    logger = None
    if ENABLE_LOGGING:
        logger = setup_logging(LOG_FILE)
        logger.info("DyLAN Inference Session Started")
        logger.info(f"Query: {QUERY}")
        logger.info(f"Model: {MODEL}")
        logger.info(f"Configuration: {len(ROLES)} agents, {ACTIVATION} activation, {TYPE} type")

    # Initialize LLMLP with logger
    llmlp = LLMLP(MODEL, len(ROLES), ROLES, 3, ACTIVATION, TYPE, MODEL, 
                  use_concurrent=USE_CONCURRENT, max_workers=MAX_WORKERS, logger=logger)

    llmlp.zero_grad()
    
    # Use InferenceTimer to track total execution time
    with InferenceTimer(logger, "DyLAN Inference Process"):
        res, resp_cnt, completions, prompt_tokens, completion_tokens = llmlp.forward(QUERY)
        imp_score = llmlp.backward(res)
        imp_score = [[imp_score[idx] for idx in range(len(ROLES)*rid, len(ROLES)*(rid+1))] for rid in range(3)]

    # Create results table
    pt = PrettyTable()
    pt.add_column("Round", ROLES)
    for rid in range(3):
        responses = [(completions[idx][rid] if completions[idx][rid] is not None else "No response.") for idx in range(len(ROLES))]
        pt.add_column(str(rid+1), responses, "l")

    # Print results
    print(r"Query: {}".format(QUERY))
    print(r"#API calls: {}".format(resp_cnt))
    print(r"Prompt Tokens: {}".format(prompt_tokens))
    print(r"Completion Tokens: {}".format(completion_tokens))
    print(pt)
    print(r"Final Answer: {}".format(res))
    print()
    print(r"Agent Importance Scores: {}".format([sum(imp_score[rid][idx] for rid in range(3)) for idx in range(len(ROLES))]))
    
    # Log final results
    if logger:
        logger.info("=== FINAL RESULTS ===")
        logger.info(f"Query: {QUERY}")
        logger.info(f"Final Answer: {res}")
        logger.info(f"Total API calls: {resp_cnt}")
        logger.info(f"Prompt Tokens: {prompt_tokens}")
        logger.info(f"Completion Tokens: {completion_tokens}")
        logger.info(f"Agent Importance Scores: {[sum(imp_score[rid][idx] for rid in range(3)) for idx in range(len(ROLES))]}")
        logger.info("DyLAN Inference Session Completed")


if __name__ == "__main__":
    main()

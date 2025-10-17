import math
import random
import logging
from LLM_Neuron import LLMNeuron, LLMEdge, listwise_ranker_2
from utils import parse_single_choice, most_frequent, is_equiv, extract_math_answer, create_concurrent_processor
from sacrebleu import sentence_bleu
from prompt_lib import GEN_THRESHOLD


ACTIVATION_MAP = {'listwise': 0, 'trueskill': 1, 'window': 2, 'none': -1} # TODO: only 0 is implemented

class LLMLP:
    
    def __init__(self, default_model_name, agents=4, agent_roles=[],
                 rounds=2, activation="listwise", qtype="single_choice", mtype="gpt-3.5-turbo",
                 use_concurrent=False, max_workers=8, logger=None):
        self.default_model_name = default_model_name
        self.agents = agents
        self.rounds = rounds
        self.activation = ACTIVATION_MAP[activation]
        self.mtype = mtype
        self.use_concurrent = use_concurrent
        self.max_workers = max_workers
        self.logger = logger
        
        # If concurrent processing is enabled, create concurrent processor
        if self.use_concurrent:
            self.concurrent_processor = create_concurrent_processor(self.mtype, self.max_workers)
        
        assert len(agent_roles) == agents and agents > 0
        self.agent_roles = agent_roles
        self.qtype = qtype
        if qtype == "single_choice":
            self.cmp_res = lambda x, y: x == y
            self.ans_parser = parse_single_choice
        elif qtype == "math_exp":
            self.cmp_res = is_equiv
            self.ans_parser = extract_math_answer
        elif qtype == "open-ended":
            self.cmp_res = lambda x, y: sentence_bleu(x, [y], lowercase=True).score >= GEN_THRESHOLD * 100
            self.ans_parser = lambda x: x
        else:
            raise NotImplementedError("Error qtype")

        self.init_nn(self.activation, self.agent_roles)

    def init_nn(self, activation, agent_roles):
        self.nodes, self.edges = [], []
        for idx in range(self.agents):
            self.nodes.append(LLMNeuron(agent_roles[idx], self.mtype, self.ans_parser, self.qtype))
        
        agents_last_round = self.nodes[:self.agents]
        for rid in range(1, self.rounds):
            for idx in range(self.agents):
                self.nodes.append(LLMNeuron(agent_roles[idx], self.mtype, self.ans_parser, self.qtype))
                for a1 in agents_last_round:
                    self.edges.append(LLMEdge(a1, self.nodes[-1]))
            agents_last_round = self.nodes[-self.agents:]

        if activation == 0:
            self.activation = listwise_ranker_2
            self.activation_cost = 1
        else:
            raise NotImplementedError("Error init activation func")
    
    def zero_grad(self):
        for edge in self.edges:
            edge.zero_weight()

    def check_consensus(self, idxs, idx_mask):
        # check consensus based on idxs (range) and idx_mask (actual members, might exceed the range)
        candidates = [self.nodes[idx].get_answer() for idx in idxs]
        consensus_answer, ca_cnt = most_frequent(candidates, self.cmp_res)
        if ca_cnt > math.floor(2/3 * len(idx_mask)):
            return True, consensus_answer
        return False, None

    def set_allnodes_deactivated(self):
        for node in self.nodes:
            node.deactivate()

    def forward(self, question):
        def get_completions():
            # get completions
            completions = [[] for _ in range(self.agents)]
            for rid in range(self.rounds):
                for idx in range(self.agents*rid, self.agents*(rid+1)):
                    if self.nodes[idx].active:
                        completions[idx % self.agents].append(self.nodes[idx].get_reply())
                    else:
                        completions[idx % self.agents].append(None)
            return completions

        resp_cnt = 0
        total_prompt_tokens, total_completion_tokens = 0, 0
        self.set_allnodes_deactivated()
        assert self.rounds > 2
        
        # Log the start of inference
        if self.logger:
            self.logger.info(f"Starting inference for question: {question}")
            self.logger.info(f"Configuration: {self.agents} agents, {self.rounds} rounds, {self.qtype} type")
            self.logger.info(f"Agent roles: {self.agent_roles}")
            self.logger.info("-" * 50)

        # shuffle the order of agents
        loop_indices = list(range(self.agents))
        random.shuffle(loop_indices)

        activated_indices = []
        if self.logger:
            self.logger.info(f"Round 1: Activating agents in order: {loop_indices}")
        
        for idx, node_idx in enumerate(loop_indices):
            if self.logger:
                self.logger.info(f"Activating agent {node_idx} (role: {self.agent_roles[node_idx % len(self.agent_roles)]})")
            
            self.nodes[node_idx].activate(question)
            resp_cnt += 1
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)
            
            if self.logger:
                self.logger.info(f"Agent {node_idx} response: {self.nodes[node_idx].get_answer()}")
                self.logger.info(f"Tokens used: prompt={self.nodes[node_idx].prompt_tokens}, completion={self.nodes[node_idx].completion_tokens}")
        
            if idx >= math.floor(2/3 * self.agents):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents)))
                if self.logger:
                    self.logger.info(f"Consensus check: reached={reached}, reply={reply}")
                if reached:
                    if self.logger:
                        self.logger.info(f"Inference completed early with consensus. Total API calls: {resp_cnt}")
                    return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

        loop_indices = list(range(self.agents, self.agents*2))
        random.shuffle(loop_indices)

        activated_indices = []
        if self.logger:
            self.logger.info(f"Round 2: Activating agents in order: {loop_indices}")
        
        for idx, node_idx in enumerate(loop_indices):
            if self.logger:
                self.logger.info(f"Activating agent {node_idx} (role: {self.agent_roles[node_idx % len(self.agent_roles)]})")
            
            self.nodes[node_idx].activate(question)
            resp_cnt += 1
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)
            
            if self.logger:
                self.logger.info(f"Agent {node_idx} response: {self.nodes[node_idx].get_answer()}")
                self.logger.info(f"Tokens used: prompt={self.nodes[node_idx].prompt_tokens}, completion={self.nodes[node_idx].completion_tokens}")
        
            if idx >= math.floor(2/3 * self.agents):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents)))
                if self.logger:
                    self.logger.info(f"Consensus check: reached={reached}, reply={reply}")
                if reached:
                    if self.logger:
                        self.logger.info(f"Inference completed early with consensus. Total API calls: {resp_cnt}")
                    return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

        idx_mask = list(range(self.agents))
        idxs = list(range(self.agents, self.agents*2))
        for rid in range(2, self.rounds):
            if self.logger:
                self.logger.info(f"Round {rid+1}: Starting activation process")
            
            # TODO: compatible with 1/2 agents
            if self.agents > 3:
                replies = [self.nodes[idx].get_reply() for idx in idxs]
                indices = list(range(len(replies)))
                random.shuffle(indices)
                shuffled_replies = [replies[idx] for idx in indices]
            
                if self.logger:
                    self.logger.info(f"Running listwise ranking on {len(shuffled_replies)} responses")
                
                tops, prompt_tokens, completion_tokens = self.activation(shuffled_replies, question, self.qtype, self.mtype)
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                idx_mask = list(map(lambda x: idxs[indices[x]] % self.agents, tops))
                resp_cnt += self.activation_cost
                
                if self.logger:
                    self.logger.info(f"Ranking result: selected agents {tops}, tokens used: prompt={prompt_tokens}, completion={completion_tokens}")

            loop_indices = list(range(self.agents*rid, self.agents*(rid+1)))
            random.shuffle(loop_indices)
            idxs = []
            
            if self.logger:
                self.logger.info(f"Round {rid+1}: Activating agents in order: {loop_indices}")
                self.logger.info(f"Selected agents for this round: {idx_mask}")
            
            for idx, node_idx in enumerate(loop_indices):
                if idx in idx_mask:
                    if self.logger:
                        self.logger.info(f"Activating agent {node_idx} (role: {self.agent_roles[node_idx % len(self.agent_roles)]})")
                    
                    self.nodes[node_idx].activate(question)
                    resp_cnt += 1
                    total_prompt_tokens += self.nodes[node_idx].prompt_tokens
                    total_completion_tokens += self.nodes[node_idx].completion_tokens
                    idxs.append(node_idx)
                    
                    if self.logger:
                        self.logger.info(f"Agent {node_idx} response: {self.nodes[node_idx].get_answer()}")
                        self.logger.info(f"Tokens used: prompt={self.nodes[node_idx].prompt_tokens}, completion={self.nodes[node_idx].completion_tokens}")
                    
                    if len(idxs) > math.floor(2/3 * len(idx_mask)):
                        reached, reply = self.check_consensus(idxs, idx_mask)
                        if self.logger:
                            self.logger.info(f"Consensus check: reached={reached}, reply={reply}")
                        if reached:
                            if self.logger:
                                self.logger.info(f"Inference completed early with consensus. Total API calls: {resp_cnt}")
                            return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

        completions = get_completions()
        final_answer = most_frequent([self.nodes[idx].get_answer() for idx in idxs], self.cmp_res)[0]
        
        if self.logger:
            self.logger.info(f"Inference completed. Final answer: {final_answer}")
            self.logger.info(f"Total API calls: {resp_cnt}")
            self.logger.info(f"Total tokens: prompt={total_prompt_tokens}, completion={total_completion_tokens}")
            self.logger.info("=" * 50)
        
        return final_answer, resp_cnt, completions, total_prompt_tokens, total_completion_tokens


    def backward(self, result):
        flag_last = False
        for rid in range(self.rounds-1, -1, -1):
            if not flag_last:
                if len([idx for idx in range(self.agents*rid, self.agents*(rid+1)) if self.nodes[idx].active]) > 0:
                    flag_last = True
                else:
                    continue

                ave_w = 1 / len([idx for idx in range(self.agents*rid, self.agents*(rid+1)) if self.nodes[idx].active and self.cmp_res(self.nodes[idx].get_answer(), result)])
                for idx in range(self.agents*rid, self.agents*(rid+1)):
                    if self.nodes[idx].active and self.cmp_res(self.nodes[idx].get_answer(), result):
                        self.nodes[idx].importance = ave_w
                    else:
                        self.nodes[idx].importance = 0
            else:
                for idx in range(self.agents*rid, self.agents*(rid+1)):
                    self.nodes[idx].importance = 0
                    if self.nodes[idx].active:
                        for edge in self.nodes[idx].to_edges:
                            self.nodes[idx].importance += edge.weight * edge.a2.importance

        return [node.importance for node in self.nodes]

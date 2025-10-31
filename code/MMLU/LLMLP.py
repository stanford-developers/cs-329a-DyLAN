import os
import math
import random
from LLM_Neuron import LLMNeuron, LLMEdge, listwise_ranker_2
from utils import parse_single_choice, most_frequent, is_equiv, extract_math_answer, judge_tie_break_weights



ACTIVATION_MAP = {'listwise': 0, 'trueskill': 1, 'window': 2, 'none': -1} # TODO: only 0 is implemented

class LLMLP:
    
    def __init__(self, default_model_name, agents=4, agent_roles=[],
                 rounds=2, activation="listwise", qtype="single_choice", mtype="gpt-3.5-turbo"):
        self.default_model_name = default_model_name
        self.agents = agents
        self.rounds = rounds
        self.activation = ACTIVATION_MAP[activation]
        self.mtype = mtype
        
        assert len(agent_roles) == agents and agents > 0
        self.agent_roles = agent_roles
        self.qtype = qtype
        if qtype == "single_choice":
            self.cmp_res = lambda x, y: x == y
            self.ans_parser = parse_single_choice
        elif qtype == "math_exp":
            self.cmp_res = is_equiv
            self.ans_parser = extract_math_answer

        # --- optional soft tie-break via LLM judge ---
        self.tie_break_judge = bool(int(os.environ.get("TIE_BREAK_JUDGE", "0")))
        # Which model to use for the judge; default to the main model type
        self.tie_break_model = os.environ.get("TIE_BREAK_MODEL", self.mtype)

        # cache the last question so backward() can access it for the judge
        self._last_question = None

        self.init_nn(self.activation, self.agent_roles)

    def init_nn(self, activation, agent_roles):
        self.nodes, self.edges = [], []
        for idx in range(self.agents):
            self.nodes.append(LLMNeuron(agent_roles[idx], self.mtype, self.ans_parser, self.qtype))
        
        agents_last_round = self.nodes[:self.agents]
        for rid in range(1, self.rounds):
            for idx in range(self.agents):
                self.nodes.append(LLMNeuron(agent_roles[idx], self.mtype, self.ans_parser, self.qtype))
                # print(len(agents_last_round)) !!!
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
            print("Consensus answer: {}".format(consensus_answer))
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
        # Remember the raw question text for the backward (tie-break) step.
        assert self.rounds > 2
        self._last_question = question
        # question = format_question(question, self.qtype)

        # shuffle the order of agents
        loop_indices = list(range(self.agents))
        random.shuffle(loop_indices)

        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            print(0, idx)
            self.nodes[node_idx].activate(question)
            resp_cnt += 1
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)
        
            if idx >= math.floor(2/3 * self.agents):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents)))
                if reached:
                    return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

        loop_indices = list(range(self.agents, self.agents*2))
        random.shuffle(loop_indices)

        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            print(1, idx)
            self.nodes[node_idx].activate(question)
            resp_cnt += 1
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)
        
            if idx >= math.floor(2/3 * self.agents):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents)))
                if reached:
                    return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

        idx_mask = list(range(self.agents))
        idxs = list(range(self.agents, self.agents*2))
        for rid in range(2, self.rounds):
            # TODO: compatible with 1/2 agents
            if self.agents > 3:
                replies = [self.nodes[idx].get_reply() for idx in idxs]
                indices = list(range(len(replies)))
                random.shuffle(indices)
                shuffled_replies = [replies[idx] for idx in indices]
            
                tops, prompt_tokens, completion_tokens = self.activation(shuffled_replies, question, self.qtype, self.mtype)
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                idx_mask = list(map(lambda x: idxs[indices[x]] % self.agents, tops))
                resp_cnt += self.activation_cost

            loop_indices = list(range(self.agents*rid, self.agents*(rid+1)))
            random.shuffle(loop_indices)
            idxs = []
            for idx, node_idx in enumerate(loop_indices):
                # TODO: report bug # if idx in idx_mask:
                if node_idx % self.agents in idx_mask:
                    print(rid, idx)
                    self.nodes[node_idx].activate(question)
                    resp_cnt += 1
                    total_prompt_tokens += self.nodes[node_idx].prompt_tokens
                    total_completion_tokens += self.nodes[node_idx].completion_tokens
                    idxs.append(node_idx)
                    if len(idxs) > math.floor(2/3 * len(idx_mask)):
                        reached, reply = self.check_consensus(idxs, idx_mask)
                        if reached:
                            return reply, resp_cnt, get_completions(), total_prompt_tokens, total_completion_tokens

        completions = get_completions()
        return most_frequent([self.nodes[idx].get_answer() for idx in idxs], self.cmp_res)[0], resp_cnt, completions, total_prompt_tokens, total_completion_tokens

    def backward(self, result):
        """
        Backward importance aggregation.
        If enabled (TIE_BREAK_JUDGE=1), the last active round with >=2 correct agents
        calls an LLM judge to return soft weights [wA, wB] that sum to 1, rather than
        splitting equally. Earlier layers aggregate via rated edges as before.
        """
        flag_last = False

        for rid in range(self.rounds - 1, -1, -1):
            layer_start = self.agents * rid
            layer_end = self.agents * (rid + 1)
            active_idxs = [idx for idx in range(layer_start, layer_end) if self.nodes[idx].active]

            if not flag_last:
                # Find the *last* layer that actually fired
                if len(active_idxs) == 0:
                    continue
                flag_last = True

                # Who got the final answer correct in this last active layer?
                correct_active = [
                    idx for idx in active_idxs
                    if self.cmp_res(self.nodes[idx].get_answer(), result)
                ]

                if len(correct_active) == 0:
                    # should not happen in normal flow; just zero the layer
                    for idx in range(layer_start, layer_end):
                        self.nodes[idx].importance = 0.0
                    continue

                # Optional: ask LLM judge to soft‑split credit between the two survivors.
                # (By design, from round 3 onward the ranker keeps top-2, but guard anyway.)
                if self.tie_break_judge and len(correct_active) >= 2 and self._last_question is not None:
                    pair = correct_active[:2]  # only judge top two
                    pair_responses = [self.nodes[i].get_reply() for i in pair]
                    weights = judge_tie_break_weights(
                        pair_responses,
                        question=self._last_question,
                        qtype=self.qtype,
                        model_name=self.tie_break_model,
                    )
                    # initialize last-layer importances
                    for idx in range(layer_start, layer_end):
                        self.nodes[idx].importance = 0.0
                    # assign weights to the two judged winners
                    for w, i in zip(weights, pair):
                        self.nodes[i].importance = float(w)
                else:
                    # Equal split among all correct actives (original behavior)
                    denom = len(correct_active)
                    ave_w = 1.0 / denom if denom > 0 else 0.0
                    for idx in range(layer_start, layer_end):
                        if self.nodes[idx].active and self.cmp_res(self.nodes[idx].get_answer(), result):
                            self.nodes[idx].importance = ave_w
                        else:
                            self.nodes[idx].importance = 0.0

            else:
                # Standard backward aggregation through rated edges
                for idx in range(layer_start, layer_end):
                    self.nodes[idx].importance = 0.0
                    if self.nodes[idx].active:
                        for edge in self.nodes[idx].to_edges:
                            # edge.a2 is the successor node
                            self.nodes[idx].importance += edge.weight * edge.a2.importance

        return [node.importance for node in self.nodes]


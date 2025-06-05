# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/MARIO-Math-Reasoning/Super_MARIO
from __future__ import annotations
import os
import json
import os.path as osp
from tqdm import tqdm
from termcolor import colored
from functools import partial
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from pebble import ProcessPool
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Any, Dict, List, Callable, Type, Tuple
from pydantic import BaseModel, ConfigDict, field_validator
from .agents.tree import BaseTree
from .agents.mcts import MCTS
from .llms.llms import llm_generate, rm_generate
from .llms.llm_engine import llm_engine, rm_engine
from .constants import TIMEOUT_SECONDS, ERROR_COLOR


class Solver(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: Any
    stop: List[str] = None
    llm: Optional[Callable[[...], List[str]]] = None
    llm_engine: Optional[LLM] = None
    generate_sampling_params: Optional[SamplingParams] = None
    need_value_func: bool = False
    max_agent_steps: int = 1
    reward_model: Optional[Any] = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.config.stop:
            self.stop = OmegaConf.to_object(self.config.stop)

        self.need_value_func = self.config.need_value_func
        if self.need_value_func:
            self.reward_model = self.create_rm()
        self.llm = self.create_llm()
        if self.config.mode == "sbs":
            self.max_agent_steps = 1
        elif self.config.mode == "mcts":
            self.max_agent_steps = self.config.iterations
            self.config.step_beam_width = 1
            

    @field_validator("config")
    def validate_config(cls, cfg: Any):
        if issubclass(type(cfg), DictConfig):
            return cfg
        raise TypeError("Wrong type for `config`, must be subclass of BaseConfig")


    def create_rm(self):
        rm, v_head, tokenizer = rm_engine(self.config)
        return partial(
            rm_generate,
            model=rm,
            v_head=v_head,
            tokenizer=tokenizer,
            max_model_len=self.config.max_model_len,
        )


    def create_llm(self):
        engine, sampling_params = llm_engine(self.config)
        self.llm_engine = engine
        self.generate_sampling_params = sampling_params
        return partial(
            llm_generate,
            engine=self.llm_engine,
        )

        
    @staticmethod
    def processor(agent, output) -> BaseTree:
        agent.generate_next_step(output)
        return agent


    @staticmethod
    def selector(agent, output) -> BaseTree:
        agent.select_next_step(output)
        return agent


    def generate_preprocess(self, agents):
        prompts = []
        rewards = []
        prompts_span = [0]        # note: each agent represents a search for a question
                                  # what the prompts_span are actually used for? conjecture: keep track of the idxes of prompts
        valid_agents = []         # agents for the next expansion
        invalid_agents = []       # agents that have already built their trees, no more searching
        expanded_agents = []      # agents that have nodes that already expanded, skipping expansion, just do selection step

        for agent in agents:
            if agent.should_generate_next(): # True if agent.current_nodes[0] is not terminal node and its depths is less max_depth
                if agent.has_expanded():     # True if agent.current_nodes[0] has children -> already expanded
                    # print("agent.has_expanded")
                    expanded_agents.append(agent)
                else:
                    agent_prompts = agent.create_prompt()
                    rewards.extend(agent.get_rewards())
                    prompts.extend(agent_prompts)
                    prompts_span.append(prompts_span[-1] + len(agent_prompts))
                    valid_agents.append(agent)      
            else:
                invalid_agents.append(agent)
        # print(prompts)
        # print(rewards)
        # print(prompts_span)
        # stop
        return prompts, prompts_span, valid_agents, invalid_agents, expanded_agents, rewards


    def generate_postprocess(
        self, 
        outputs: List[List[RequestOutput]], 
        valid_agents: List[BaseTree],
    ) -> List[BaseTree]:
        '''
        # update leaf nodes of the current_nodes
        # add new generated nodes (not already a child and have visitation less than one) as children of the current_nodes
        '''
        
        post_agents = []
        #with ProcessPool(max_workers=min(len(valid_agents), os.cpu_count())) as pool:
        with ProcessPool(max_workers=12) as pool:
            future = pool.map(self.__class__.processor, valid_agents, outputs, timeout=TIMEOUT_SECONDS)
            iterator = future.result()
        
        progress_bar = tqdm(total=len(valid_agents), desc="generate_postprocess", disable=True)  
        while True:
            try:
                result = next(iterator)
                post_agents.append(result)
            except StopIteration:
                break
            except Exception as error:
                print(colored(f"{error}\n", ERROR_COLOR))
                post_agents.append(None)
            progress_bar.update(1) 
        progress_bar.close() 
            
        # update agents
        # why is this needed? it seems like post_agents are the same as valid_agents 
        updated_agents = [
            post_agent if post_agent is not None else valid_agent
            for post_agent, valid_agent in zip(post_agents, valid_agents)
        ]
        return updated_agents
    

    def value_preprocess(self, agents: List[BaseTree]) -> Tuple[List[str], List[int]]:
        prompts = []
        prompts_span = [0]
        for agent in agents:
            agent_prompts = agent.create_prompt(is_value_only=True)   # only create prompts for candidate_nodes 
            prompts.extend(agent_prompts)
            prompts_span.append(prompts_span[-1] + len(agent_prompts))
        # print(agent_prompts)
        return prompts, prompts_span
    
    
    def value_postprocess(
        self, 
        outputs, 
        valid_agents,
    ) -> List[BaseTree]:
        '''
        # select the next current_nodes for expansion
        '''
        for agent, output in zip(valid_agents, outputs):
            if agent is not None:
                self.selector(agent, output)
        return valid_agents
    

    def save_intermediate_metric(self, path: str, agents: List[MCTS], rollout) -> None:
        if self.config.is_sampling: return
        states = [s.intermediate_metric for s in agents]
        statics = []
        for i in range(rollout + 1):
            pass1, passn = 0, 0
            for idx, state in enumerate(states):
                max_value = -100
                max_value_result = False
                pass1_ans = False
                for idx, rollout_index in enumerate(state["rollout_indexs"]):
                    if rollout_index <= i:
                        if state["value_estimate"][idx] > max_value:
                            max_value = state["value_estimate"][idx]
                            max_value_result = state["judgements"][idx]
                        if state["judgements"][idx]:
                            pass1_ans = True
                if max_value_result:
                    pass1 += 1
                if pass1_ans:
                    passn += 1
            statics.append({
                "rollout": i,
                "pass1": pass1,
                "passn": passn,
                "len": len(states),
            })
        with open(path, "w", encoding='utf-8') as f:
            json.dump([statics,states], f, ensure_ascii=False, indent=4)

    
    def save_intermediate_rollouts(self, saved_jsonl_file, cur_data, agents, rollout_idx):
        if self.config.save_intermediate_rollouts and saved_jsonl_file and self.config.mode == "mcts":
            saved_json_dir = osp.dirname(saved_jsonl_file)
            saved_jsonl_file_name = osp.basename(saved_jsonl_file)
            saved_json_path = osp.join(saved_json_dir, f"rollout")
            if not os.path.exists(saved_json_path):
                os.mkdir(saved_json_path)
            self.save_intermediate_metric(osp.join(saved_json_path, f"intermediate_metric_{saved_jsonl_file_name}"), agents, rollout_idx)
            outs = self.output(agents)
            with open(osp.join(saved_json_path, f"rollout{rollout_idx:02}" + saved_jsonl_file_name), "a+", encoding='utf-8') as writer:
                for d in cur_data:
                    question = d["question"]
                    d["rstar"] = outs[question]
                    writer.write(json.dumps(d, ensure_ascii=False) + '\n')
                    writer.flush()
    
    def output(self, agents: List[BaseTree]):
        jsonlines = {}
        for i, agent in enumerate(agents):         
            jsonlines[agent.question] = agent.return_states() # outputs all the nodes/states within the tree
        
        return jsonlines
    
    def solve(self, agents: List[BaseTree], saved_jsonl_file: str, cur_data: List[Dict[str, Any]]):

        # max_agent_steps is number of rollouts or simulations 
        #    (default value is 12, the paper uses 4 rollouts for test-time compute) 
        for rollout in tqdm(range(self.max_agent_steps), desc="Rollout Processing", disable=True):
            # Initialize the initial search starting point of agents, 
            # and the initial point of each rollout is root
            for agent in agents:
                agent.select_next_step(from_root=True)      # This code resets current_nodes = [root]
                agent.rollout_idx = rollout
                # print(agent.current_nodes)

            # max_depth leads to max number of steps can be generated (# default value is 16, same in the paper).
            for step in range(self.config.max_depth):  
                print("-----------------Current Rollout: ", rollout, "-----------------")
                print("-----------------Current Step: ", step, "-----------------")

                print("\n-> current_agents")
                for agent in agents:
                    print(agent.current_nodes)

                # step expansion
                
                # valid_agents: agents for the next expansion
                # invalid_agents: agents that have already built their trees, no more searching
                # expanded_agents: agents in which their current_nodes already have expanded (i.e., already have children), 
                #     skipping expansion, just do selection step
                #     In what cases can this occur?
                
                # example of prompt 
                # "<|user|>:\nConvert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$\n\n<|assistant|>: Let's think step by step and solve the problem with code."
                
                # The prompt seems lack of some information, i.e., system prompt      
                # Here is a prompt template used in current experiment
                # Note: The generation stops with the special stop tokens ["<end_of_step>", "<end_of_code>", "<end_of_answer>"] 
                #  Todo: I have to check whether the current model Llama-3.2 has the these special stop tokens. 
                """
                <|start_header_id|>system<|end_header_id>

                Cutting Knowledge Date: December 2023
                Today Date: 26 Feb 2025
        
                Solve the following math problem efficiently and clearly:
        
                - For simple problems (2 steps or fewer):
                Provide a concise solution with minimal explanation.
        
                - For complex problems (3 steps or more):
                Use this step-by-step format:
        
                ## Step 1: [Concise description]
                [Brief explanation and calculations]
        
                ## Step 2: [Concise description]
                [Brief explanation and calculations]
        
                ...
        
                Regardless of the approach, always conclude with:
        
                Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.
        
                Where [answer] is just the final number or expression that solves the problem.
                <|eot_id|>
        
                <|start_header_id|>user<|end_header_id>
        
                Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  
                Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$
                <|eot_id|>
        
                <|start_header_id|>assistant<|end_header_id>
                """

                
                prompts, prompts_span, valid_agents, invalid_agents, expanded_agents, valid_rewards = self.generate_preprocess(agents)
                # Notes: 
                #  It is different from the Mario repo, which does't have the expanded_agents and valid_rewards
                #  I'm not so sure how valid_rewards are assigned to the nodes. 
                #  The valid_rewards does not seem to be the PRM scores.
                #  The default value of valid_rewards are 0
                #  I conjecture valid_rewards are only used in training.
                #  where we can get rewards from the tree built from previous round 
                
                print("\n-> llm prompts")
                print(prompts)
                print(valid_rewards)

                # print("\n-> valid_agents")
                # for agent in valid_agents:
                #     print(agent.current_nodes)

                # print("\n-> invalid_agents")
                # for agent in invalid_agents:
                #     print(agent.current_nodes)

                # print("\n-> expanded_agents")
                # for agent in expanded_agents:
                #     print(agent.current_nodes)

                # check if there are still agents that are 
                #  (i) valid for expansion 
                #  or (ii) already expanded and require the selection step 
                if len(valid_agents + expanded_agents) < 1:
                    break
                

                # generate batch of step nodes, the default value of batch_size is 32
                outputs = self.llm(prompts, self.generate_sampling_params)
                print("step_expansion outputs")
                print(outputs)  
                
                for output, reward in zip(outputs, valid_rewards): # attach reward to prevent repeat rewarding
                    output.value_estimate = reward
                reconstructed_outputs = [outputs[bos_idx : eos_idx] for bos_idx, eos_idx in zip(prompts_span, prompts_span[1:])]
                
                # process output 
                # update leaf nodes of the current_nodes
                # add new generated nodes (not already a child and have visitation less than one) as children of the current_nodes
                valid_agents = self.generate_postprocess(reconstructed_outputs, valid_agents)
                # step evaluation
                
                # what are the differences between this llm_prompts and rm_prompts?
                #   - the prompt is a dictionary that contains a empty 'prefix' and 'text' which is the 'prompt' 
                # why don't value/reward prompts contain the solution steps?
                # todo: how does the prm model actually works?
                
                # example of prompts: 
                # {'prefix': '', 'text': "<|user|>:\nConvert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$\n\n<|assistant|>: Let's think step by step and solve the problem with code."} 
                prompts, prompts_span = self.value_preprocess(valid_agents)
                print("\n-> value prompts")
                print(prompts)
                
                if self.need_value_func:  # always True 
                    # todo: check how the reward model works
                    outputs = self.reward_model(prompts=prompts)
                    print("\n-> value outputs")
                    print(outputs)
                    reconstructed_outputs = [outputs[bos_idx : eos_idx] for bos_idx, eos_idx in zip(prompts_span, prompts_span[1:])]
                else:
                    # print("False")
                    reconstructed_outputs = [None] * (len(prompts_span) - 1)
                
                # selection
                # select the next current_nodes for expansion
                # if no further nodes can be selected for next expansion 
                #     (i.e. the current_nodes[0] is a terminal node or its children are terminal nodes), 
                #     the rollout is finished. 
                print("\n-> valid_agents")
                valid_agents = self.value_postprocess(reconstructed_outputs, valid_agents)
                print("\n-> expanded_agents")
                for agent in expanded_agents:
                    print(agent.current_nodes)
                expanded_agents = self.value_postprocess([None] * len(expanded_agents), expanded_agents) # for expanded agents, just do selection step

                # print("\n-> invalid_agents")
                # for agent in invalid_agents:
                #     print(agent.current_nodes)

                # print("\n-> expanded_agents")
                # for agent in expanded_agents:
                #     print(agent.current_nodes)
                    
                # keep all agents
                agents = valid_agents + invalid_agents + expanded_agents

            # Save agents internal rollouts
            self.save_intermediate_rollouts(saved_jsonl_file, cur_data, agents, rollout)
            
        return self.output(agents)
    
    
